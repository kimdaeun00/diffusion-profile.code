#!/usr/bin/env python3
"""Latency + compute / bandwidth utilization of SD3-medium and FLUX.2-klein-9B, in one run.

Two phases per (model, resolution), merged into one table and results/profile/summary.json:

  1. latency  -- unprofiled generation, N repeats. CUDA-event timers on the three
     stages (text encoder, transformer forward x steps, VAE decode), wall time,
     peak memory, and a coarse GPU busy % sampled from nvidia-smi.
  2. ncu      -- (--ncu) one mid-loop transformer forward per resolution under
     Nsight Compute (application replay, NVTX-filtered): duration-weighted
     Compute / DRAM / Tensor speed-of-light and the GEMM / attention / elementwise
     time split. ncu locks clocks and serialises kernels, so its milliseconds are
     not latency -- that is why phase 1 exists.

    CUDA_VISIBLE_DEVICES=0 python3 profile_diffusion.py                               # latency only, both models
    CUDA_VISIBLE_DEVICES=0 python3 profile_diffusion.py --models sd3 --sizes 512,1024
    CUDA_VISIBLE_DEVICES=0 python3 profile_diffusion.py --ncu                          # + ncu, ~8-15 min per (model,res)
    CUDA_VISIBLE_DEVICES=0 python3 profile_diffusion.py --batch 1,2,4                   # num_images_per_prompt sweep
    CUDA_VISIBLE_DEVICES=0 python3 profile_diffusion.py --compile                       # torch.compile'd transformer

--compile runs the transformer forward through torch.compile (Inductor): pointwise
chains (adaLN modulation, residual adds, RoPE, casts, SiLU) become fused Triton
kernels and norms become Triton reductions; GEMM and attention keep their cuBLAS /
flash kernels. --compile-mode reduce-overhead adds CUDA graphs, which also removes
launch gaps. Each (res, batch) shape compiles once during its warmup (1-3 min for
the 9B model); results are stored under a separate '<model>_compile-<mode>' name.
Batch = num_images_per_prompt; SD3's CFG doubles the DiT batch on top of it. A
configuration that OOMs is recorded as such and skipped. ncu results are cached per
(model, res, batch) in --out; delete the .csv to re-measure. Weights come from the HF
cache and are downloaded on first use if missing.
Internal: `--driver MODEL --res N` is the process ncu launches; not for direct use.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODELS = {
    "sd3": dict(repo="stabilityai/stable-diffusion-3-medium-diffusers", steps=28, guidance=7.0,
                max_seq=256, label="SD3-medium (2B MM-DiT, 28 steps, CFG 7.0 -> DiT batch 2)"),
    "flux2klein": dict(repo="black-forest-labs/FLUX.2-klein-9B", steps=4, guidance=1.0,
                       max_seq=512, label="FLUX.2-klein-9B (distilled, 4 steps, guidance 1.0)"),
}
PROMPT = ("A vibrant blue-and-yellow macaw perched on a weathered wooden fence, surrounded by a lush "
          "tropical garden. Soft warm sunlight, golden glow, wide-angle view, highly detailed.")
NCU_METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum", "gpu__time_duration.sum"])


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- pipelines
COMPILE_MODE = None       # None = eager; else a torch.compile mode string


def load_pipeline(model: str, dtype, drop_text_encoder: bool = False):
    """Return (pipe, extra_call_kwargs). With drop_text_encoder the prompt is encoded
    once and the encoder is unregistered (FLUX: frees 16 GB; and .to('cpu') would
    make DiffusionPipeline._execution_device CPU, since it reads the first module)."""
    import torch
    cfg = MODELS[model]
    if model == "sd3":
        from diffusers import StableDiffusion3Pipeline as cls
    else:
        from diffusers import Flux2KleinPipeline as cls
    try:                       # cached weights: no hub round-trip
        pipe = cls.from_pretrained(cfg["repo"], torch_dtype=dtype, local_files_only=True)
    except Exception as e:     # not (fully) in the cache: download, then fail for real if it still fails
        log(f"[{model}] {cfg['repo']} not in the HF cache ({type(e).__name__}); downloading")
        pipe = cls.from_pretrained(cfg["repo"], torch_dtype=dtype, local_files_only=False)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    extra = {}
    if drop_text_encoder and model == "flux2klein":
        import gc
        with torch.no_grad():
            embeds, _ = pipe.encode_prompt(PROMPT, device="cuda", max_sequence_length=cfg["max_seq"])
        te = pipe.text_encoder
        pipe.register_modules(text_encoder=None)
        del te
        gc.collect(); torch.cuda.empty_cache()
        extra = dict(prompt=None, prompt_embeds=embeds)
        assert str(pipe._execution_device).startswith("cuda")
    if COMPILE_MODE:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64          # one graph per (res, batch) shape
        # compile the bound forward, not the module: pipe.transformer stays a plain nn.Module
        # (cache_context etc. intact) and the timing / NVTX wrappers stack on top of it.
        pipe.transformer.forward = torch.compile(pipe.transformer.forward, mode=COMPILE_MODE, dynamic=False)
        log(f"[{model}] transformer.forward compiled (mode={COMPILE_MODE}); each new shape compiles in its warmup")
    return pipe, extra


def generate(pipe, model: str, res: int, steps: int, extra: dict, seed: int = 0, batch: int = 1):
    import torch
    cfg = MODELS[model]
    kw = dict(prompt=PROMPT, height=res, width=res, num_inference_steps=steps,
              guidance_scale=cfg["guidance"], max_sequence_length=cfg["max_seq"],
              num_images_per_prompt=batch,
              generator=torch.Generator(device="cuda").manual_seed(seed))
    kw.update(extra)
    return pipe(**kw)


class StageTimer:
    """CUDA-event timing around encode_prompt / transformer.forward / vae.decode.
    Events are recorded on the stream, so nothing here synchronises the GPU."""

    def __init__(self, pipe):
        import torch
        self.torch = torch
        self.events: dict[str, list] = {"text": [], "dit": [], "vae": []}
        self._originals = []
        self._wrap(pipe, "encode_prompt", "text")
        self._wrap(pipe.transformer, "forward", "dit")
        self._wrap(pipe.vae, "decode", "vae")

    def restore(self):
        for obj, name, orig in self._originals:
            setattr(obj, name, orig)
        self._originals.clear()

    def _wrap(self, obj, name, stage):
        orig = getattr(obj, name)
        self._originals.append((obj, name, orig))
        torch = self.torch

        def wrapped(*a, **k):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            try:
                return orig(*a, **k)
            finally:
                e.record(); self.events[stage].append((s, e))
        setattr(obj, name, wrapped)

    def reset(self):
        for v in self.events.values(): v.clear()

    def ms(self, stage):  # call after torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in self.events[stage]]


class SmiSampler:
    """nvidia-smi utilization.gpu (fraction of wall time with a kernel resident) at 100 ms.
    Coarse -- the driver's own window is 1/6..1 s -- but it catches launch gaps that
    ncu's per-kernel view cannot."""

    def __init__(self):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        self.gpu = vis; self.proc = None; self.file = None

    def start(self):
        if not shutil.which("nvidia-smi"): return
        self.file = tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False)
        self.proc = subprocess.Popen(
            ["nvidia-smi", "-i", self.gpu, "--query-gpu=utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits", "-lms", "100"], stdout=self.file, stderr=subprocess.DEVNULL)

    def stop(self):
        if not self.proc: return None
        self.proc.terminate(); self.proc.wait()
        self.file.flush(); self.file.seek(0)
        rows = [ln.split(",") for ln in self.file.read().splitlines() if "," in ln]
        os.unlink(self.file.name); self.proc = None
        rows = rows[1:] or rows                         # first sample may predate the run
        if not rows: return None
        return dict(busy=sum(float(r[0]) for r in rows) / len(rows),
                    memctl=sum(float(r[1]) for r in rows) / len(rows), samples=len(rows))


# ----------------------------------------------------------------------------- phase 1: latency
def variant():
    return f"compile-{COMPILE_MODE}" if COMPILE_MODE else "eager"


def latency_phase(model, sizes, batches, repeats, warmup_steps, out: Path, save_images):
    import torch
    cfg = MODELS[model]
    dtype = torch.bfloat16
    log(f"[{model}] loading {cfg['repo']}")
    t0 = time.perf_counter()
    pipe, extra = load_pipeline(model, dtype)
    timer = StageTimer(pipe)
    log(f"[{model}] loaded in {time.perf_counter() - t0:.1f}s, resident {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    results = []
    for res in sizes:
        for batch in batches:
            rec = dict(model=model, variant=variant(), res=res, batch=batch, steps=cfg["steps"], repeats=repeats)
            try:
                generate(pipe, model, res, min(warmup_steps, cfg["steps"]), extra, batch=batch)  # autotune at this shape
                torch.cuda.synchronize()
                runs = []
                sampler = SmiSampler(); sampler.start()
                for r in range(repeats):
                    timer.reset(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    t = time.perf_counter()
                    out_img = generate(pipe, model, res, cfg["steps"], extra, seed=r, batch=batch)
                    torch.cuda.synchronize()
                    total = (time.perf_counter() - t) * 1e3
                    dit = timer.ms("dit")
                    runs.append(dict(total_ms=total, text_ms=sum(timer.ms("text")), dit_ms=sum(dit),
                                     dit_calls=len(dit), dit_call_ms=sum(dit) / max(len(dit), 1),
                                     vae_ms=sum(timer.ms("vae")), peak_gb=torch.cuda.max_memory_allocated() / 2**30))
                    if save_images and r == 0:
                        out_img.images[0].save(out / f"{model}{"" if not COMPILE_MODE else "_" + variant()}_r{res}_b{batch}.png")
                smi = sampler.stop()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:   # compiled code may surface OOM as RuntimeError
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                rec.update(oom=True, total_ms=None)
                results.append(rec)
                log(f"[{model}] {res}x{res} batch {batch}: OOM, skipped")
                continue
            mean = lambda k: sum(x[k] for x in runs) / len(runs)
            rec.update(oom=False, total_ms=mean("total_ms"), ms_per_image=mean("total_ms") / batch,
                       images_per_s=batch / (mean("total_ms") / 1e3),
                       text_ms=mean("text_ms"), dit_ms=mean("dit_ms"), dit_calls=runs[0]["dit_calls"],
                       dit_call_ms=mean("dit_call_ms"), vae_ms=mean("vae_ms"),
                       peak_gb=max(x["peak_gb"] for x in runs), busy_pct=smi["busy"] if smi else None,
                       memctl_pct=smi["memctl"] if smi else None, runs=runs)
            results.append(rec)
            log(f"[{model}] {res}x{res} batch {batch}: total {rec['total_ms']:.0f} ms ({rec['ms_per_image']:.0f} ms/img) = "
                f"text {rec['text_ms']:.0f} + DiT {rec['dit_ms']:.0f} ({rec['dit_calls']} x {rec['dit_call_ms']:.1f}) + "
                f"VAE {rec['vae_ms']:.0f}; peak {rec['peak_gb']:.1f} GB; busy {rec['busy_pct'] if smi else float('nan'):.0f}%")
    # Release for real: the timing wrappers form a module <-> closure cycle and dynamo's
    # cache pins the compiled graph's parameters, so a bare `del pipe` leaves the whole
    # model resident -- and the ncu child processes then OOM against this process.
    import gc, torch._dynamo
    timer.restore()
    del pipe, timer, extra
    torch._dynamo.reset()
    gc.collect(); torch.cuda.empty_cache()
    log(f"[{model}] released; {torch.cuda.memory_allocated() / 2**30:.2f} GB still allocated in this process")
    return results


# ----------------------------------------------------------------------------- phase 2: ncu
def driver(model, res, steps, target_step, batch=1):
    """The process ncu launches: warm up, then one NVTX-wrapped transformer forward."""
    import torch
    cfg = MODELS[model]
    steps = steps or cfg["steps"]
    pipe, extra = load_pipeline(model, torch.bfloat16, drop_text_encoder=True)
    log(f"[driver {model} r{res}] resident {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    generate(pipe, model, res, 1, extra, batch=batch); torch.cuda.synchronize()
    orig = pipe.transformer.forward; state = {"i": 0}

    def wrapped(*a, **k):
        i = state["i"]; state["i"] += 1
        if i != target_step:
            return orig(*a, **k)
        hs = k.get("hidden_states", a[0] if a else None)
        log(f"[driver {model} r{res} b{batch}] PROFILING forward #{i} hidden_states={tuple(hs.shape)}")
        torch.cuda.nvtx.range_push("profile_region")
        try:
            return orig(*a, **k)
        finally:
            torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
    pipe.transformer.forward = wrapped
    generate(pipe, model, res, steps, extra, batch=batch); torch.cuda.synchronize()
    log(f"[driver {model} r{res}] done, {state['i']} forwards")


def classify(n):
    s = n.lower()
    # Inductor names a fused kernel after every op it absorbed, including neighbours such as
    # "scaled_dot_product_flash_attention" -- so decide Triton kernels by their prefix first.
    if s.startswith("triton_tem"):                                        return "GEMM"
    if s.startswith(("triton_red", "triton_per")):                        return "norm" if "norm" in s else "elementwise"
    if s.startswith(("triton_poi", "triton_for")):                        return "elementwise"
    if any(k in s for k in ("flash", "fmha", "attention", "sdp")): return "attention"
    if any(k in s for k in ("gemm", "gemv", "cutlass", "xmma", "cublas", "nvjet", "ampere_", "sm80_", "sm86_",
                            "sm90_", "sm100_", "sm120_", "umma", "tcgen05", "splitk")): return "GEMM"
    if "triton_tem" in s: return "GEMM"                                   # Inductor template matmul
    if "norm" in s: return "norm"
    if any(k in s for k in ("elementwise", "unrolled", "vectorized", "reduce", "softmax",
                            "triton_poi", "triton_red", "triton_per", "triton_for")): return "elementwise"
    return "other"


def parse_ncu_csv(path: Path):
    TO_MS = {"nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0, "second": 1e3}
    TO_B = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}
    C, D, T, B, U = NCU_METRICS.split(",")[:5]   # order in NCU_METRICS: bytes before duration
    rows = list(csv.reader(open(path)))
    if len(rows) < 3: return None
    hdr, units = rows[0], rows[1]
    ix = {c: hdr.index(c) for c in (C, D, T, U, B) if c in hdr}
    ni = hdr.index("Kernel Name")
    ms_s, b_s = TO_MS[units[ix[U]].strip()], TO_B[units[ix[B]].strip()]
    ks = []
    for r in rows[2:]:
        try: d = float(r[ix[U]]) * ms_s
        except ValueError: continue
        f = lambda c: float(r[ix[c]]) if r[ix[c]] not in ("", "n/a") else None
        ks.append(dict(cls=classify(r[ni]), ms=d, c=f(C), d=f(D), t=f(T), by=(f(B) or 0) * b_s))
    tot = sum(k["ms"] for k in ks)
    w = lambda items, key: (sum(k[key] * k["ms"] for k in items if k[key] is not None) /
                            max(sum(k["ms"] for k in items if k[key] is not None), 1e-9))
    classes = {}
    for c in ("GEMM", "attention", "elementwise", "norm", "other"):
        it = [k for k in ks if k["cls"] == c]
        if it: classes[c] = dict(n=len(it), ms=sum(k["ms"] for k in it), pct=100 * sum(k["ms"] for k in it) / tot,
                                 compute=w(it, "c"), dram=w(it, "d"), tensor=w(it, "t"))
    return dict(kernels=len(ks), kernel_ms=tot, compute=w(ks, "c"), dram=w(ks, "d"), tensor=w(ks, "t"),
                gbps=sum(k["by"] for k in ks) / (tot / 1e3) / 1e9, classes=classes)


def ncu_phase(model, sizes, batches, out: Path, steps, target_step, force):
    ncu = shutil.which("ncu")
    if not ncu: raise SystemExit("ncu not on PATH")
    res_out = {}
    for res in sizes:
        for batch in batches:
            tag = f"{model}" + (f"_{variant()}" if COMPILE_MODE else "") + f"_r{res}" + (f"_b{batch}" if batch != 1 else "")
            rep, csvp, logp = out / f"{tag}.ncu-rep", out / f"{tag}.csv", out / f"{tag}.ncu.log"
            if csvp.exists() and csvp.stat().st_size > 0 and not force:
                log(f"[ncu {tag}] cached {csvp}")
            else:
                log(f"[ncu {tag}] profiling ({time.strftime('%H:%M:%S')}) -> {logp}")
                t0 = time.perf_counter()
                cmd = [ncu, "--nvtx", "--nvtx-include", "profile_region/", "--metrics", NCU_METRICS,
                       "--replay-mode", "application", "--app-replay-mode", "relaxed",
                       "--export", str(rep), "--force-overwrite",
                       sys.executable, os.path.abspath(__file__), "--driver", model, "--res", str(res),
                       "--batch", str(batch), "--target-step", str(target_step)] + \
                      (["--steps", str(steps)] if steps else []) + \
                      (["--compile", "--compile-mode", COMPILE_MODE] if COMPILE_MODE else [])
                with open(logp, "w") as lf:
                    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
                if rc != 0:
                    log(f"[ncu {tag}] FAILED rc={rc}, see {logp}"); continue
                with open(csvp, "w") as cf:
                    subprocess.call([ncu, "--import", str(rep), "--csv", "--page", "raw"], stdout=cf, stderr=subprocess.DEVNULL)
                log(f"[ncu {tag}] {time.perf_counter() - t0:.0f}s")
            parsed = parse_ncu_csv(csvp)
            if parsed: res_out[(res, batch)] = parsed
    return res_out


# ----------------------------------------------------------------------------- report
def report(model, lat, ncu_res):
    cfg = MODELS[model]
    print(f"\n== {cfg['label']}  [{variant()}]")
    hdr = (f"{'res':>5} {'batch':>5} {'total ms':>9} {'ms/img':>7} {'img/s':>6} {'text':>6} {'DiT':>8} {'calls':>5} "
           f"{'/call':>7} {'VAE':>6} {'peak GB':>7} {'busy%':>6}")
    if ncu_res:
        hdr += f" | {'kern ms':>8} {'Comp%':>6} {'DRAM%':>6} {'Tens%':>6} {'GB/s':>5} {'GEMM':>5} {'attn':>5} {'elem':>5}"
    print(hdr); print("-" * len(hdr))
    for r in lat:
        if r.get("oom"):
            print(f"{r['res']:>5} {r['batch']:>5} {'OOM':>9}"); continue
        line = (f"{r['res']:>5} {r['batch']:>5} {r['total_ms']:>9.0f} {r['ms_per_image']:>7.0f} {r['images_per_s']:>6.2f} "
                f"{r['text_ms']:>6.0f} {r['dit_ms']:>8.0f} {r['dit_calls']:>5} {r['dit_call_ms']:>7.1f} {r['vae_ms']:>6.0f} "
                f"{r['peak_gb']:>7.1f} {(r['busy_pct'] if r['busy_pct'] is not None else float('nan')):>6.0f}")
        n = (ncu_res or {}).get((r["res"], r["batch"]))
        if n:
            g = lambda c: n["classes"].get(c, {}).get("pct", 0.0)
            line += (f" | {n['kernel_ms']:>8.1f} {n['compute']:>6.1f} {n['dram']:>6.1f} {n['tensor']:>6.1f} "
                     f"{n['gbps']:>5.0f} {g('GEMM'):>4.0f}% {g('attention'):>4.0f}% {g('elementwise'):>4.0f}%")
        elif ncu_res:
            line += " | (no ncu result)"
        print(line)
    print("  latency: wall-clock, unprofiled; text/DiT/VAE are CUDA-event GPU time of that stage, DiT = all transformer "
          "calls; batch = num_images_per_prompt (SD3's CFG doubles the DiT batch).\n  ncu: one transformer forward, "
          "duration-weighted speed-of-light; kern ms is at locked clocks, not latency. busy% is nvidia-smi utilization.gpu.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="sd3,flux2klein")
    p.add_argument("--sizes", default="512,1024,2048")
    p.add_argument("--batch", default="1", help="num_images_per_prompt list for the latency phase, e.g. 1,2,4")
    p.add_argument("--ncu-batch", default="1", help="batch list for the ncu phase (default: 1 only)")
    p.add_argument("--compile", action="store_true", help="torch.compile the transformer forward (fused elementwise)")
    p.add_argument("--compile-mode", default="default",
                   help="torch.compile mode: default | reduce-overhead (CUDA graphs) | max-autotune-no-cudagraphs")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--ncu", action="store_true", help="also run the Nsight Compute phase")
    p.add_argument("--ncu-steps", type=int, default=None, help="steps in the ncu driver run (default: model's)")
    p.add_argument("--target-step", type=int, default=2, help="0-based transformer call to profile")
    p.add_argument("--force", action="store_true", help="re-run ncu even if cached")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/profile"))
    p.add_argument("--driver", default=None, help=argparse.SUPPRESS)
    p.add_argument("--res", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    global COMPILE_MODE
    COMPILE_MODE = args.compile_mode if args.compile else None

    if args.driver:
        driver(args.driver, args.res, args.steps, args.target_step, int(args.batch)); return

    import torch
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    batches = [int(b) for b in args.batch.split(",") if b.strip()]
    ncu_batches = [int(b) for b in args.ncu_batch.split(",") if b.strip()]
    for m in models:
        if m not in MODELS: raise SystemExit(f"unknown model {m}; choose from {list(MODELS)}")
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for model in models:
        lat_path = args.out / (f"{model}_latency.json" if not COMPILE_MODE else f"{model}_{variant()}_latency.json")
        if args.skip_latency and lat_path.exists():
            lat = [r for r in json.load(open(lat_path)) if r["res"] in sizes and r.get("batch", 1) in batches]
        else:
            lat = latency_phase(model, sizes, batches, args.repeats, args.warmup_steps, args.out, args.save_images)
            json.dump(lat, open(lat_path, "w"), indent=1)
        ncu_res = ncu_phase(model, sizes, ncu_batches, args.out, args.ncu_steps, args.target_step, args.force) if args.ncu else None
        for r in lat:
            r.pop("runs", None); r["ncu"] = (ncu_res or {}).get((r["res"], r.get("batch", 1)))
        summary[f"{model}" + (f"/{variant()}" if COMPILE_MODE else "")] = lat
        report(model, lat, ncu_res)
    sp = args.out / ("summary.json" if not COMPILE_MODE else f"summary_{variant()}.json")
    merged = json.load(open(sp)) if sp.exists() else {}
    merged.update(gpu=torch.cuda.get_device_name(0), date=time.strftime("%Y-%m-%d %H:%M"))
    merged.setdefault("models", {}).update(summary)          # partial re-runs keep the other models
    json.dump(merged, open(sp, "w"), indent=1)
    print(f"\nwrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
