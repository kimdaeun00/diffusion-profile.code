#!/usr/bin/env python3
"""Summarise profile_diffusion.py result directories into markdown tables and a flat CSV.

    python scripts/summarize_profile.py results/profile                       # tables to stdout
    python scripts/summarize_profile.py results/profile --md results/profile/REPORT.md --csv results/profile/table.csv
    python scripts/summarize_profile.py results/profile_a6000 results/profile_h200 \\
        --price a6000=0.6 --price h200=4.0                                    # + speedup and $/1000 images

Reads every summary*.json in each directory (eager `summary.json`, compiled
`summary_compile-<mode>.json`). One table per (directory, model, variant); an
eager-vs-compiled speedup table when both variants exist; and, with two or more
directories, a cross-GPU table of ms per image, speedup vs the first directory.
Per profiled forward the --top N kernel names (launches merged) are listed with their
own Compute / DRAM / Tensor figures, read from the per-kernel CSV next to the summary
and -- if --price <label>=<$/h> is given for each -- $ per 1000 images with the
cheapest GPU marked. Labels default to the directory name's last path component.
No GPU needed.
"""
import argparse, csv, json, re, sys
from pathlib import Path

from collections import defaultdict

# ---- per-kernel CSV parsing (from ncu_util/report_flux2klein.py) ----
C = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
D = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
T = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
U = "gpu__time_duration.sum"
B = "dram__bytes.sum"
TO_MS = {"nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0, "second": 1e3}
TO_BYTES = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}


def classify(n):
    s = n.lower()
    if s.startswith("triton_tem"):                                        return "GEMM"
    if s.startswith(("triton_red", "triton_per")):                        return "norm" if "norm" in s else "elementwise"
    if s.startswith(("triton_poi", "triton_for")):                        return "elementwise"
    if any(k in s for k in ("flash", "fmha", "attention", "sdp")):        return "attention"
    if any(k in s for k in ("gemm", "gemv", "cutlass", "xmma", "cublas",
                            "ampere_", "sm80_", "sm86_", "splitk")):       return "GEMM"
    if "norm" in s:                                                        return "norm"
    if "reduce" in s or "softmax" in s:                                    return "reduce"
    if any(k in s for k in ("elementwise", "unrolled", "vectorized")):     return "elementwise"
    if any(k in s for k in ("copy", "cat", "index", "gather", "scatter",
                            "transpose", "fill")):                         return "copy/index"
    return "other"


def short(n):
    """Collapse a demangled kernel name to what it does, e.g.
    'cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>' -> 'cutlass_gemm[256x128_32x3]'
    'elementwise_kernel<..BinaryFunctor<float,..MulFunctor<float>>..>'         -> 'elementwise[mul f32]'."""
    n2 = n.replace("void ", "").replace("(anonymous namespace)", "").replace("<unnamed>", "")
    base = re.sub(r"<.*", "", n2).strip().split("(")[0].rstrip(":").split("::")[-1]
    m = re.search(r"Kernel2?<(cutlass_[^>]+)>", n2)
    if m:
        tile = re.search(r"(\d+x\d+_\d+x\d+)", m.group(1))
        return f"cutlass_gemm[{tile.group(1) if tile else m.group(1)}]"
    if re.match(r"(ampere|sm\d+|turing|volta)_.*gemm", base):
        tile = re.search(r"gemm_\w+?_(\d+x\d+)", base)
        return f"cublas_gemm[{tile.group(1)}]" if tile else base
    if "flash_fwd" in base or "fmha" in base:
        return base
    # elementwise family: name the functor and the dtype it runs in
    ops = [f for f in re.findall(r"(\w+?)Functor(?:_(\w+))?\b", n2)]
    op = None
    for a, b in reversed(ops):
        if a in ("CUDA",) and b: op = b; break
        if a not in ("Binary", "Unary", "Ternary", "AUnary", "BUnary", "CUDA"): op = a; break
    if op is None:
        m = re.search(r"(silu|gelu|rms_?norm|layer_norm|softmax|sigmoid|arange|fill|where|"
                      r"direct_copy|CatArray|index|gather|scatter)", n2, re.I)
        op = m.group(1).lower() if m else None
        if op == "direct_copy": op = "copy+cast" if "WithCast" in n2 else "copy"
        if op == "catarray": op = "cat"
    dt = re.search(r"Functor<(float|c10::BFloat16|c10::Half)", n2) or \
         re.search(r"\[lambda\((float|c10::BFloat16|c10::Half)\)", n2) or \
         re.search(r"kernel<(c10::BFloat16|float|c10::Half)", n2)
    dts = {"float": "f32", "c10::BFloat16": "bf16", "c10::Half": "f16"}.get(dt.group(1), "") if dt else ""
    fam = base.replace("_kernel", "").replace("vectorized_", "vec_").replace("unrolled_", "unr_")
    if op:
        return f"{fam}[{op.lower()}{' ' + dts if dts else ''}]"
    return fam


def load(path):
    rows = list(csv.reader(open(path)))
    if len(rows) < 3:
        return []
    hdr, units = rows[0], rows[1]
    ix = {c: hdr.index(c) for c in (C, D, T, U, B) if c in hdr}
    ni = hdr.index("Kernel Name")
    gi = hdr.index("Grid Size") if "Grid Size" in hdr else None
    bi = hdr.index("Block Size") if "Block Size" in hdr else None
    ms = TO_MS[units[ix[U]].strip()]
    bs = TO_BYTES[units[ix[B]].strip()] if B in ix else 1.0

    def f(r, c):
        try: return float(r[ix[c]])
        except (ValueError, KeyError, IndexError): return None

    ks = []
    for r in rows[2:]:
        d = f(r, U)
        if d is None:
            continue
        ks.append(dict(name=r[ni], short=short(r[ni]), cls=classify(r[ni]), ms=d * ms,
                       c=f(r, C), d=f(r, D), t=f(r, T), by=(f(r, B) or 0.0) * bs,
                       grid=r[gi] if gi is not None else "", block=r[bi] if bi is not None else ""))
    return ks


def wmean(ks, key):
    num = sum(k[key] * k["ms"] for k in ks if k[key] is not None)
    den = sum(k["ms"] for k in ks if k[key] is not None)
    return num / den if den else float("nan")


def group(ks, key):
    g = defaultdict(list)
    for k in ks:
        g[k[key]].append(k)
    tot = sum(k["ms"] for k in ks)
    out = []
    for name, items in g.items():
        ms = sum(k["ms"] for k in items)
        out.append(dict(name=name, n=len(items), ms=ms, pct=100 * ms / tot,
                        c=wmean(items, "c"), d=wmean(items, "d"), t=wmean(items, "t"),
                        gbps=sum(k["by"] for k in items) / (ms / 1e3) / 1e9 if ms else 0.0,
                        example=max(items, key=lambda k: k["ms"])))
    return sorted(out, key=lambda x: -x["ms"])


ncu_load, ncu_group = load, group


def load_dir(d: Path):
    """-> list of records with dir/label/gpu/variant added."""
    out = []
    for p in sorted(d.glob("summary*.json")):
        s = json.load(open(p))
        for key, recs in s.get("models", {}).items():
            model = key.split("/")[0]
            variant = key.split("/")[1] if "/" in key else "eager"
            for r in recs:
                r = dict(r); r.update(model=model, variant=variant, gpu=s.get("gpu", ""), date=s.get("date", ""), dir=str(d))
                out.append(r)
    return out


def f(v, d=0):
    return "–" if v is None else f"{v:,.{d}f}"


def latency_table(recs):
    lines = ["| res | batch | total ms | text ms | transformer ms | VAE ms | peak GB |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda r: (r["res"], r["batch"])):
        if r.get("oom"):
            lines.append(f"| {r['res']}² | {r['batch']} | OOM | | | | |"); continue
        lines.append(f"| {r['res']}² | {r['batch']} | {f(r['total_ms'])} | {f(r['text_ms'])} | {f(r['dit_ms'])} | {f(r['vae_ms'])} | {f(r['peak_gb'],1)} |")
    return lines


def top_kernels_table(csv_path: Path, top: int):
    """Top kernel names of one profiled forward by summed duration, each with its own SoL figures."""
    if not ncu_load or not csv_path.exists():
        return []
    ks = ncu_load(csv_path)
    if not ks:
        return []
    tot = sum(k["ms"] for k in ks)
    lines = ["| # | kernel | ms | share | Compute % | DRAM % | GB/s |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for i, g in enumerate(ncu_group(ks, "short")[:top], 1):
        lines.append(f"| {i} | `{g['name']}` | {g['ms']:.2f} | {g['pct']:.1f}% | {g['c']:.1f} | {g['d']:.1f} | {g['gbps']:.0f} |")
    rest = ncu_group(ks, "short")[top:]
    if rest:
        lines.append(f"| | ({len(rest)} more names) | {sum(g['ms'] for g in rest):.2f} | {100 * sum(g['ms'] for g in rest) / tot:.1f}% | | | |")
    return lines


def pick(recs, model, variant, res, batch):
    return next((r for r in recs if r["model"] == model and r["variant"] == variant and r["res"] == res and r["batch"] == batch), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--label", action="append", default=[], help="dir label override: <dirname>=<label>")
    ap.add_argument("--price", action="append", default=[], help="<label>=<$ per hour> for the cost table")
    ap.add_argument("--top", type=int, default=3, help="kernels per profiled forward to list (0 = none)")
    ap.add_argument("--md", type=Path, help="also write the markdown here")
    ap.add_argument("--csv", type=Path, help="also write a flat CSV (one row per record) here")
    a = ap.parse_args()

    labels = {d: d.name for d in a.dirs}
    for kv in a.label:
        k, v = kv.split("=", 1); labels[next(d for d in a.dirs if d.name == k)] = v
    prices = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.price)}

    data = {labels[d]: load_dir(d) for d in a.dirs}
    for lab, recs in data.items():
        if not recs: sys.exit(f"{lab}: no summary*.json found")
    out = []
    w = out.append

    # ---- per directory
    for lab, recs in data.items():
        gpu = recs[0]["gpu"]; date = max(r["date"] for r in recs)
        w(f"# {lab} — {gpu} ({date})\n")
        for model in sorted({r["model"] for r in recs}):
            for variant in sorted({r["variant"] for r in recs if r["model"] == model}):
                rs = [r for r in recs if r["model"] == model and r["variant"] == variant]
                w(f"## {model} · {variant}\n"); out.extend(latency_table(rs)); w("")
                if a.top:
                    d = next(d for d in a.dirs if labels[d] == lab)
                    for r in sorted(rs, key=lambda r: r["res"]):
                        if r["batch"] != 1 or not r.get("ncu"): continue
                        csvp = d / f"{model}{'' if variant == 'eager' else '_' + variant}_r{r['res']}.csv"
                        t = top_kernels_table(csvp, a.top)
                        if t:
                            w(f"### top {a.top} kernels — {model} · {variant} · {r['res']}² · batch 1 "
                              f"({r['ncu']['kernels']} launches, {r['ncu']['kernel_ms']:.1f} ms)\n"); out.extend(t); w("")
            variants = sorted({r["variant"] for r in recs if r["model"] == model})
            if "eager" in variants and len(variants) > 1:
                for variant in (v for v in variants if v != "eager"):
                    w(f"### {model}: eager → {variant}\n")
                    w("| res | batch | ms / image eager | ms / image compiled | speedup | per call eager | per call compiled | kernel ms eager | kernel ms compiled | launches |")
                    w("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                    for res in sorted({r["res"] for r in recs}):
                        for b in sorted({r["batch"] for r in recs}):
                            e, c = pick(recs, model, "eager", res, b), pick(recs, model, variant, res, b)
                            if not e or not c or e.get("oom") or c.get("oom"): continue
                            ne, nc = e.get("ncu") or {}, c.get("ncu") or {}
                            w(f"| {res}² | {b} | {f(e['ms_per_image'])} | {f(c['ms_per_image'])} | {e['ms_per_image']/c['ms_per_image']:.2f}× | "
                              f"{f(e['dit_call_ms'],1)} | {f(c['dit_call_ms'],1)} | {f(ne.get('kernel_ms'),1)} | {f(nc.get('kernel_ms'),1)} | "
                              f"{f(ne.get('kernels'))} → {f(nc.get('kernels'))} |")
                    w("")

    # ---- across directories (GPUs)
    labs = list(data)
    if len(labs) > 1:
        base = labs[0]
        w(f"# Across GPUs — ms per image at batch 1 (speedup vs {base})\n")
        hdr = "| model | variant | res | " + " | ".join(f"{l} ms/img" for l in labs) + " | " + " | ".join(f"{l} ×" for l in labs[1:]) + " |"
        w(hdr); w("|" + "---:|" * (hdr.count("|") - 1))
        keys = sorted({(r["model"], r["variant"], r["res"]) for recs in data.values() for r in recs})
        rows = []
        for model, variant, res in keys:
            vals = [pick(data[l], model, variant, res, 1) for l in labs]
            ms = [v["ms_per_image"] if v and not v.get("oom") else None for v in vals]
            rows.append((model, variant, res, ms))
            w(f"| {model} | {variant} | {res}² | " + " | ".join(f(m) for m in ms) + " | " +
              " | ".join(f"{ms[0]/m:.2f}" if ms[0] and m else "–" for m in ms[1:]) + " |")
        w("")
        if all(l in prices for l in labs):
            w("# Cost — $ per 1000 images at batch 1 (" + ", ".join(f"{l} ${prices[l]}/h" for l in labs) + ")\n")
            w("| model | variant | res | " + " | ".join(labs) + " | cheapest |"); w("|" + "---:|" * (len(labs) + 4))
            for model, variant, res, ms in rows:
                cost = [prices[l] * m / 3.6e6 * 1000 if m else None for l, m in zip(labs, ms)]
                best = min((c, l) for c, l in zip(cost, labs) if c is not None)[1] if any(cost) else "–"
                w(f"| {model} | {variant} | {res}² | " + " | ".join("–" if c is None else f"{c:,.2f}" for c in cost) + f" | **{best}** |")
            w("")

    text = "\n".join(out)
    print(text)
    if a.md:
        a.md.write_text(text); print(f"\nwrote {a.md}", file=sys.stderr)
    if a.csv:
        fields = ["label", "gpu", "date", "model", "variant", "res", "batch", "oom", "total_ms", "ms_per_image", "images_per_s", "text_ms",
                  "dit_ms", "dit_calls", "dit_call_ms", "vae_ms", "peak_gb", "busy_pct",
                  "ncu_kernels", "ncu_kernel_ms", "ncu_compute", "ncu_dram", "ncu_tensor", "ncu_gbps",
                  "pct_GEMM", "pct_attention", "pct_elementwise", "pct_norm"]
        with open(a.csv, "w", newline="") as fh:
            cw = csv.DictWriter(fh, fieldnames=fields); cw.writeheader()
            for lab, recs in data.items():
                for r in sorted(recs, key=lambda r: (r["model"], r["variant"], r["res"], r["batch"])):
                    n = r.get("ncu") or {}
                    row = {k: r.get(k) for k in fields if k in r}
                    row.update(label=lab, oom=bool(r.get("oom")), ncu_kernels=n.get("kernels"), ncu_kernel_ms=n.get("kernel_ms"),
                               ncu_compute=n.get("compute"), ncu_dram=n.get("dram"), ncu_tensor=n.get("tensor"), ncu_gbps=n.get("gbps"))
                    for c in ("GEMM", "attention", "elementwise", "norm"):
                        row[f"pct_{c}"] = n.get("classes", {}).get(c, {}).get("pct")
                    cw.writerow(row)
        print(f"wrote {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
