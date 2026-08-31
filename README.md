# Profiling SD3-medium and FLUX.2-klein-9B on one GPU

One command measures both models at 512² / 1024² / 2048² and batch 1 / 2 / 4 with the
transformer under `torch.compile`: end-to-end latency with a text-encoder / transformer /
VAE split, then one transformer forward per resolution under Nsight Compute for
compute / DRAM utilization. Everything lands in `results/profile/`.

## 1. Environment

```bash
cd docker/
# set <hf-repository> before launch. 
# <hf-repository> should contain `stabilityai/stable-diffusion-3-medium-diffusers` (15 GB) and `black-forest-labs/FLUX.2-klein-9B` (50 GB)
sh launch.sh 
```
`--cap-add=SYS_ADMIN` is what lets `ncu` read GPU performance counters; without it the
latency phase still works but `--ncu` fails with `ERR_NVGPUCTRPERM`.

Python env (uv, one line, inside `/workspace`):
```bash
uv venv --python 3.12 .venv && uv pip install "torch==2.11.0" "diffusers==0.38.0" "transformers==5.14.1" accelerate sentencepiece protobuf --torch-backend=cu129
```

`--torch-backend=cu129` picks the CUDA 12.9 wheel (driver ≥ 525 via minor-version compat;
this box is on 570). The wheel carries sm_80/86/90/100/120, so the same env runs on
A6000, A100, H100/H200 and B200.

## 2. Run


```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python profile_diffusion.py --compile --ncu --batch 1,2,4
```

Pin one GPU. Takes ~2 h on an RTX A6000 (latency ~30 min incl. per-shape compilation,
ncu ~10 min per model × resolution). Run it under tmux. A shape that does not fit in memory is recorded as OOM and skipped (on 48 GB: SD3 2048² batch 4, FLUX.2 2048² batch 2/4).

For an eager baseline run the same command without `--compile` (results get separate
file names, see below). `--skip-latency` re-parses/re-runs only the ncu part;
`--ncu-batch 1,4` profiles more batches under ncu; `--models sd3` / `--sizes 1024` narrow
the sweep.

## 3. Output files (`results/profile/`)

`<model>` is `sd3` or `flux2klein`; without `--compile` the `_compile-default` part is
dropped and the summary is `summary.json`.

| file | what | count |
|---|---|---|
| `<model>_compile-default_latency.json` | latency phase: one record per (resolution, batch) — total ms, ms per image, images/s, text / transformer / VAE ms, transformer calls and ms per call, peak GB, nvidia-smi busy %, OOM flag, per-repeat raw values | 1 per model |
| `<model>_compile-default_r<res>.csv` | `ncu --import --csv --page raw` of one transformer forward, one row per kernel launch. Also the cache key: delete it to re-measure that resolution | 1 per model × resolution |
| `<model>_compile-default_r<res>.ncu-rep` | ncu binary report of the same forward (50–70 MB; open in Nsight Compute) | 1 per model × resolution |
| `<model>_compile-default_r<res>.ncu.log` | stdout/stderr of the driver process under ncu — look here when a resolution failed | 1 per model × resolution |
| `summary_compile-default.json` | everything merged: `gpu`, `date`, `models → [records]`; each record has the latency fields plus `ncu` = duration-weighted Compute / DRAM / Tensor % of peak, GB/s, launch count, kernel ms and the GEMM / attention / elementwise / norm split (`null` where no ncu was run or the shape OOMed) | 1 |

Optional: `--save-images` adds `<model>_compile-default_r<res>_b<batch>.png` (one image per shape).

To report a run from another machine, send `summary_compile-default.json`, the two
`*_latency.json` and the six `*.csv`; the `.ncu-rep` files are optional.
