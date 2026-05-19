# Reproducibility

This document is the contract between the manuscript and the codebase. Anyone running the commands below on the prescribed environment **must** recover the headline numbers within the tolerance stated in each row.

---

## 1. Environment

| Item | Pinned value |
| --- | --- |
| OS | Ubuntu 22.04 LTS (or any Linux with glibc ≥ 2.31) |
| Python | **3.10** or **3.11** |
| CUDA | **12.1** |
| cuDNN | 9.0 |
| PyTorch | **2.3.x** (cu121 wheels) |
| `mamba-ssm` | ≥ 1.2 |
| `causal-conv1d` | ≥ 1.2 |
| `torch-geometric` | ≥ 2.5 |
| GPU | **NVIDIA A100 80GB** (SXM4 or PCIe) |
| GPU driver | ≥ 535.86 |
| RAM | ≥ 64 GB |
| Disk | ≥ 500 GB SSD (datasets dominate) |

A pinned environment is reproduced by:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.3.* --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

The `pyproject.toml` lower bounds correspond to the versions actually used during the runs reported below.

---

## 2. Seeds

| Source of randomness | Seed |
| --- | --- |
| `numpy.random` | 1337 |
| `torch.manual_seed` | 1337 |
| `torch.cuda.manual_seed_all` | 1337 |
| Python `random` | 1337 |
| `cudnn.benchmark` | `True` (training), `False` (eval) |
| `torch.use_deterministic_algorithms` | `False` (training), `True` (eval) |

Seeds and determinism are wired through `mambaguard.utils.seed.set_global_seed(1337)`, which is called at the top of every script entry point.

---

## 3. Expected results

| Table / figure in paper | Command | Wall-clock | Expected value (± tolerance) |
| --- | --- | --- | --- |
| Table I — composite detection | `mambaguard-train --config-name=default && mambaguard-eval --config-name=eval_composite` | ~14 h train + 12 min eval | **Macro-F1 = 0.978 ± 0.003** |
| Table II — agent benchmarks | `mambaguard-eval --config-name=eval_agent ckpt=checkpoints/mambaguard_v1.pt` | ~25 min | **ASR = 2.4 % ± 0.4 pp** on AgentDojo+InjecAgent+ProtocolBench |
| Table III — certified radius | `mambaguard-certify --config-name=certify smoothing.sigma=0.05` | ~3 h (n=1e5) | **R = 0.041 ± 0.002** at ε = 0.05 |
| Table IV — Stackelberg | `python -m mambaguard.certification.stackelberg --config=configs/certify/stackelberg.yaml` | < 2 s | **V\* ≥ 0.91** |
| Table V — latency / throughput | `python tools/bench_latency.py && python tools/bench_throughput.py` | 8 min | **3.7 ms / 1.13 Mmsg/s** |
| Figure 3 — radius vs σ | `python tools/figure_radius_sigma.py` | ~12 h (sweep over σ) | Curve matches Fig. 3, ε at σ=0.05 within ±0.002 |
| Figure 4 — Hedge regret | `python tools/figure_hedge_regret.py` | ~30 min | Regret ≤ $1.05\,B\sqrt{T\ln|A_D|}$ |
| Figure 5 — ablations | `bash scripts/run_ablations.sh` | ~36 h | Ranking unchanged; deltas within ±0.5 pp |

All tolerances are 95% intervals across 3 seeds (1337, 2025, 4242).

---

## 4. Hardware budget

| Phase | Budget on A100 80GB |
| --- | --- |
| End-to-end **training** | **~14 h** |
| Full **certification** sweep (Tables III, IV, V) | ~3.5 h |
| Agent-benchmark **evaluation** | ~25 min |
| Ablation matrix (Fig. 5) | ~36 h (8 configs × 3 seeds) |

Total end-to-end reproduction of every table and figure: **< 60 GPU-hours** on a single A100, or **< 8 h** wallclock on a 4×A100 node with the multi-seed bash launchers.

---

## 5. Reproduction recipe (compact)

```bash
# 0. environment (see Section 1)

# 1. data
python -m mambaguard.data.fetch ics3d iis3d ids-pqc agentdojo injecagent protocolbench tgb2

# 2. train
mambaguard-train --config-name=default trainer.seed=1337

# 3. evaluate
mambaguard-eval --config-name=eval_composite ckpt=checkpoints/mambaguard_v1.pt
mambaguard-eval --config-name=eval_agent     ckpt=checkpoints/mambaguard_v1.pt

# 4. certify
mambaguard-certify --config-name=certify ckpt=checkpoints/mambaguard_v1.pt \
                   smoothing.sigma=0.05 smoothing.n=100000 smoothing.alpha=1e-3
python -m mambaguard.certification.stackelberg --config=configs/certify/stackelberg.yaml
python -m mambaguard.certification.composed    --config=configs/certify/composed.yaml

# 5. figures
python tools/figure_radius_sigma.py
python tools/figure_hedge_regret.py

# 6. multi-seed
for s in 1337 2025 4242; do
    mambaguard-train --config-name=default trainer.seed=$s
done
python tools/aggregate_seeds.py
```

---

## 6. Provenance

Every run writes a `manifest.json` to its output directory containing: git commit SHA, Python version, PyTorch version, CUDA version, GPU name, full Hydra config, environment variables, and the SHA-256 of every input dataset shard. The manifest is the unit of reproducibility — please attach it to bug reports.

Pre-trained checkpoints used to produce the camera-ready numbers are released on Zenodo (DOI [10.5281/zenodo.19129512](https://doi.org/10.5281/zenodo.19129512)) with their accompanying manifest. Loading is `torch.load("mambaguard_v1.pt", map_location="cuda", weights_only=True)`.
