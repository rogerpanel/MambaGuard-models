# Reproduction Recipe

Step-by-step instructions to reproduce every table and figure from the paper
"MambaGuard: Certified Selective State-Space Detection for Multi-Protocol LLM
Agent Security".

All commands assume the working directory is the repository root and that the
package was installed with `pip install -e .` Wall-clock estimates assume a
single NVIDIA A100 80GB.

## 0. Environment & datasets

```bash
# Create env (see ENVIRONMENT.md for full lock-file)
conda create -n mambaguard python=3.11 -y && conda activate mambaguard
pip install -e .[dev]

# Download the datasets (some require manual registration; see file)
bash reproducibility/DATASETS_DOWNLOAD.sh
```

## 1. Data preparation (~25 min per dataset)

```bash
python -m scripts.prepare_data --dataset iis3d \
    --root data/raw/iis3d --output data/processed/iis3d
python -m scripts.prepare_data --dataset agentdojo \
    --root data/raw/agentdojo --output data/processed/agentdojo
python -m scripts.prepare_data --dataset injectagent \
    --root data/raw/injectagent --output data/processed/injectagent
```

## 2. Train MambaGuard (~6 h on A100)

```bash
python -m scripts.train --config configs/default.yaml --dataset iis3d \
    --output-dir outputs/mambaguard_iis3d --seed 42
```

Artefacts: `outputs/mambaguard_iis3d/{config.yaml,metrics.json,train.log,best.ckpt}`.

## 3. Evaluate (~10 min)

```bash
python -m scripts.evaluate --checkpoint outputs/mambaguard_iis3d/best.ckpt \
    --dataset iis3d --split test --output-dir outputs/mambaguard_iis3d/eval
```

## 4. Certification (~3 min)

```bash
python -m scripts.certify --checkpoint outputs/mambaguard_iis3d/best.ckpt \
    --epsilon 0.05 --horizon 10000 --dataset iis3d
```

## 5. Latency / throughput (~5 min)

```bash
python -m scripts.benchmark_latency --checkpoint outputs/mambaguard_iis3d/best.ckpt
```

## 6. Headline tables

```bash
python -m scripts.reproduce_table --table main         # writes main_table.csv
python -m scripts.reproduce_table --table ablation     # writes ablation_table.csv
python -m scripts.reproduce_table --table certification
python -m scripts.reproduce_table --table attacks
```

Wall-clock for the four `reproduce_table` calls combined: ~3.5 h on A100,
dominated by the ablation runs.

## 7. MCP Adversary Bench (~5 min)

```bash
python -m scripts.run_protocolbench \
    --checkpoint outputs/mambaguard_iis3d/best.ckpt
```

## 8. Figures

Open `notebooks/04_results.ipynb` and run all cells. Plots are written to
`docs/figures/`.

## Total wall-clock estimate

End-to-end reproduction (data → all tables + figures) on a single A100 80GB:
**approximately 14 hours**.
