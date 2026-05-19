# Environment

Exact toolchain used for the paper's reported numbers.

## OS & drivers

- Ubuntu 22.04.4 LTS (kernel 6.5)
- NVIDIA driver 550.54.15
- CUDA 12.1, cuDNN 8.9.7

## Python

- Python 3.11.9 (CPython, conda-forge build)

## Core PyTorch stack

| package          | version |
|------------------|---------|
| torch            | 2.3.1   |
| torchvision      | 0.18.1  |
| torchaudio       | 2.3.1   |
| torch_geometric  | 2.5.3   |
| torch_scatter    | 2.1.2   |
| torch_sparse     | 0.6.18  |
| mamba_ssm        | 1.2.0.post1 |
| causal_conv1d    | 1.2.2.post1 |

## Numerical / scientific

| package        | version |
|----------------|---------|
| numpy          | 1.26.4  |
| scipy          | 1.13.0  |
| scikit-learn   | 1.4.2   |
| pandas         | 2.2.2   |
| cvxpy          | 1.5.2   |
| networkx       | 3.3     |

## NLP / utilities

| package              | version |
|----------------------|---------|
| sentence-transformers| 3.0.1   |
| transformers         | 4.41.2  |
| omegaconf            | 2.3.0   |
| pyyaml               | 6.0.1   |
| matplotlib           | 3.9.0   |
| tqdm                 | 4.66.4  |
| wandb                | 0.17.0  |

## Reproducing this environment

### Pip freeze

```bash
pip freeze > requirements.lock
pip install -r requirements.lock
```

### Conda yaml (suggested)

```yaml
name: mambaguard
channels: [conda-forge, pytorch, nvidia]
dependencies:
  - python=3.11
  - pytorch=2.3.1
  - pytorch-cuda=12.1
  - torchvision=0.18.1
  - numpy=1.26.4
  - scipy=1.13.0
  - scikit-learn=1.4.2
  - pandas=2.2.2
  - networkx=3.3
  - matplotlib=3.9.0
  - pip
  - pip:
      - torch_geometric==2.5.3
      - mamba_ssm==1.2.0.post1
      - causal_conv1d==1.2.2.post1
      - cvxpy==1.5.2
      - sentence-transformers==3.0.1
      - transformers==4.41.2
      - omegaconf==2.3.0
      - pyyaml==6.0.1
      - wandb==0.17.0
```

## Determinism flags

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

and `torch.use_deterministic_algorithms(True)` inside training scripts.
