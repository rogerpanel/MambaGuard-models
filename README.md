# MambaGuard

**Certified Selective State-Space Detection for Multi-Protocol LLM Agent Security**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19129512-1682d4.svg)](https://doi.org/10.5281/zenodo.19129512)

**Authors**

- Roger Nick Anaedevha — `ar006@campus.mephi.ru`
- Alexander G. Trofimov — `agtrofimov@mephi.ru`

**Affiliation**

Institute of Cyber Intelligence Systems (ICIS), National Research Nuclear University MEPhI, Kashirskoye sh. 31, 115409 Moscow, Russian Federation.

**Manuscript companion**: <https://github.com/rogerpanel/MambaGuard-models>
**Production platform**: <https://github.com/rogerpanel/robustidps.ai> (Zenodo DOI [10.5281/zenodo.19129512](https://doi.org/10.5281/zenodo.19129512))

---

## Abstract

Modern LLM agent ecosystems exchange messages across heterogeneous protocols — the **Model Context Protocol (MCP)**, **Agent Communication Protocol (ACP)**, **Agent-to-Agent (A2A)**, and the **Agent Network Protocol (ANP)** — each of which exposes a distinct adversarial surface. We introduce **MambaGuard**, a unified detector that ingests an event-driven **Unified Heterogeneous Protocol Graph** $G(t)=(V(t),E(t),X(t))$ and produces fine-grained intrusion verdicts over a 34-class taxonomy of tool, communication, capability, data, and control plane attacks. The temporal backbone is a stack of **selective state-space (Mamba) blocks** with input-dependent discretisation $\bar A_t=\exp(\Delta_t A)$ and a depthwise causal Conv1D mixer; spatial dependencies on the protocol graph are encoded by a **temporal GATv2** with Bochner-style time embeddings $\Phi(\Delta t)$. The model is trained end-to-end with a focal objective and a spectral Lipschitz regulariser that yields the certified constant used downstream.

To convert empirical detection accuracy into operational guarantees we couple the detector with a **three-layer composed certificate**: (i) randomised smoothing on the Mamba feature space gives a per-message certified $\ell_2$ radius; (ii) a **Stackelberg** leader–follower linear program produces a robust mixed policy $\pi^*$ against the worst-case attacker mixture; and (iii) a **Hedge** multiplicative-weights regret bound transfers the offline guarantee to the streaming setting. The composed bound $\mathbb{E}[V_{\mathrm{ach}}(\hat\pi_T,\varepsilon)]\geq V^* - L_f\varepsilon - B\sqrt{\ln|A_D|/(2T)}$ certifies the achievable security value as a function of perturbation budget $\varepsilon$, model Lipschitz constant $L_f$, defender action-set size $|A_D|$, and horizon $T$. On a held-out temporal split MambaGuard attains **Macro-F1 = 0.978** at **3.7 ms** mean inference latency and **1.13 Mmsg/s** throughput on a single A100, while reducing the **attack success rate (ASR)** to **2.4 %** and certifying a smoothed radius of **0.041** at noise level $\varepsilon = 0.05$. This repository reproduces every table and figure in the paper.

---

## Key results

| Metric | Value | Conditions |
| --- | --- | --- |
| **Macro-F1** (34-class detection) | **0.978** | Temporal hold-out, Sarhan-2022 protocol |
| **Inference latency (mean)** | **3.7 ms / msg** | A100-80GB, batch 1, fp16, Mamba kernel fused |
| **Throughput** | **1.13 Mmsg/s** | A100-80GB, batch 4096, mixed precision |
| **Attack Success Rate (ASR)** | **2.4 %** | Composite AgentDojo + InjecAgent + ProtocolBench |
| **Certified $\ell_2$ radius** | **0.041** at $\varepsilon = 0.05$ | Randomised smoothing, $n=10^5$, $\alpha=10^{-3}$ |
| **Stackelberg robust value $V^*$** | $\geq 0.91$ | LP over $|A_D|=12$, $|A_A|=34$ |
| **Hedge regret** $R_T$ | $\mathcal{O}(\sqrt{T\ln |A_D|})$ | Online, $\eta=\sqrt{2\ln|A_D|/T}$ |

See `docs/reproducibility.md` for the full results matrix and the commands that regenerate each row.

---

## Architecture

```
                     incoming MCP / ACP / A2A / ANP messages
                                       |
                                       v
              +--------------------------------------------------+
              |   1. Message canonicalisation                     |
              |   m = (tau, s, d, p, mu, t_m)                     |
              |   tau in {M_tool, M_comm, M_cap, M_data, M_ctrl}  |
              +--------------------------------------------------+
                                       |
                                       v
              +--------------------------------------------------+
              |   2. Unified Heterogeneous Protocol Graph         |
              |   G(t) = (V(t), E(t), X(t))                       |
              |   V = V_A (agents) U V_T (tools)                  |
              |     U V_C (capabilities) U V_S (sessions)         |
              +--------------------------------------------------+
                                       |
                                       v
              +-----------------+     +-----------------+
              | Sentence-MiniLM |     | Bochner time    |
              | (frozen) -> x_p |     | encoder Phi(dt) |
              +--------+--------+     +--------+--------+
                       \                       /
                        \                     /
                         v                   v
              +--------------------------------------------------+
              |   3. Selective SSM (Mamba) backbone  x 4 layers   |
              |   h'(t)   = A h(t) + B u(t)                       |
              |   bar A_t = exp(Delta_t A)                        |
              |   bar B_t = (Delta_t A)^-1 (exp(Delta_t A) - I)   |
              |             Delta_t B_t                           |
              |   h^a_t   = bar A_t h^a_{t-1} + bar B_t m^a_t     |
              |   z^a_t   = C_t h^a_t (.) SiLU(W_g m^a_t)         |
              |   + depthwise causal Conv1D, kernel k=4           |
              |   d_z = 256, N = 16, expansion E = 2              |
              +--------------------------------------------------+
                                       |
                                       v
              +--------------------------------------------------+
              |   4. Temporal GATv2 spatial mixer                 |
              |   alpha_ij = softmax(a^T LeakyReLU(W[h_i||h_j||   |
              |              Phi(dt_ij)]))                        |
              |   H = 4 heads, d_T = 64                           |
              +--------------------------------------------------+
                                       |
                                       v
              +--------------------------------------------------+
              |   5. Detection head: softmax over 34 classes      |
              |   + Lipschitz-regularised logits f_theta          |
              +--------------------------------------------------+
                                       |
                +----------------------+----------------------+
                v                                             v
   +----------------------------+              +-------------------------------+
   | 6a. Randomised smoothing   |              | 6b. Stackelberg LP (offline)  |
   |     -> certified radius R  |              |     + Hedge MWU (online)      |
   +----------------------------+              +-------------------------------+
                \                                          /
                 \                                        /
                  v                                      v
              +--------------------------------------------------+
              |   7. Composed three-layer certificate             |
              |   E[V_ach(pi_T, eps)] >= V* - L_f eps             |
              |                       - B sqrt(ln|A_D| / (2T))    |
              +--------------------------------------------------+
                                       |
                                       v
                       verdict + action + certified bound
```

For the full formal description (vertex partitions, message schema, ZOH discretisation derivation, GATv2 attention, head construction, and Lipschitz analysis) see [`docs/architecture.md`](docs/architecture.md).

---

## Installation

MambaGuard targets **Python 3.10 / 3.11** and **CUDA 12.1**. The selective scan kernels (`mamba-ssm`, `causal-conv1d`) need a CUDA toolchain matching your PyTorch build.

```bash
# 1. Clone
git clone https://github.com/rogerpanel/MambaGuard-models.git mambaguard
cd mambaguard

# 2. Create environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# 3. PyTorch with CUDA 12.1 (must precede mamba-ssm install)
pip install torch==2.3.* --index-url https://download.pytorch.org/whl/cu121

# 4. Project + dependencies (editable for development)
pip install -e ".[dev]"

# 5. Sanity check
python -c "import mambaguard, torch; print(mambaguard.__version__, torch.cuda.is_available())"
pytest -q
```

### Docker (production)

A ready-to-deploy stack — FastAPI inference service, PostgreSQL event store, Redis stream broker, and Cloudflare Tunnel — is provided in the [`robustidps.ai`](https://github.com/rogerpanel/robustidps.ai) sister repository (MambaGuard is detector **#14**). See [`docs/deployment.md`](docs/deployment.md).

---

## Quickstart

```bash
# Train MambaGuard on the default ICS3D + IIS3D + IDS-PQC composite split
mambaguard-train --config-name=default \
    data=composite model=mambaguard_base trainer=a100

# Evaluate the released checkpoint on AgentDojo + InjecAgent + ProtocolBench
mambaguard-eval --config-name=eval_agent \
    ckpt=checkpoints/mambaguard_v1.pt

# Produce the certified-radius CSV (table III in the paper)
mambaguard-certify --config-name=certify \
    ckpt=checkpoints/mambaguard_v1.pt \
    smoothing.sigma=0.05 smoothing.n=100000 smoothing.alpha=1e-3
```

Each command accepts standard [Hydra](https://hydra.cc) overrides; the configuration tree lives under `configs/`.

---

## Repository layout

```
MambaGuard/
├── README.md                  ← this file
├── LICENSE                    ← MIT
├── CITATION.cff               ← machine-readable citation
├── requirements.txt           ← pinned dependencies (pip)
├── setup.py / pyproject.toml  ← packaging metadata
├── docs/                      ← architecture, datasets, algorithms, deployment, reproducibility
├── configs/                   ← Hydra YAML configs (data, model, trainer, certify, attacks)
├── mambaguard/                ← Python package
│   ├── data/                  ← loaders for ICS3D, IIS3D, IDS-PQC, AgentDojo, InjecAgent, TGB
│   ├── models/                ← selective SSM block, temporal GATv2, detection head
│   ├── certification/         ← randomised smoothing, Lipschitz bound, Stackelberg LP, Hedge
│   ├── training/              ← train / validate loops, focal loss, schedulers
│   ├── attacks/               ← prompt-injection, MCP rebinding, A2A forgery, ANP DID attacks
│   ├── evaluation/            ← Macro-F1, ASR, latency / throughput meters
│   └── utils/                 ← seeds, logging, graph builders, time encoders
├── scripts/                   ← thin CLI wrappers wired to console-script entry points
├── tests/                     ← pytest unit + integration tests
├── benchmarks/baselines/      ← Transformer-IDS, GraphIDS, GNN-CL, DeepLog reference impls
└── notebooks/                 ← exploratory analyses, figure generation
```

---

## Datasets

MambaGuard is trained and evaluated on three families of datasets — see [`docs/datasets.md`](docs/datasets.md) for size, download links, DOIs, licences, and preprocessing recipes.

| Family | Components | Purpose |
| --- | --- | --- |
| **ICS3D** | Microsoft Cloud telemetry, Edge-IIoTset, Kubernetes / Docker audit logs | Industrial / cloud control-plane attacks |
| **IIS3D** | CSE-CIC-IDS2018, CIC-IoT2023, UNSW-NB15 | Network intrusion baseline |
| **IDS-PQC** | NF-CSE-CIC-IDS2018-v3 with post-quantum-TLS captures | PQC-aware network IDS |
| **Agent benchmarks** | [AgentDojo](https://github.com/ethz-spylab/agentdojo) (629), [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (1054), **ProtocolBench** (50+) | LLM-agent prompt-injection / tool-misuse |
| **Temporal graph** | [TGB 2.0](https://tgb.complexdatalab.com/) | Pretraining of the temporal GATv2 mixer |
| **Classical IDS** | CICIDS2017, NSL-KDD, CIC-IoT-2023, UNSW-NB15 / NF-ToN-IoT-V2, CIC-DDoS-2019, [RobustIDPS-PQC](https://doi.org/10.34740/kaggle/dsv/15424420), Edge-IIoTset | Cross-domain transfer baselines |

All splits follow the temporal protocol of **Sarhan et al. 2022** ("Towards a Standard Feature Set for Network Intrusion Detection System Datasets") to avoid the random-split inflation that has plagued earlier IDS benchmarks.

---

## Citing MambaGuard

If you use this software or its results in academic work, please cite both the software (`CITATION.cff`) and the article:

```bibtex
@article{anaedevha2026mambaguard,
  title   = {MambaGuard: Certified Selective State-Space Detection for
             Multi-Protocol LLM Agent Security},
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander G.},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026},
  note    = {Software: \url{https://github.com/rogerpanel/MambaGuard-models};
             Platform: \url{https://github.com/rogerpanel/robustidps.ai};
             DOI: 10.5281/zenodo.19129512}
}
```

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE). Trained checkpoints distributed via Zenodo and the production platform are released under **CC-BY-4.0**.

---

## Contact

- Roger Nick Anaedevha — `ar006@campus.mephi.ru`
- Alexander G. Trofimov — `agtrofimov@mephi.ru`
- Issues and pull requests: <https://github.com/rogerpanel/MambaGuard-models/issues>
- Production platform feedback: <https://github.com/rogerpanel/robustidps.ai/discussions>
