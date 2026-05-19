# MambaGuard documentation index

This directory holds the reference documentation for the MambaGuard reproducibility codebase. The top-level [`README.md`](../README.md) is the project landing page; the files below give the formal and operational detail required to reproduce the paper.

| File | Contents |
| --- | --- |
| [`architecture.md`](architecture.md) | Unified Heterogeneous Protocol Graph, message schema, selective SSM equations and ZOH discretisation, depthwise causal Conv1D, temporal GATv2 with Bochner time encoding, 34-class detection head. |
| [`datasets.md`](datasets.md) | Every dataset MambaGuard touches: ICS3D, IIS3D, IDS-PQC, AgentDojo, InjecAgent, ProtocolBench, TGB 2.0, CICIDS2017, NSL-KDD, CIC-IoT-2023, UNSW-NB15 / NF-ToN-IoT-V2, CIC-DDoS-2019, RobustIDPS-PQC, Edge-IIoTset. Download links, DOIs, citations, preprocessing notes, temporal-split protocol. |
| [`hyperparameters.md`](hyperparameters.md) | Selective block geometry, GATv2 widths, training schedule, focal-loss / Lipschitz regulariser weights, AdamW settings, Hedge learning rate, discretisation bounds, embedding choice. |
| [`algorithms.md`](algorithms.md) | Pseudocode for Algorithm 1 (online defence with Hedge MWU), Lipschitz constant computation, Stackelberg LP, and the composed three-layer certificate. |
| [`deployment.md`](deployment.md) | RobustIDPS.ai integration (detector #14), Docker stack (FastAPI + PostgreSQL + Redis + Cloudflare Tunnel), latency and throughput SLOs. |
| [`reproducibility.md`](reproducibility.md) | Environment versions, seeds, hardware budget, and the exact CLI commands that reproduce each table and figure in the manuscript. |

Notation conventions used throughout the docs:

- Lower-case bold (e.g. $\mathbf{h}_t$) is a vector; upper-case bold ($\mathbf{A}$) a matrix; calligraphic ($\mathcal{G}$, $\mathcal{V}$) sets.
- $\tau$ denotes protocol type, $\Delta_t$ the input-dependent discretisation step, $\Phi(\cdot)$ the Bochner time encoder, $L_f$ the model Lipschitz constant, $V^*$ the Stackelberg robust value, $R_T$ the Hedge regret.
- All proofs are deferred to the appendix of the manuscript; the docs cross-reference equation numbers.
