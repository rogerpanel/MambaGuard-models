# Datasets

MambaGuard is evaluated on three dataset families plus several classical IDS corpora used for cross-domain transfer. All splits follow the **temporal protocol** of Sarhan, Layeghy and Portmann (2022) to avoid the random-split inflation that has dominated earlier intrusion-detection benchmarks: training data must strictly precede validation data in wall-clock time, which in turn must precede the test data. Loaders that enforce this contract live under `mambaguard/data/`.

For every dataset the table below gives the canonical citation, the access URL, the licence, and the preprocessing notes used by the loader. Downloads are not committed; the helper `python -m mambaguard.data.fetch <name>` resolves each dataset to `data/raw/<name>/`.

---

## 1. ICS3D — Industrial / Cloud / Container

A composite **I**ndustrial, **C**loud, and **C**ontainer **S**ecurity **3D**ataset assembled in this work. Components:

### 1.1 Microsoft Cloud telemetry (GUIDE-2024)

- Source: Microsoft Defender for Cloud Apps anonymised telemetry release (used under the dataset's research-only licence; access via Microsoft Research).
- Volume: ~13 M events, 21 control-plane attack categories.
- Reference: Freitas et al., *GUIDE: a Guideline-aware, Unified IDS for Enterprise cloud telemetry*, IEEE BigData 2024.
- Preprocessing: per-tenant tokenisation, control-plane events labelled to $M_{\mathrm{ctrl}}$, OAuth grants to $M_{\mathrm{cap}}$.

### 1.2 Edge-IIoTset

- URL: <https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot>
- Volume: ~20 M records, 14 attack types across MQTT, Modbus, OPC-UA layers.
- Citation: Ferrag et al., *Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications*, IEEE Access 2022.
- Preprocessing: protocol fields mapped to $M_{\mathrm{tool}}$ (Modbus/OPC), $M_{\mathrm{data}}$ (MQTT payload).

### 1.3 Kubernetes / Docker audit logs

- Source: CNCF Falco public corpus + author-collected lab traces (released alongside the manuscript on Zenodo).
- Volume: ~4 M Kubernetes audit-log records, 8 attack categories (container escape, secret theft, RBAC misuse, ...).
- Citation: Falco Project, *Cloud Native Runtime Security*, CNCF 2023.
- Preprocessing: each `audit.k8s.io` event is canonicalised to $(\tau{=}M_{\mathrm{ctrl}},\,s{=}user,\,d{=}resource)$; image-pull events are $M_{\mathrm{tool}}$.

---

## 2. IIS3D — IP / IoT / Industrial Network IDS

### 2.1 CSE-CIC-IDS2018

- URL: <https://www.unb.ca/cic/datasets/ids-2018.html>
- Volume: ~16 M flows, 14 attack types (DoS, Botnet, infiltration, etc.).
- Citation: Sharafaldin, Lashkari, Ghorbani, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterisation*, ICISSP 2018.
- Preprocessing: NetFlow v9 features; chronological train/val/test = 0.7/0.1/0.2 by `Timestamp`.

### 2.2 CIC-IoT2023

- URL: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>
- Volume: ~46 M records, 33 attacks across 105 IoT devices.
- Citation: Neto et al., *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment*, Sensors 2023.
- Preprocessing: device-type one-hot used as auxiliary feature on $V_T$.

### 2.3 UNSW-NB15

- URL: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- Volume: ~2.5 M records, 9 attack categories.
- Citation: Moustafa, Slay, *UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS 2015.
- Preprocessing: Argus-features → NetFlow v9 (matching CSE-CIC-IDS2018), per the harmonisation of Sarhan et al. 2022.

---

## 3. IDS-PQC — Post-Quantum-TLS-aware IDS

- Base: **NF-CSE-CIC-IDS2018-v3** (Sarhan et al. NetFlow-v3 harmonisation).
- Augmentation: PQC-TLS handshake captures (Kyber-768, Dilithium-3, SPHINCS+) collected with the [oqs-openssl](https://github.com/open-quantum-safe/openssl) stack against a hybrid TLS-1.3 reverse-proxy fleet (8 servers, 30 days).
- Volume: 47.2 M flows total (~3.1 M with PQC handshakes).
- Citation: Sarhan, Layeghy, Portmann, *Towards a Standard Feature Set for Network Intrusion Detection System Datasets*, MONET 2022; PQC augmentation released with this paper as **RobustIDPS-PQC** (Kaggle DOI [10.34740/kaggle/dsv/15424420](https://doi.org/10.34740/kaggle/dsv/15424420)).
- Preprocessing: cipher-suite indicator added as a categorical feature; handshake-only flows kept for the certified-robustness study because they encode the worst-case fingerprintable surface.

---

## 4. LLM-agent attack benchmarks

### 4.1 AgentDojo

- URL: <https://github.com/ethz-spylab/agentdojo>
- Volume: **629** prompt-injection / tool-misuse cases across 4 environments (workspace, banking, travel, Slack).
- Citation: Debenedetti et al., *AgentDojo: A Dynamic Environment to Evaluate Attacks and Defences for LLM Agents*, NeurIPS 2024 D&B.
- Preprocessing: each case is replayed against a sandboxed MCP shim; the resulting tool-call trace is canonicalised to $M_{\mathrm{tool}}$ / $M_{\mathrm{cap}}$.

### 4.2 InjecAgent

- URL: <https://github.com/uiuc-kang-lab/InjecAgent>
- Volume: **1054** indirect-prompt-injection cases across 17 user-tools × 62 attacker-tools.
- Citation: Zhan et al., *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents*, ACL 2024 Findings.
- Preprocessing: identical sandbox to AgentDojo; payloads tokenised by sentence-encoder.

### 4.3 ProtocolBench (this work)

- Description: a new benchmark with **50+** end-to-end scenarios spanning MCP, ACP, A2A, and ANP, covering 12 tool-call, 8 comm, 6 capability, 5 data, and 3 control-plane attack patterns.
- Release: bundled with the project under `benchmarks/protocolbench/` and mirrored on Zenodo (DOI assigned at camera-ready).
- Preprocessing: scenarios ship as YAML manifests + replay logs.

---

## 5. Temporal Graph Benchmark (TGB 2.0)

- URL: <https://tgb.complexdatalab.com/>
- Volume: 9 temporal graphs, up to 28 M temporal edges.
- Citation: Gastinger et al., *Towards Better Evaluation for Dynamic Link Prediction* (TGB 2.0), NeurIPS 2024 D&B.
- Use: pretraining of the temporal GATv2 mixer on link-prediction objectives before fine-tuning on the protocol graph.

---

## 6. Classical IDS corpora (transfer evaluation)

| Dataset | URL | Citation |
| --- | --- | --- |
| **CICIDS2017** | <https://www.unb.ca/cic/datasets/ids-2017.html> | Sharafaldin et al., ICISSP 2018 |
| **NSL-KDD** | <https://www.unb.ca/cic/datasets/nsl.html> | Tavallaee et al., CISDA 2009 |
| **CIC-IoT-2023** | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | Neto et al., Sensors 2023 |
| **UNSW-NB15 / NF-ToN-IoT-V2** | <https://staff.itee.uq.edu.au/marius/NIDS_datasets/> | Sarhan et al., MONET 2022 |
| **CIC-DDoS-2019** | <https://www.unb.ca/cic/datasets/ddos-2019.html> | Sharafaldin et al., ICCST 2019 |
| **RobustIDPS-PQC** | <https://doi.org/10.34740/kaggle/dsv/15424420> | Anaedevha & Trofimov, Kaggle 2025 |
| **Edge-IIoTset** | <https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot> | Ferrag et al., IEEE Access 2022 |

---

## 7. Splitting protocol

For each dataset the loader emits `(X_train, y_train, X_val, y_val, X_test, y_test)` with the following invariant:

$$
\max_{i\in\mathrm{train}} t_i \;<\; \min_{j\in\mathrm{val}} t_j \;\leq\; \max_{j\in\mathrm{val}} t_j \;<\; \min_{k\in\mathrm{test}} t_k,
$$

i.e. **no temporal leakage** between splits. This follows Sarhan et al. 2022. Random splits — the convention in much of the historical IDS literature — inflate F1 by 2–8 pp because flows from the same attack session leak across splits; we report only the temporal-split numbers.

Class imbalance is handled at the loss level via the focal weighting $(\alpha, \gamma)$ documented in [`hyperparameters.md`](hyperparameters.md), not by oversampling, again to keep the temporal contract intact.

---

## 8. Preprocessing pipeline summary

1. **Parse** raw `.pcap` / `.csv` / `.json` to NetFlow-v3 features (Sarhan et al. harmonisation).
2. **Canonicalise** each event to the six-tuple $m=(\tau,s,d,p,\mu,t_m)$ defined in [`architecture.md`](architecture.md).
3. **Embed** payloads with `sentence-transformers/all-MiniLM-L6-v2` (frozen, $d_p=384$).
4. **Build graph** $G(t)$ incrementally; persist as PyG `TemporalData` shards of 1 M events.
5. **Split temporally** as above; cache to `data/processed/<dataset>/{train,val,test}.pt`.
