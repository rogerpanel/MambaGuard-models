# Hyperparameters

All values below are the defaults shipped in `configs/`. Every entry is overridable on the command line via Hydra (`mambaguard-train model.d_z=512 trainer.epochs=50`). Numbers in **bold** are the ones used to produce the headline results in the manuscript.

---

## 1. Selective state-space block (`configs/model/mambaguard_base.yaml`)

| Hyperparameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| State dimension | $N$ | **16** | HiPPO-S4D initialisation |
| Hidden / model dimension | $d_z$ | **256** | Per-token feature width |
| Expansion factor | $E$ | **2** | Inner width is $E\cdot d_z = 512$ |
| Conv1D kernel width | $k$ | **4** | Depthwise causal |
| Stacked layers | $L$ | **4** | Mamba × GATv2 interleaved |
| Discretisation step | $\Delta_t$ | $\in[\Delta_{\min},\Delta_{\max}]$ | $\Delta_{\min}=10^{-3},\ \Delta_{\max}=10^{-1}$ |
| Hurwitz condition | — | $\Delta_{\max}\,\alpha(\mathbf{A})<0$ | Enforced by HiPPO-S4D init |
| Activation | — | SiLU | Gate branch |
| Normalisation | — | RMSNorm | Pre-norm |

---

## 2. Temporal GATv2 mixer (`configs/model/gatv2.yaml`)

| Hyperparameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Attention heads | $H$ | **4** | Concatenated |
| Time-encoding dim | $d_T$ | **64** | Bochner encoder $\Phi(\Delta t)$ |
| Head dimension | — | 64 | Per head; total width $H\cdot 64=256$ |
| Dropout | — | 0.1 | Edge + attention |
| Activation | — | LeakyReLU(0.2) | Attention pre-softmax |

---

## 3. Embedding (`configs/data/embedding.yaml`)

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` | Frozen at training time |
| Payload dim | $d_p = 384$ | Concatenated with $\tau$-one-hot and counters |
| Max payload length | 256 tokens | Truncated / padded |

---

## 4. Training (`configs/trainer/a100.yaml`)

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| Loss | **Focal** with $\alpha=0.25$, $\gamma=2$ | Lin et al. 2017 |
| Lipschitz regulariser | $\lambda_L = 10^{-3}$ on $(\|\mathbf{W}^{(\ell)}\|_2 - 1)^2_+$ | Spectral; computed via power iteration, 1 step per minibatch |
| Optimiser | AdamW | $\beta_1=0.9$, $\beta_2=0.999$ |
| Learning rate | $3\times 10^{-4}$ | Peak |
| Weight decay | $10^{-2}$ | |
| Batch size | **256** | Per-GPU |
| Epochs | **30** | Cosine decay to $10^{-6}$ |
| LR schedule | Cosine | 1 epoch linear warmup |
| Gradient clip (L2) | 1.0 | |
| Mixed precision | bf16 (fp16 fallback) | |
| Hardware | **NVIDIA A100 80GB** | Single GPU sufficient |
| Seed | 1337 | Reproducible across DataLoader workers |

Total training cost ≈ **14 h** on a single A100; see [`reproducibility.md`](reproducibility.md).

---

## 5. Online defence (Hedge MWU)

| Hyperparameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Learning rate | $\eta$ | $\sqrt{2\ln|A_D|\,/\,T}$ | Optimal Hedge rate; see Algorithm 1 |
| Defender action space | $|A_D|$ | 12 | Block, alert, isolate, downgrade-tool, revoke-cap, ... |
| Attacker action space | $|A_A|$ | 34 | One per attack class |
| Stream horizon | $T$ | per-deployment | Reset on policy update |

---

## 6. Randomised smoothing (`configs/certify/smooth.yaml`)

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| Noise distribution | $\mathcal{N}(0,\sigma^2\mathbf{I})$ | Applied in Mamba feature space |
| Noise scale | $\sigma\in\{0.01,0.025,0.05,0.10\}$ | Default 0.05 |
| Monte-Carlo samples | $n = 10^5$ | Cohen et al. 2019 |
| Failure probability | $\alpha = 10^{-3}$ | Clopper-Pearson lower bound |

---

## 7. Stackelberg LP (`configs/certify/stackelberg.yaml`)

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| Solver | `cvxpy` + ECOS | LP, < 50 ms |
| Defender utility matrix | $U_D \in \mathbb{R}^{|A_D|\times|A_A|}$ | Per-deployment cost model |
| Tie-break | Strong Stackelberg equilibrium | Conitzer & Sandholm 2006 |

---

## 8. Reproducibility knobs

- **Determinism**: `torch.use_deterministic_algorithms(True)` enabled in evaluation; disabled during training because the fused selective scan kernel is non-deterministic on A100 (the trained-model variance is < 0.2 pp Macro-F1).
- **DataLoader**: `num_workers=8`, `persistent_workers=True`, `pin_memory=True`.
- **CUDA**: 12.1. PyTorch 2.3. `cuDNN` benchmark on. `TF32` matmul on.

All values are mirrored in the `configs/` tree; the loader unit tests in `tests/test_configs.py` verify drift.
