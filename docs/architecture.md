# MambaGuard architecture

This document gives the formal specification of MambaGuard's data structures and neural modules. Implementation lives under `mambaguard/models/` (selective SSM block, GATv2 mixer, detection head) and `mambaguard/data/graph_builder.py` (graph construction). Equation numbers match the manuscript.

---

## 1. Unified Heterogeneous Protocol Graph

The detector ingests an event-driven temporal graph

$$
G(t) = \bigl(V(t),\,E(t),\,X(t)\bigr),
$$

continuously updated as MCP, ACP, A2A, and ANP messages arrive. The vertex set is partitioned into four disjoint subsets:

$$
V(t) \;=\; V_A(t)\,\dot\cup\,V_T(t)\,\dot\cup\,V_C(t)\,\dot\cup\,V_S(t),
$$

where

- $V_A$ — **agents** (LLM-driven autonomous principals identified by their DID / OAuth subject);
- $V_T$ — **tools** (MCP servers, REST endpoints, function-call targets);
- $V_C$ — **capabilities** (declared scopes, OAuth grants, ACP role tokens);
- $V_S$ — **sessions** (conversation, RPC stream, or ANP routing context).

Edges $E(t)$ are typed: ownership (`agent → capability`), invocation (`agent → tool`), delegation (`agent → agent`), and session membership (`* → session`). Each edge carries the protocol family $\tau$ and the timestamp of the most recent message.

Vertex feature matrix $X(t)\in\mathbb{R}^{|V(t)|\times d}$ is the concatenation of (i) a frozen sentence-encoder embedding of the most recent payload, (ii) a one-hot of $\tau$, (iii) cumulative counters (messages-in, messages-out, error rate), and (iv) the Bochner time encoding $\Phi(\Delta t)$ of the staleness of the vertex.

### 1.1 Message canonicalisation

Every raw protocol frame is mapped to the canonical six-tuple

$$
m \;=\; (\tau,\,s,\,d,\,p,\,\mu,\,t_m), \qquad
\tau\in\{M_{\mathrm{tool}},\,M_{\mathrm{comm}},\,M_{\mathrm{cap}},\,M_{\mathrm{data}},\,M_{\mathrm{ctrl}}\},
$$

with

- $\tau$ — protocol-class label (tool-call, agent-comm, capability-grant, data-payload, control-plane);
- $s, d\in V$ — source and destination vertices;
- $p\in\mathcal{P}$ — payload (string, JSON, or binary blob);
- $\mu$ — protocol-specific metadata (headers, signatures, DID, MCP method, ACP action, A2A intent, ANP service descriptor);
- $t_m\in\mathbb{R}_{\geq 0}$ — wall-clock timestamp.

The canonicalisation step is intentionally lossy on irrelevant transport metadata (HTTP server header, TLS cipher) but preserves every field used by the threat model in Sec. III of the paper. See `mambaguard/data/canonicalise.py`.

---

## 2. Selective state-space (Mamba) backbone

Each vertex $a\in V$ owns a per-agent token stream $\{m^a_t\}_{t}$. The backbone applies $L=4$ stacked selective SSM layers to this stream. A single block follows the Mamba formulation of Gu and Dao (2024).

### 2.1 Continuous-time linear system

$$
h'(t) \;=\; \mathbf{A}\,h(t) + \mathbf{B}\,u(t),
\qquad
y(t) \;=\; \mathbf{C}\,h(t),
$$

with state $h(t)\in\mathbb{R}^{N}$, $N=16$ HiPPO-S4D-initialised modes, and input $u(t)\in\mathbb{R}^{d_z}$, $d_z=256$.

### 2.2 Input-dependent zero-order-hold discretisation

For each token the network predicts a positive step $\Delta_t = \mathrm{softplus}(W_\Delta\,m^a_t)\in[\Delta_{\min},\Delta_{\max}]$ and discretises by ZOH:

$$
\bar{\mathbf{A}}_t \;=\; \exp(\Delta_t \mathbf{A}),
\qquad
\bar{\mathbf{B}}_t \;=\; (\Delta_t \mathbf{A})^{-1}\bigl(\exp(\Delta_t \mathbf{A}) - \mathbf{I}\bigr)\Delta_t \mathbf{B}_t.
$$

For numerical stability the inverse is implemented via the Padé-(3,3) approximation when $\|\Delta_t\mathbf{A}\|<10^{-3}$ and via diagonal-mode evaluation otherwise. The discretisation respects the **Hurwitz condition** $\Delta_{\max}\,\alpha(\mathbf{A})<0$ where $\alpha(\cdot)$ is the spectral abscissa; with HiPPO-S4D this is satisfied for any $\Delta_{\max}\leq 1$.

### 2.3 Selective recurrence

$$
h^{a}_{t} \;=\; \bar{\mathbf{A}}_t\,h^{a}_{t-1} + \bar{\mathbf{B}}_t\,m^{a}_t,
\qquad
z^{a}_{t} \;=\; \mathbf{C}_t\,h^{a}_t \;\odot\; \mathrm{SiLU}(\mathbf{W}_g\,m^{a}_t),
$$

where $\mathbf{B}_t,\mathbf{C}_t$ are also linear functions of the current input (this is the **selective** mechanism that distinguishes Mamba from S4), and the gating SiLU branch realises the standard Mamba gated-MLP.

### 2.4 Depthwise causal Conv1D mixer

Between the input projection and the selective scan we apply a depthwise causal convolution of kernel width $k=4$:

$$
\tilde m^{a}_t \;=\; \sum_{j=0}^{k-1} \mathbf{w}_j \odot m^{a}_{t-j},
$$

which gives a short receptive field on top of the long-range SSM recurrence and dramatically improves the latency on bursty traffic. The fused kernel from `causal-conv1d>=1.2` is used at inference.

### 2.5 Stacking

The block expansion factor is $E=2$, so the inner hidden dimension is $E\cdot d_z = 512$. After the SSM output the hidden state is projected back to $d_z$, residual-connected to the input, and `RMSNorm`-ed. We stack $L=4$ such blocks.

---

## 3. Temporal GATv2 spatial mixer

After the per-agent SSM pass we mix neighbours on the protocol graph. We use a Brody-et-al. 2022 GATv2 with Bochner time encodings (Xu et al. 2020):

$$
\Phi(\Delta t) \;=\; \sqrt{\tfrac{2}{d_T}}\,\bigl[\cos(\omega_1\Delta t + \varphi_1),\ldots,\cos(\omega_{d_T}\Delta t + \varphi_{d_T})\bigr]^\top,
$$

with learnable $\{\omega_k,\varphi_k\}$ and $d_T = 64$.

For each ordered edge $(i,j)$ the attention coefficient is

$$
\alpha_{ij} \;=\; \mathrm{softmax}_{j\in\mathcal{N}(i)}\Bigl(\mathbf{a}^\top \mathrm{LeakyReLU}\bigl(\mathbf{W}\,[\,z_i\,\|\,z_j\,\|\,\Phi(\Delta t_{ij})\,]\bigr)\Bigr),
$$

and the output, with $H=4$ heads concatenated, is

$$
\tilde z_i \;=\; \big\Vert_{h=1}^{H} \;\sigma\!\Bigl(\sum_{j\in\mathcal{N}(i)} \alpha_{ij}^{(h)}\,\mathbf{W}^{(h)}_v\,z_j\Bigr).
$$

We use a single GATv2 layer per Mamba layer (interleaved), giving a temporal–spatial alternation pattern.

---

## 4. Detection head

The final representation $\tilde z_i\in\mathbb{R}^{d_z}$ for the **message-bearing vertex** is passed through a 2-layer MLP and a softmax over the **34-class** attack taxonomy (12 tool-call attacks, 8 comm-plane, 6 capability, 5 data-plane, 3 control-plane attacks, plus the benign class — see Table II of the manuscript). Logits are constrained via spectral normalisation so the Lipschitz constant $L_f$ used by the certificate is tight.

$$
\hat y \;=\; \mathrm{softmax}\bigl(\mathbf{W}_2\,\mathrm{GELU}(\mathbf{W}_1\,\tilde z_i)\bigr).
$$

The Lipschitz bound

$$
L_f \;\leq\; M_h\cdot L_{\mathrm{gat}}\cdot \frac{\gamma\beta(1+\kappa_g)}{1-\rho}
$$

is derived in [`algorithms.md`](algorithms.md), where $M_h$ is the head Lipschitz, $L_{\mathrm{gat}}$ the GATv2 contribution, $\beta=\|\mathbf{B}\|$, $\gamma=\|\mathbf{C}\|$, $\kappa_g$ the gate-SiLU constant, and $\rho<1$ the contraction rate guaranteed by the Hurwitz condition on $\bar{\mathbf{A}}_t$.

---

## 5. Inputs, outputs, and contracts

| Symbol | Shape | Description |
| --- | --- | --- |
| $m^a_t$ | $(d_z,)$ | Canonicalised message embedding for agent $a$ at step $t$ |
| $h^a_t$ | $(N,)$ | Mamba hidden state |
| $z^a_t$ | $(d_z,)$ | SSM output (per agent, per step) |
| $\tilde z_i$ | $(d_z,)$ | GATv2-mixed representation at vertex $i$ |
| $\hat y$ | $(34,)$ | Class probabilities |
| $\Delta_t$ | $(1,)$ | Input-dependent discretisation step, clipped to $[\Delta_{\min},\Delta_{\max}]$ |
| $\Phi(\Delta t)$ | $(d_T,)$ | Bochner time encoding |

The exact tensor shapes after batching are documented in the `forward` docstring of `mambaguard.models.MambaGuard`.

---

## 6. References

- A. Gu and T. Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** COLM 2024.
- S. Brody, U. Alon, E. Yahav. **How Attentive are Graph Attention Networks?** ICLR 2022.
- D. Xu et al. **Inductive Representation Learning on Temporal Graphs.** ICLR 2020.
- A. Gu, K. Goel, C. Ré. **Efficiently Modeling Long Sequences with Structured State Spaces (S4).** ICLR 2022.
- M. Sarhan, S. Layeghy, M. Portmann. **Towards a Standard Feature Set for Network Intrusion Detection System Datasets.** Mobile Networks and Applications, 2022.
