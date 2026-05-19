# Algorithms

This file specifies the four algorithmic pieces that together produce MambaGuard's certified guarantee. Each section gives (a) the formal statement / pseudocode, (b) the file in `mambaguard/certification/` that implements it, and (c) the relevant references.

---

## Algorithm 1 — MambaGuard online defence loop (Hedge MWU)

**Setting.** A defender chooses an action distribution $p_t\in\Delta(A_D)$ at each step $t=1,\ldots,T$. After acting, the defender observes a loss vector $\ell_t\in[0,1]^{|A_D|}$ derived from the MambaGuard verdict and the realised attacker action $a^A_t$. The goal is no-regret behaviour, i.e.

$$
R_T \;=\; \sum_{t=1}^{T} \langle p_t,\ell_t\rangle \;-\; \min_{i\in A_D}\sum_{t=1}^{T} \ell_t(i) \;=\; \mathcal{O}\!\bigl(\sqrt{T\ln|A_D|}\bigr).
$$

**Pseudocode** (`mambaguard/certification/hedge.py`):

```
Algorithm 1: MambaGuard online defence loop
Input:   stream of canonicalised messages {m_t}_{t>=1}
         trained detector f_theta with Lipschitz bound L_f
         Stackelberg mixed strategy pi* over A_D
         learning rate eta = sqrt( 2 * ln |A_D| / T )
State:   weights w_t in R_{>=0}^{|A_D|}, initialised w_1(i) = 1

for t = 1, 2, ..., T do
    # 1. detection
    y_hat_t  <- f_theta(G(t), m_t)            # softmax over 34 attack classes
    r_t      <- certified_radius(f_theta, m_t)  # randomised-smoothing radius

    # 2. policy: mix Hedge with Stackelberg anchor pi*
    p_t      <- w_t / sum(w_t)
    p_t      <- (1 - lambda_t) * p_t + lambda_t * pi*    # lambda_t = 1/sqrt(t)

    # 3. act: sample defender action a_t ~ p_t and apply
    a_t      <- sample(p_t)
    execute(a_t)                                # block / alert / isolate / ...

    # 4. observe loss and update
    ell_t    <- loss_vector(y_hat_t, a_t, observed_outcome)
    for i in A_D:
        w_{t+1}(i) <- w_t(i) * exp( -eta * ell_t(i) )
end for
```

**Guarantee.** Standard Hedge analysis (Freund and Schapire 1997) plus the Stackelberg anchor yields, with $B=\max_t \|\ell_t\|_\infty\leq 1$,

$$
\mathbb{E}\bigl[V_{\mathrm{ach}}(\hat\pi_T,\varepsilon)\bigr] \;\geq\; V^{*} \;-\; L_f\,\varepsilon \;-\; B\sqrt{\frac{\ln|A_D|}{2T}},
$$

which is the composed bound recovered in Section 4.

---

## Algorithm 2 — Lipschitz constant computation

**Statement.** Let $f_\theta = h\circ\mathrm{GAT}\circ\mathrm{SSM}^{L}$ be the composed detector. Under the assumptions

- spectrally normalised weights at each layer, $\|\mathbf{W}^{(\ell)}\|_2\leq 1+\lambda_L^{-1/2}\cdot \mathrm{slack}$;
- SiLU and GELU are 1-Lipschitz (gate constant $\kappa_g\leq 1$);
- the GATv2 attention output is $L_{\mathrm{gat}}$-Lipschitz in its concatenated input;
- the Hurwitz condition $\Delta_{\max}\,\alpha(\mathbf{A})<0$ gives a strict contraction rate $\rho<1$ on the SSM recurrence;

we obtain the composed bound

$$
L_f \;\leq\; M_h\cdot L_{\mathrm{gat}}\cdot \frac{\gamma\,\beta\,(1+\kappa_g)}{1-\rho},
$$

where $\beta=\sup_t \|\bar{\mathbf{B}}_t\|_2$, $\gamma=\sup_t\|\mathbf{C}_t\|_2$, and $M_h$ is the head Lipschitz constant.

**Pseudocode** (`mambaguard/certification/lipschitz.py`):

```
Algorithm 2: Lipschitz bound L_f
Input:  trained model f_theta with weight matrices {W^(l)}
        SSM parameters (A, B_t, C_t), discretisation bound Delta_max

# 1. per-layer spectral norms via power iteration (k = 30 iters)
for each linear / conv layer l do
    sigma_l <- power_iteration(W^(l), n_iter = 30)
end for
M_h <- prod_{l in head} sigma_l
L_gat <- spectral_norm(W_gatv2)

# 2. SSM contraction rate
rho <- max_t spectral_radius( bar_A_t )    # bar_A_t = exp(Delta_t * A)
assert rho < 1                              # Hurwitz check

beta  <- spectral_bound( bar_B_t )
gamma <- spectral_bound( C_t )
kappa_g <- 1.0                              # SiLU gate

L_f <- M_h * L_gat * gamma * beta * (1 + kappa_g) / (1 - rho)
return L_f
```

The values are reported alongside every checkpoint in `mambaguard/certification/lipschitz_report.csv`.

---

## Algorithm 3 — Stackelberg leader–follower LP

**Setting.** The defender (leader) commits to a mixed strategy $\pi\in\Delta(A_D)$. The attacker (follower) best-responds with $a^A\in A_A$ that maximises its expected payoff against $\pi$. The defender's robust value is

$$
V^{*} \;=\; \max_{\pi\in\Delta(A_D)} \min_{a^A\in A_A} \mathbb{E}_{a^D\sim\pi}\bigl[U_D(a^D,a^A)\bigr].
$$

This is a finite zero-sum (or general-sum, with the strong-Stackelberg tie-break of Conitzer and Sandholm 2006) game and is solved by **one LP per attacker action**, taking the best:

$$
\begin{aligned}
\text{for each } a^A\in A_A:\quad
& \max_{\pi,\, v}\; v \\
\text{s.t.}\quad
& \sum_{a^D\in A_D} \pi(a^D)\,U_D(a^D,a^A) \;\geq\; v, \\
& \sum_{a^D\in A_D} \pi(a^D)\,U_A(a^D,a^{A\prime}) \;\leq\; \sum_{a^D\in A_D} \pi(a^D)\,U_A(a^D,a^A)\quad \forall a^{A\prime}\neq a^A, \\
& \sum_{a^D} \pi(a^D) = 1,\quad \pi(a^D)\geq 0.
\end{aligned}
$$

The Stackelberg value is the maximum $v$ across the $|A_A|$ LPs.

**Pseudocode** (`mambaguard/certification/stackelberg.py`):

```
Algorithm 3: Strong Stackelberg LP solver
Input:  utility tensors U_D, U_A in R^{|A_D| x |A_A|}

best_v   <- -inf
best_pi  <- None
for a_A in A_A:
    # solve LP with cvxpy
    pi = Variable(|A_D|, nonneg=True)
    v  = Variable()
    constraints  = [ sum(pi) == 1 ]
    constraints += [ U_D[:, a_A] @ pi >= v ]
    for a_A_prime in A_A if a_A_prime != a_A:
        constraints += [ U_A[:, a_A_prime] @ pi <= U_A[:, a_A] @ pi ]
    Problem( Maximize(v), constraints ).solve(solver = 'ECOS')
    if pi.value is not None and v.value > best_v:
        best_v, best_pi = v.value, pi.value

return V_star = best_v,  pi_star = best_pi
```

For our deployment with $|A_D|=12$ and $|A_A|=34$ the full sweep takes < 1.8 s on a CPU.

---

## Algorithm 4 — Composed three-layer certificate

Putting the three pieces together gives the operational guarantee certified in Theorem 1 of the manuscript.

**Theorem 1 (composed certificate).** For any input perturbation $\delta$ with $\|\delta\|_2\leq\varepsilon$, the Hedge-Stackelberg policy $\hat\pi_T$ satisfies

$$
\boxed{\;
\mathbb{E}\bigl[V_{\mathrm{ach}}(\hat\pi_T,\varepsilon)\bigr] \;\geq\; V^{*} \;-\; L_f\,\varepsilon \;-\; B\sqrt{\dfrac{\ln|A_D|}{2T}}.
\;}
$$

The three penalty terms correspond, in order, to (i) randomised smoothing on the detector feature space (the certified $\ell_2$ radius bounds $L_f\varepsilon$), (ii) the Stackelberg robust value $V^*$ obtained from Algorithm 3, and (iii) the Hedge no-regret penalty from Algorithm 1.

**Implementation** (`mambaguard/certification/composed.py`):

```
Algorithm 4: composed_certificate
Input:  model f_theta, utilities U_D, U_A, eps, T, |A_D|

L_f      <- Algorithm 2 (Lipschitz bound)
V_star,_ <- Algorithm 3 (Stackelberg LP)
B        <- 1.0                                       # loss is in [0,1]
penalty  <- L_f * eps + B * sqrt( ln(|A_D|) / (2*T) )
return  V_lower_bound = V_star - penalty
```

The bound is reported in `outputs/cert_<timestamp>.json` and tracked across runs in `mambaguard/certification/cert_log.csv`.

---

## References

- Y. Freund, R. E. Schapire. **A Decision-Theoretic Generalisation of On-Line Learning and an Application to Boosting.** JCSS 1997.
- J. M. Cohen, E. Rosenfeld, J. Z. Kolter. **Certified Adversarial Robustness via Randomised Smoothing.** ICML 2019.
- V. Conitzer, T. Sandholm. **Computing the Optimal Strategy to Commit to.** EC 2006.
- M. Tambe. **Security and Game Theory: Algorithms, Deployed Systems, Lessons Learned.** Cambridge UP, 2011.
- N. Cesa-Bianchi, G. Lugosi. **Prediction, Learning, and Games.** Cambridge UP, 2006.
- T.-Y. Lin et al. **Focal Loss for Dense Object Detection.** ICCV 2017.
