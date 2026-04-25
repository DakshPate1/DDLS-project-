# PROJECT_CONTEXT.md
# Fed-CT-RS-q: Federated & Decentralized Risk-Sensitive Q-Learning in Continuous Time
# Course: Distributed Deep Learning Systems, UniBE Spring 2026
# Student: Daksh Patel (solo)

---

## KICKSTART PROMPT
# Copy everything below this line and paste as your first message to Claude Code.
# Attach this file + the paper PDF (2512.02386v1.pdf) alongside it.

---

I am a solo MSc student reproducing and extending a research paper for a graduate
course in Distributed Deep Learning Systems at Universität Bern. I have attached
two files: this PROJECT_CONTEXT.md (full spec, equations, ground truth) and the
paper PDF (2512.02386v1.pdf).

Read both files completely before writing any code.

Build this project in three strict phases. Do not begin a phase until the previous
one is numerically verified. At the end of each phase, print a verification
summary showing learned vs. ground truth values.

─────────────────────────────────────────────
PHASE 1 — REPRODUCE THE PAPER (start here)
─────────────────────────────────────────────
Build these files in order:
  src/sde.py           — Euler-Maruyama SDE simulator
  src/models.py        — J_theta and q_psi parameterizations (exact Appendix B.2)
  src/policies.py      — analytical optimal policy, baseline (a=0.5), trained policy
  src/metrics.py       — MV objective, cumulative return
  src/ct_rs_q.py       — Algorithm 2 training loop, full episode, TD updates
  experiments/reproduce.py — generates Table 1, Figure 1, Figure 2

Verification gate: learned parameters must be within 10% of ground truth values
listed in this file before Phase 1 is declared complete.

─────────────────────────────────────────────
PHASE 2 — FEDERATED EXTENSION
─────────────────────────────────────────────
Build src/federated.py implementing three methods:
  1. Local CT-RS-q — each worker trains independently, no communication
  2. Vanilla FedAvg CT-RS-q — central server averages all parameters each round
  3. Fed-CT-RS-q (proposed) — regime-aware proximal weighting + shared/local split

Build experiments/federated_exp.py comparing all three on:
  - Each worker's own market (own-regime performance)
  - Held-out market regime (generalization)

─────────────────────────────────────────────
PHASE 3 — DECENTRALIZED EXTENSION
─────────────────────────────────────────────
Build src/decentralized.py implementing gossip-based Dec-CT-RS-q:
  - Ring topology (each worker communicates with 2 neighbors)
  - Fully connected topology
  - Metropolis-Hastings mixing weights
  - No central server

Build experiments/decentralized_exp.py comparing Dec-CT-RS-q against Phase 2 results.

─────────────────────────────────────────────
CODING RULES (enforce throughout)
─────────────────────────────────────────────
- Use PyTorch autograd for ALL gradients of J_theta and q_psi
- Implement every equation exactly as specified in this file
- Add inline comments citing paper equation numbers (e.g., # Eq. 35)
- Save all figures to plots/ and all numerical results to results/
- Never use external RL libraries — implement Algorithm 2 from scratch
- Start by showing me src/sde.py

---

## COURSE CONTEXT

Course: Distributed Deep Learning Systems, UniBE Spring 2026
Instructor: Professor Chen
Project weight: 70% of final grade

Grading breakdown:
  Final report:         65%  (7-8 pages, ICML 2025 LaTeX template)
  Presentation:         15%  (20 min, Week 15)
  Individual contrib:   20%

Key dates:
  Week 10 (Apr 21):  Intermediate presentation (ungraded checkpoint)
  Week 11 (Apr 28):  Project midterm presentation
  Week 15 (May 26):  Final report + presentation

Course modules covered by this project (Chen requires 2 of 3):
  Module A — Federated Learning (Weeks 3-6: HFL, VFL, FedGenAI)
             → covered by Phase 2: FedAvg + Fed-CT-RS-q
  Module B — Decentralized RL (Week 7)
             → covered by Phase 3: gossip-based Dec-CT-RS-q

Report structure:
  1. Motivation
  2. Background of the paper
  3. Method proposed by the paper
  4. Reproducing results (figures + tables)
  5. Proposed improvement (weakness + fix)
  6. Conclusions + future directions
  7. References
  Must include: summary figure, link to code repo

---

## SELECTED PAPER

Title:  Risk-Sensitive Q-Learning in Continuous Time with Application
        to Dynamic Portfolio Selection
Author: Chuhan Xie (Peking University)
Venue:  NeurIPS 2025 Workshop: Generative AI in Finance
ArXiv:  arxiv.org/abs/2512.02386

---

## PAPER SUMMARY

The paper extends Q-learning to continuous time with risk-sensitive objectives.

Environment: controlled SDE
  dX_s = μ(s,X_s,a_s)dt + σ(s,X_s,a_s)dW_s

Objective: maximize OCE (Optimized Certainty Equivalent) of cumulative rewards,
           not just expected reward. Generalizes mean-variance, CVaR, entropic risk.

Key contribution 1 — Augmented Markov state:
  Under OCE objectives, optimal policy is Markovian in augmented state
  (t, X_t, B0_t, B1_t) where B0 tracks cumulative reward, B1 tracks discounting.
  In this paper's portfolio problem: r≡0, δ=0, so B0=0 and B1=1 throughout.
  Augmented state simplifies to just (t, X_t, 0, 1).

Key contribution 2 — Martingale characterization:
  J_hat = J* and q_hat = q* iff this process is a martingale:
  J_hat(s, X_s, Y_s, e^{-δ(s-t)}) - ∫q_hat(u,X_u,Y_u,e^{-δ(u-t)},a_u)du

Key contribution 3 — CT-RS-q algorithm (Algorithm 2):
  On-policy, parameterizes J_theta and q_psi simultaneously,
  updates via averaged TD errors per episode.

---

## PHASE 1: REPRODUCTION TARGETS

### Market parameters (Appendix B.3)
r1 = 0.15
r2 = 0.25
sigma1 = 0.10
sigma2 = 0.12
T = 1.0
dt = 0.001          # timestep (K=1000 steps per episode)
alpha = 1.0         # risk aversion
N_episodes = 10000
X0 = 1.0
tau = 0.1           # exploration temperature (start here, tune if needed)
lr_theta = 1e-3     # tune if convergence doesn't match Figure 1
lr_psi   = 1e-3

### Ground truth optimal parameters (Appendix B.3)
Px  = 0.1910    # = (r1*sigma2^2 + r2*sigma1^2) / (sigma1^2 + sigma2^2)
Pxx = 0.0030    # = sigma1^2*sigma2^2 / (2*(sigma1^2+sigma2^2))
Pnl = 0.2049    # = (r1-r2)^2 / (2*(sigma1^2+sigma2^2))

theta*_Px  = 0.1910
theta*_Pxx = 0.0030
theta*_Pnl = 0.2049
psi*_a0    = 0.5902
psi*_a1    = -4.0984
psi*_sv    = 0.0244
psi*_ce1   = -0.2189
psi*_ce2   = -0.0220

### Reproduce these exactly
Table 1: Cumulative return + MV objective for Baseline / CT-RS-q / Optimal
Figure 1: 8-panel parameter convergence plot (3 theta params + 5 psi params)
Figure 2: Time evolution of cumulative return and MV objective for 3 policies

---

## KEY EQUATIONS

### SDE — portfolio wealth dynamics (Eq. 35)
dX_t = X_t * (a_t*r1 + (1-a_t)*r2) * dt
     + X_t * (a_t*sigma1*dW1 + (1-a_t)*sigma2*dW2)

Euler-Maruyama discretization:
X_{k+1} = X_k + X_k*(a_k*r1 + (1-a_k)*r2)*dt
         + X_k*(a_k*sigma1*sqrt(dt)*z1 + (1-a_k)*sigma2*sqrt(dt)*z2)
where z1,z2 ~ N(0,1) iid

### Value function parameterization (Appendix B.2)
J_theta(t, x, b0, b1) = c0 + c1*x + c2*x^2

c2(t,b0,b1) = -(alpha/2) * b1^2 * exp(2*(theta_Px + theta_Pxx - theta_Pnl)*(T-t))
c1(t,b0,b1) = (1 - alpha*b0) * b1 * exp((theta_Px - 2*theta_Pnl)*(T-t))
c0(t,b0,b1) = b0*(1 - alpha/2*b0)
            + (1-alpha*b0)^2 * theta_Pnl / (2*alpha*(theta_Pxx + theta_Pnl))
            * (1 - exp(-2*(theta_Pxx + theta_Pnl)*(T-t)))

### Q-function parameterization (Appendix B.2)
q_psi(t, x, b0, b1, a) = psi_sv * c2_psi * x^2 * (a - a_psi)^2

c1_psi(t,b0,b1) = (1 - alpha*b0) * b1 * exp(psi_ce1*(T-t))
c2_psi(t,b0,b1) = -(alpha/2) * b1^2 * exp(psi_ce2*(T-t))
a_psi(t,x,b0,b1) = psi_a0 - psi_a1 * (1 + c1_psi / (2*c2_psi*x))

### Policy — Gaussian derived from q_psi
Mean:     a_psi(t,x,b0,b1)   [as above]
Variance: -tau*b1 / (2*psi_sv*c2_psi*x^2)
Sample:   a_t ~ N(mean, variance)

### Analytical optimal policy (Appendix B.1, tau=0)
a*(t,x,b0,b1) = sigma2^2/(sigma1^2+sigma2^2)
              - (r1-r2)/(sigma1^2+sigma2^2) * (1 + c1/(2*c2*x))
using true c1, c2 computed from Px, Pxx, Pnl above

### CT-RS-q TD error (Algorithm 2, line 9)
TD_k = J_theta(t_{k+1}, X_{k+1}, B0_{k+1}, B1_{k+1})
     - J_theta(t_k,     X_k,     B0_k,     B1_k    )
     - q_psi(t_k, X_k, B0_k, B1_k, a_k) * dt

### Parameter updates (Algorithm 2, lines 9-10)
delta_theta = sum_k [ (dJ_theta/dtheta at step k) * TD_k ]
delta_psi   = sum_k [ (dq_psi/dpsi   at step k) * TD_k ]
theta <- theta + lr_theta * delta_theta
psi   <- psi   + lr_psi   * delta_psi

### Mean-variance objective (Eq. 36)
MV(X_T) = E[X_T] - (alpha/2) * Var(X_T)

---

## PHASE 2: FEDERATED EXTENSION

### Worker market configurations
Worker 1: r1=0.15, r2=0.25, σ1=0.10, σ2=0.12  (paper baseline)
Worker 2: r1=0.30, r2=0.10, σ1=0.20, σ2=0.15  (high drift spread)
Worker 3: r1=0.05, r2=0.08, σ1=0.05, σ2=0.06  (low volatility)
Worker 4: r1=0.20, r2=0.20, σ1=0.25, σ2=0.30  (high volatility, equal drift)
Held-out: r1=0.12, r2=0.18, σ1=0.15, σ2=0.18  (never seen during training)

Federation hyperparameters:
  K = 4 workers
  R = 20 federation rounds
  E = 500 local episodes per round
  mu = 0.01  (proximal penalty base)

### Method 1: Local CT-RS-q
Each worker runs standard CT-RS-q independently.
No communication. No aggregation.
Evaluate on own market AND held-out market.

### Method 2: Vanilla FedAvg CT-RS-q
Each round:
  1. Broadcast global (theta, psi) to all workers
  2. Each worker runs E local episodes of CT-RS-q
  3. Server averages: theta_global = (1/K) * sum_k theta_k
                      psi_global   = (1/K) * sum_k psi_k
  4. Repeat R rounds

### Method 3: Fed-CT-RS-q (proposed)
Component A — Regime-aware proximal weighting:
  phi_k = (r1_k, r2_k, sigma1_k, sigma2_k)  # regime vector for worker k
  phi_bar = mean(phi_k over all k)
  lambda_k = ||phi_k - phi_bar|| / max_j(||phi_j - phi_bar||)  # in [0,1]

  Each worker's local update adds proximal correction to SHARED params only:
  delta_theta_sh -= mu * lambda_k * (theta_sh_k - theta_sh_global)
  delta_psi_sh   -= mu * lambda_k * (psi_sh_k   - psi_sh_global)

Component B — Shared/local parameter split:
  SHARED (federated, aggregated each round):
    theta_Pxx, psi_sv
    (pure volatility structure — generalizes across regimes)
  
  LOCAL (kept on worker, never shared):
    theta_Px, theta_Pnl, psi_a0, psi_a1, psi_ce1, psi_ce2
    (drift-dependent, regime-specific)

  Aggregation: average SHARED params only
  theta_sh_global = (1/K) * sum_k theta_sh_k
  psi_sh_global   = (1/K) * sum_k psi_sh_k

### Evaluation metrics
  - MV(X_T) on own market (higher is better)
  - MV(X_T) on held-out market (measures generalization)
  - Convergence curve: MV vs. federation round
  - Parameter recovery: ||theta_k - theta*_k||

---

## PHASE 3: DECENTRALIZED EXTENSION

### Architecture
No central server. Workers communicate peer-to-peer on a graph G=(V,E).
Each worker k aggregates with neighbors N(k) using mixing weights w_{kj}.

### Mixing weights — Metropolis-Hastings
w_{kj} = 1 / (1 + max(deg(k), deg(j)))   for j in N(k), j != k
w_{kk} = 1 - sum_{j in N(k)} w_{kj}

### Gossip update (one round)
theta_k_new = sum_{j in N(k) union {k}} w_{kj} * theta_j
psi_k_new   = sum_{j in N(k) union {k}} w_{kj} * psi_j

### Topologies to test
  Ring:           0-1-2-3-0  (each worker has 2 neighbors)
  Fully connected: all workers connected to all others

### Experiment
  Same 4 workers + held-out setup as Phase 2
  Compare: Local vs FedAvg vs Fed-CT-RS-q vs Dec-CT-RS-q (ring) vs Dec-CT-RS-q (FC)
  Metrics: same as Phase 2 + communication cost (bytes per round)

---

## FILE STRUCTURE

project/
├── src/
│   ├── sde.py              # Euler-Maruyama simulator
│   ├── models.py           # J_theta, q_psi parameterizations
│   ├── policies.py         # analytical optimal, baseline, trained policy
│   ├── metrics.py          # MV objective, cumulative return
│   ├── ct_rs_q.py          # Algorithm 2 training loop
│   ├── federated.py        # Local, FedAvg, Fed-CT-RS-q
│   └── decentralized.py    # Dec-CT-RS-q, graph topologies
├── experiments/
│   ├── reproduce.py        # Table 1, Figure 1, Figure 2
│   ├── federated_exp.py    # Phase 2 experiments
│   └── decentralized_exp.py # Phase 3 experiments
├── plots/                  # all output figures
├── results/                # all numerical results (CSV or JSON)
├── PROJECT_CONTEXT.md      # this file
└── 2512.02386v1.pdf        # paper

---

## STACK
- Python 3.10+
- PyTorch (autograd for all gradients — do not compute gradients manually)
- NumPy (SDE simulation)
- Matplotlib (all plots)
- No external RL libraries
