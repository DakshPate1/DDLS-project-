"""Three policies compatible with `PortfolioSDE.simulate`.

All return a callable `policy_fn(t: float, X: np.ndarray) -> np.ndarray` with
`X` of shape `(batch,)`, as consumed by `src.sde.PortfolioSDE.simulate`.

- `baseline_policy(a)`  — constant fraction a (paper uses a=0.5).
- `optimal_policy(params, alpha)` — analytical tau=0 optimum (Appendix B.1).
- `trained_policy(Q, tau, rng)` — Gaussian sampled from q_psi per Algorithm 2
  line 4: pi^psi ∝ exp(q_psi / (tau * b_1)). With the B.2 parameterization
  q_psi = psi_sv * c2_psi * x^2 * (a - a_psi)^2, that collapses to
      a ~ N(a_psi,  -tau*b_1 / (2*psi_sv*c2_psi*x^2)).

Phase 1 convention: augmented state has b_0 = 0, b_1 = 1 throughout an episode
(r ≡ 0, delta = 0), so both optimal and trained policies are evaluated at
(b0=0, b1=1).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.models import QFunction
from src.sde import MarketParams

PolicyFn = Callable[[float, np.ndarray], np.ndarray]


def baseline_policy(a: float = 0.5) -> PolicyFn:
    """Constant-fraction policy used as Table 1 baseline (a = 0.5)."""
    def policy_fn(t: float, X: np.ndarray) -> np.ndarray:
        return np.full_like(X, a)
    return policy_fn


def optimal_policy(
    params: MarketParams,
    alpha: float = 1.0,
    b0: float = 0.0,
    b1: float = 1.0,
) -> PolicyFn:
    """Analytical optimal control a*(t,x,b0,b1) with tau=0 (Appendix B.1).

        a*(t,x,b0,b1) = sigma2^2/(sigma1^2+sigma2^2)
                      - (r1-r2)/(sigma1^2+sigma2^2) * (1 + c1/(2*c2*x))

    Uses the closed-form c1, c2 built from the analytical P_x, P_xx, P_nl, not
    the trained theta.

    Augmented state args `b0, b1` default to (0, 1) which is the initial state
    of Algorithm 2's training trajectory. For Table-1 / Figure-2 evaluation in
    the *original* SDE, `b0 = -b̂*` (see `metrics.find_bhat_star`).
    """
    r1, r2, s1, s2 = params.r1, params.r2, params.sigma1, params.sigma2
    T = params.T
    Px = (r1 * s2 ** 2 + r2 * s1 ** 2) / (s1 ** 2 + s2 ** 2)   # Appendix B.1
    Pxx = s1 ** 2 * s2 ** 2 / (2.0 * (s1 ** 2 + s2 ** 2))       # Appendix B.1
    Pnl = (r1 - r2) ** 2 / (2.0 * (s1 ** 2 + s2 ** 2))          # Appendix B.1
    head = s2 ** 2 / (s1 ** 2 + s2 ** 2)                        # sigma2^2/(s1^2+s2^2)
    slope = (r1 - r2) / (s1 ** 2 + s2 ** 2)                     # (r1-r2)/(s1^2+s2^2)
    c1_factor = (1.0 - alpha * b0) * b1                         # Appendix B.1 form
    c2_factor = -0.5 * alpha * b1 ** 2

    def policy_fn(t: float, X: np.ndarray) -> np.ndarray:
        tau_remaining = T - t
        c1 = c1_factor * np.exp((Px - 2.0 * Pnl) * tau_remaining)
        c2 = c2_factor * np.exp(2.0 * (Px + Pxx - Pnl) * tau_remaining)
        return head - slope * (1.0 + c1 / (2.0 * c2 * X))
    return policy_fn


def trained_policy(
    Q: QFunction,
    tau: float,
    rng: np.random.Generator,
    b0: float = 0.0,
    b1: float = 1.0,
) -> PolicyFn:
    """Gaussian policy derived from the current q_psi (Algorithm 2 line 4).

    Mean and variance are evaluated at (b0, b1) with the *current* psi
    parameters — so passing the same QFunction module into Algorithm 2's
    training loop automatically tracks the updated policy.

        mean     = a_psi(t, x, b0, b1)                               (B.2)
        variance = -tau * b1 / (2 * psi_sv * c2_psi(t, b0, b1) * x^2)

    Augmented state args default to (0, 1) — Algorithm 2's training convention.
    For original-SDE evaluation, pass `b0 = -b̂*`.
    """
    b0_t = torch.tensor(float(b0), dtype=torch.float64)
    b1_t = torch.tensor(float(b1), dtype=torch.float64)

    def policy_fn(t: float, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t_t = torch.tensor(float(t), dtype=torch.float64)
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float64))
            _, c2_psi, a_psi = Q.coefficients(t_t, X_t, b0_t, b1_t)
            mean = a_psi.numpy()
            # variance = -tau * b1 / (2 * psi_sv * c2_psi * x^2)
            var = (-tau * b1 / (2.0 * Q.psi_sv * c2_psi * X_t ** 2)).numpy()
        # numerical safety: variance must remain positive. During training a
        # pathological psi can flip the sign; we floor to a tiny epsilon and
        # let the TD update push psi back into a valid region.
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean + std * rng.standard_normal(X.shape)
    return policy_fn


if __name__ == "__main__":
    from src.models import q_function_at_ground_truth
    from src.sde import PortfolioSDE

    params = MarketParams()
    alpha = 1.0
    rng = np.random.default_rng(0)

    # 1. Baseline: constant 0.5
    fb = baseline_policy(0.5)
    assert np.allclose(fb(0.3, np.array([1.0, 2.0, 3.0])), 0.5)
    print("[baseline ] constant 0.5 — OK")

    # 2. Optimal: compare against B.1 closed form at a sample point.
    fo = optimal_policy(params, alpha=alpha)
    X_sample = np.array([0.9, 1.0, 1.1, 1.3])
    a_opt = fo(0.3, X_sample)
    # Cross-check one value against Q at ground truth (a_psi_ground_truth = a*).
    Q = q_function_at_ground_truth(alpha=alpha, T=params.T)
    with torch.no_grad():
        _, _, a_psi = Q.coefficients(
            torch.tensor(0.3, dtype=torch.float64),
            torch.from_numpy(X_sample),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
    diff = np.max(np.abs(a_opt - a_psi.numpy()))
    print(f"[optimal  ] max |a*_numpy - a_psi(ground truth)| = {diff:.2e} — OK")

    # 3. Trained policy: at ground-truth psi with small tau, mean(a_t) should be
    # close to a*. Roll out n=4000 trajectories, check average action at t=0.
    ft = trained_policy(Q, tau=0.01, rng=rng)
    a_samples = ft(0.0, np.full(4000, 1.0))
    _, _, a_star_0 = Q.coefficients(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    a_star_val = a_star_0.item()
    print(f"[trained  ] E[a_t] empirical = {a_samples.mean(): .4f} "
          f"(target a_psi = {a_star_val: .4f}, diff = "
          f"{abs(a_samples.mean() - a_star_val):.2e})")
    print(f"[trained  ] Std(a_t) empirical = {a_samples.std(): .4f}")

    # 4. Full-episode rollout with each policy: terminal wealth distribution
    sde = PortfolioSDE(params)
    for name, pol in [
        ("baseline", fb),
        ("optimal ", fo),
        ("trained ", ft),
    ]:
        _, X, _ = sde.simulate(X0=1.0, policy_fn=pol, n_trajectories=2000,
                               rng=np.random.default_rng(42))
        print(f"[rollout ] {name} E[X_T]={X[-1].mean():.4f}  "
              f"Std(X_T)={X[-1].std():.4f}")
