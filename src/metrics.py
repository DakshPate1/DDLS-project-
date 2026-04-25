"""Mean-variance objective, cumulative return, and OCE evaluation helpers.

Paper-target statistics:
- Mean-variance objective  MV(X_T) = E[X_T] - (alpha/2) Var(X_T)        (Eq. 36)
- Cumulative return        E[X_T] - X_0          (Table 1 "Cum Return")
- Terminal std             Std(X_T)              (Table 1 "Std Dev")
- Time paths of both       E[X_t]-X_0, MV(X_t)   (Figure 2)

OCE outer optimisation (Algorithm 1 line 5-8) — for the MV objective, the
outer dual variable b̂* equals E[X_T]^π at the self-consistent optimum:

    b̂*  = argmax_b { b + J*(t, x, -b, 1) }
        = argmax_b { b + E[φ(-b + X_T)] }  with φ(x) = x − α/2 x²
        = E[X_T]^π     (first-order condition w.r.t. b)

Hence `find_bhat_star` runs a fixed-point iteration: simulate under the policy
evaluated at b0 = −b̂_k, take the empirical mean of X_T, set b̂_{k+1} to that.
Usually converges in 2–3 iterations.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.sde import PortfolioSDE


def mean_variance_objective(X_T: np.ndarray, alpha: float) -> float:
    """MV(X_T) = E[X_T] - (alpha/2) * Var(X_T)   (Eq. 36)."""
    return float(np.mean(X_T) - 0.5 * alpha * np.var(X_T))


def terminal_cumulative_return(X_T: np.ndarray, X0: float) -> float:
    """Scalar cumulative return reported in Table 1: E[X_T] - X_0."""
    return float(np.mean(X_T) - X0)


def terminal_std(X_T: np.ndarray) -> float:
    """Std(X_T) — Table 1 "(Std. Dev.)" column."""
    return float(np.std(X_T))


def cumulative_return_path(X: np.ndarray, X0: float) -> np.ndarray:
    """E[X_t] - X_0 along the time grid (mean curve for Figure 2 top panel).

    X has shape (K+1, n_trajectories) as returned by `PortfolioSDE.simulate`.
    """
    return X.mean(axis=1) - X0


def cumulative_return_std_path(X: np.ndarray) -> np.ndarray:
    """Std(X_t) along the time grid (shaded band for Figure 2 top panel)."""
    return X.std(axis=1)


def mv_path(X: np.ndarray, alpha: float) -> np.ndarray:
    """MV(X_t) along the time grid (Figure 2 bottom panel).

    Starts at X_0 (no variance yet) and grows toward MV(X_T).
    """
    return X.mean(axis=1) - 0.5 * alpha * X.var(axis=1)


# -- OCE outer optimisation (b̂*) ------------------------------------------

PolicyFactory = Callable[[float], Callable[[float, np.ndarray], np.ndarray]]


def find_bhat_star(
    policy_factory: PolicyFactory,
    sde: PortfolioSDE,
    X0: float,
    n_trajectories: int,
    rng: np.random.Generator,
    max_iter: int = 6,
    tol: float = 1e-3,
    b_hat_init: float | None = None,
    verbose: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fixed-point iteration for the OCE dual variable b̂* under MV.

    Args:
        policy_factory: callable `b0 -> policy_fn`. Returns a policy evaluated
            at augmented state b0 (and b1=1 implicitly). Must close over its
            own parameters (Q-function / market params).
        sde, X0, n_trajectories, rng: as for `PortfolioSDE.simulate`.
        max_iter, tol: stopping criteria (iterate until |b̂_{k+1}−b̂_k| < tol
            or `max_iter` sweeps exhausted).
        b_hat_init: initial guess. If None, start from a baseline rollout so
            the first iterate is already in the right ballpark.

    Returns:
        b_hat:  converged b̂*.
        times:  (K+1,) time grid from the final rollout.
        X_final: (K+1, n_trajectories) trajectories under the converged policy.
    """
    if b_hat_init is None:
        # Warm start: rollout under b0=0 and use that E[X_T] as the seed.
        policy0 = policy_factory(0.0)
        _, X0_roll, _ = sde.simulate(X0, policy0, n_trajectories, rng)
        b_hat = float(X0_roll[-1].mean())
    else:
        b_hat = float(b_hat_init)

    X_final = None
    times = None
    for k in range(max_iter):
        # b0 = -b_hat per Algorithm 1: pi*_0 ( .| s,X_s,Y_s ) = pi*( . | s,X_s, Y_s - b̂*, 1)
        policy = policy_factory(-b_hat)
        times, X_final, _ = sde.simulate(X0, policy, n_trajectories, rng)
        new_b_hat = float(X_final[-1].mean())
        if verbose:
            print(f"  [bhat*] iter {k}: b̂_k = {b_hat:.5f} -> {new_b_hat:.5f} "
                  f"(|Δ| = {abs(new_b_hat - b_hat):.2e})")
        if abs(new_b_hat - b_hat) < tol:
            b_hat = new_b_hat
            break
        b_hat = new_b_hat

    assert X_final is not None and times is not None
    return b_hat, times, X_final


if __name__ == "__main__":
    from src.models import q_function_at_ground_truth
    from src.policies import baseline_policy, optimal_policy, trained_policy
    from src.sde import MarketParams

    params = MarketParams()
    sde = PortfolioSDE(params)
    X0 = 1.0
    alpha = 1.0

    # --- 1. Baseline (no b̂* needed — policy doesn't use b0) -----
    pol = baseline_policy(0.5)
    rng = np.random.default_rng(0)
    _, X, _ = sde.simulate(X0, pol, 10_000, rng)
    print(f"[baseline] cum return = {terminal_cumulative_return(X[-1], X0):.4f}")
    print(f"           std        = {terminal_std(X[-1]):.4f}")
    print(f"           MV         = {mean_variance_objective(X[-1], alpha):.4f}")

    # --- 2. Optimal under Algorithm 1 (b̂* fixed-point) -----------
    print("\n[optimal under b̂* fixed point]")
    rng = np.random.default_rng(1)
    b_hat, _, X_opt = find_bhat_star(
        policy_factory=lambda b0: optimal_policy(params, alpha=alpha, b0=b0, b1=1.0),
        sde=sde, X0=X0, n_trajectories=10_000, rng=rng, verbose=True,
    )
    print(f"  converged b̂* = {b_hat:.4f}")
    print(f"  cum return  = {terminal_cumulative_return(X_opt[-1], X0):.4f}")
    print(f"  std         = {terminal_std(X_opt[-1]):.4f}")
    print(f"  MV          = {mean_variance_objective(X_opt[-1], alpha):.4f}")

    # --- 3. Trained (ground-truth psi used here as a sanity check) --
    print("\n[trained=ground-truth psi under b̂* fixed point]")
    Q = q_function_at_ground_truth(alpha=alpha, T=params.T)
    rng = np.random.default_rng(2)
    b_hat_trn, _, X_trn = find_bhat_star(
        policy_factory=lambda b0: trained_policy(
            Q, tau=0.1, rng=rng, b0=b0, b1=1.0),
        sde=sde, X0=X0, n_trajectories=10_000, rng=rng, verbose=True,
    )
    print(f"  converged b̂* = {b_hat_trn:.4f}")
    print(f"  cum return  = {terminal_cumulative_return(X_trn[-1], X0):.4f}")
    print(f"  std         = {terminal_std(X_trn[-1]):.4f}")
    print(f"  MV          = {mean_variance_objective(X_trn[-1], alpha):.4f}")

    print("\nPaper Table 1 targets:")
    print("  Baseline   cum=0.2217 (std 0.0957)  MV=1.2171")
    print("  CT-RS-q    cum=0.8163 (std 0.8716)  MV=1.4365")
    print("  Optimal    cum=0.7128 (std 0.7205)  MV=1.4532")
