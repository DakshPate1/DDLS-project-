"""Euler-Maruyama simulator for the portfolio-wealth SDE (Eq. 35).

The controlled wealth process X_t evolves as (Eq. 35):
    dX_t = X_t * (a_t*r1 + (1-a_t)*r2) * dt
         + X_t * (a_t*sigma1*dW_{1,t} + (1-a_t)*sigma2*dW_{2,t})
with W_1, W_2 independent standard Brownian motions. Euler-Maruyama:
    X_{k+1} = X_k + X_k*drift_k*dt + X_k*(a_k*sigma1*sqrt(dt)*z1
                                        + (1-a_k)*sigma2*sqrt(dt)*z2)
where z1, z2 ~ N(0,1) iid.

For Phase 1 the instantaneous reward r is 0 and discount delta is 0, so the
augmented state components are fixed at B0=0, B1=1 throughout an episode
(PROJECT_CONTEXT.md, "Key contribution 1").
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MarketParams:
    r1: float = 0.15
    r2: float = 0.25
    sigma1: float = 0.10
    sigma2: float = 0.12
    T: float = 1.0
    dt: float = 0.001

    @property
    def K(self) -> int:
        return int(round(self.T / self.dt))


class PortfolioSDE:
    """Euler-Maruyama simulator for the two-asset portfolio wealth SDE."""

    def __init__(self, params: MarketParams):
        self.p = params
        self.sqrt_dt = np.sqrt(params.dt)

    def step(
        self,
        X: np.ndarray,
        a: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray,
    ) -> np.ndarray:
        """Single Euler-Maruyama step (Eq. 35). All inputs shape (batch,) or scalar."""
        p = self.p
        drift = a * p.r1 + (1.0 - a) * p.r2
        diffusion = a * p.sigma1 * z1 + (1.0 - a) * p.sigma2 * z2
        return X + X * drift * p.dt + X * diffusion * self.sqrt_dt

    def simulate(
        self,
        X0: float,
        policy_fn: Callable[[float, np.ndarray], np.ndarray],
        n_trajectories: int,
        rng: np.random.Generator,
    ):
        """Simulate n_trajectories in parallel under policy_fn(t, X_batch) -> a_batch.

        Returns:
            times: (K+1,) array of time grid values.
            X:     (K+1, n_trajectories) wealth trajectories.
            A:     (K,   n_trajectories) actions taken at each step.
        """
        p = self.p
        K = p.K
        times = np.linspace(0.0, p.T, K + 1)
        X = np.empty((K + 1, n_trajectories), dtype=np.float64)
        A = np.empty((K, n_trajectories), dtype=np.float64)
        X[0] = X0

        for k in range(K):
            a_k = policy_fn(times[k], X[k])
            z1 = rng.standard_normal(n_trajectories)
            z2 = rng.standard_normal(n_trajectories)
            X[k + 1] = self.step(X[k], a_k, z1, z2)
            A[k] = a_k
        return times, X, A


if __name__ == "__main__":
    # Quick sanity check: under the baseline a=0.5 policy, mean terminal wealth
    # should be close to the analytical expectation exp(0.5*(r1+r2)*T).
    params = MarketParams()
    sde = PortfolioSDE(params)
    rng = np.random.default_rng(0)
    times, X, A = sde.simulate(
        X0=1.0,
        policy_fn=lambda t, x: np.full_like(x, 0.5),
        n_trajectories=5000,
        rng=rng,
    )
    expected = np.exp(0.5 * (params.r1 + params.r2) * params.T)
    print(f"Baseline a=0.5 | E[X_T] empirical = {X[-1].mean():.4f} "
          f"(analytic ~ {expected:.4f})")
    print(f"                 Std(X_T)          = {X[-1].std():.4f}")
    print(f"                 trajectories shape {X.shape}, actions shape {A.shape}")
