"""J_theta and q_psi parameterizations (Appendix B.2 of the paper).

Both modules expose torch.nn.Parameter tensors so PyTorch autograd can compute
dJ_theta/dtheta and dq_psi/dpsi as required by Algorithm 2 lines 6-9.

Value function (cf. B.2):
    J_theta(t, x, b0, b1) = c0(t,b0,b1) + c1(t,b0,b1)*x + c2(t,b0,b1)*x^2
with theta = (theta_Px, theta_Pxx, theta_Pnl).

Q-function (cf. B.2):
    q_psi(t, x, b0, b1, a) = psi_sv * c2_psi(t,b0,b1) * x^2 * (a - a_psi(t,x,b0,b1))^2
with psi = (psi_a0, psi_a1, psi_sv, psi_ce1, psi_ce2).

Ground-truth parameter values (B.3) are provided at the bottom.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _as_param(value: float) -> nn.Parameter:
    return nn.Parameter(torch.tensor(float(value), dtype=torch.float64))


class ValueFunction(nn.Module):
    """J_theta(t, x, b0, b1) = c0 + c1*x + c2*x^2  (Appendix B.2)."""

    def __init__(
        self,
        alpha: float,
        T: float,
        theta_Px: float,
        theta_Pxx: float,
        theta_Pnl: float,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.T = float(T)
        self.theta_Px = _as_param(theta_Px)
        self.theta_Pxx = _as_param(theta_Pxx)
        self.theta_Pnl = _as_param(theta_Pnl)

    def coefficients(self, t, b0, b1):
        """Return (c0, c1, c2) at (t, b0, b1). Inputs broadcast-compatible tensors."""
        alpha = self.alpha
        tau = self.T - t  # time remaining to horizon
        pxx_plus_pnl = self.theta_Pxx + self.theta_Pnl

        # c2(t,b0,b1) = -(alpha/2) * b1^2 * exp(2*(Px+Pxx-Pnl)*(T-t))    (B.2)
        c2 = -0.5 * alpha * b1 ** 2 * torch.exp(
            2.0 * (self.theta_Px + self.theta_Pxx - self.theta_Pnl) * tau
        )
        # c1(t,b0,b1) = (1 - alpha*b0) * b1 * exp((Px - 2*Pnl)*(T-t))    (B.2)
        c1 = (1.0 - alpha * b0) * b1 * torch.exp(
            (self.theta_Px - 2.0 * self.theta_Pnl) * tau
        )
        # c0(t,b0,b1) = b0*(1 - alpha/2*b0)
        #             + (1-alpha*b0)^2 * Pnl / (2*alpha*(Pxx+Pnl))
        #               * (1 - exp(-2*(Pxx+Pnl)*(T-t)))                  (B.2)
        c0 = b0 * (1.0 - 0.5 * alpha * b0) + (
            (1.0 - alpha * b0) ** 2
            * self.theta_Pnl
            / (2.0 * alpha * pxx_plus_pnl)
            * (1.0 - torch.exp(-2.0 * pxx_plus_pnl * tau))
        )
        return c0, c1, c2

    def forward(self, t, x, b0, b1):
        c0, c1, c2 = self.coefficients(t, b0, b1)
        return c0 + c1 * x + c2 * x ** 2


class QFunction(nn.Module):
    """q_psi(t, x, b0, b1, a) = psi_sv * c2_psi * x^2 * (a - a_psi)^2  (Appendix B.2)."""

    def __init__(
        self,
        alpha: float,
        T: float,
        psi_a0: float,
        psi_a1: float,
        psi_sv: float,
        psi_ce1: float,
        psi_ce2: float,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.T = float(T)
        self.psi_a0 = _as_param(psi_a0)
        self.psi_a1 = _as_param(psi_a1)
        self.psi_sv = _as_param(psi_sv)
        self.psi_ce1 = _as_param(psi_ce1)
        self.psi_ce2 = _as_param(psi_ce2)

    def coefficients(self, t, x, b0, b1):
        """Return (c1_psi, c2_psi, a_psi) at (t, x, b0, b1)."""
        alpha = self.alpha
        tau = self.T - t
        # c1_psi(t,b0,b1) = (1 - alpha*b0) * b1 * exp(psi_ce1*(T-t))     (B.2)
        c1_psi = (1.0 - alpha * b0) * b1 * torch.exp(self.psi_ce1 * tau)
        # c2_psi(t,b0,b1) = -(alpha/2) * b1^2 * exp(psi_ce2*(T-t))       (B.2)
        c2_psi = -0.5 * alpha * b1 ** 2 * torch.exp(self.psi_ce2 * tau)
        # a_psi(t,x,b0,b1) = psi_a0 - psi_a1 * (1 + c1_psi/(2*c2_psi*x)) (B.2)
        a_psi = self.psi_a0 - self.psi_a1 * (
            1.0 + c1_psi / (2.0 * c2_psi * x)
        )
        return c1_psi, c2_psi, a_psi

    def forward(self, t, x, b0, b1, a):
        _, c2_psi, a_psi = self.coefficients(t, x, b0, b1)
        return self.psi_sv * c2_psi * x ** 2 * (a - a_psi) ** 2


# Ground-truth parameters at the analytical optimum (Appendix B.3).
# Derived from market params (r1=0.15, r2=0.25, sigma1=0.10, sigma2=0.12):
#   Px  = (r1*sigma2^2 + r2*sigma1^2) / (sigma1^2+sigma2^2)     = 0.1910
#   Pxx = sigma1^2*sigma2^2 / (2*(sigma1^2+sigma2^2))            = 0.0030
#   Pnl = (r1-r2)^2 / (2*(sigma1^2+sigma2^2))                    = 0.2049
GROUND_TRUTH_THETA = dict(theta_Px=0.1910, theta_Pxx=0.0030, theta_Pnl=0.2049)
GROUND_TRUTH_PSI = dict(
    psi_a0=0.5902,   # = sigma2^2/(sigma1^2+sigma2^2)
    psi_a1=-4.0984,  # = (r1-r2)/(sigma1^2+sigma2^2)
    psi_sv=0.0244,   # = sigma1^2+sigma2^2
    psi_ce1=-0.2189, # = Px - 2*Pnl
    psi_ce2=-0.0220, # = 2*(Px + Pxx - Pnl)
)


def value_function_at_ground_truth(alpha: float = 1.0, T: float = 1.0) -> ValueFunction:
    return ValueFunction(alpha=alpha, T=T, **GROUND_TRUTH_THETA)


def q_function_at_ground_truth(alpha: float = 1.0, T: float = 1.0) -> QFunction:
    return QFunction(alpha=alpha, T=T, **GROUND_TRUTH_PSI)


if __name__ == "__main__":
    # Smoke test: at ground-truth parameters, the optimal analytical action
    # should fall out of a_psi. Compare against the closed form from B.1:
    #   a*(t,x,b0,b1) = sigma2^2/(s1^2+s2^2) - (r1-r2)/(s1^2+s2^2)*(1 + c1/(2 c2 x))
    # evaluated with the *optimal* c1, c2 from B.1.
    import numpy as np

    alpha, T = 1.0, 1.0
    r1, r2, s1, s2 = 0.15, 0.25, 0.10, 0.12
    Px = (r1 * s2 ** 2 + r2 * s1 ** 2) / (s1 ** 2 + s2 ** 2)
    Pxx = s1 ** 2 * s2 ** 2 / (2.0 * (s1 ** 2 + s2 ** 2))
    Pnl = (r1 - r2) ** 2 / (2.0 * (s1 ** 2 + s2 ** 2))

    J = value_function_at_ground_truth(alpha, T)
    Q = q_function_at_ground_truth(alpha, T)

    # Phase 1 convention: b0=0, b1=1 throughout.
    t = torch.tensor(0.3, dtype=torch.float64)
    x = torch.tensor(1.1, dtype=torch.float64)
    b0 = torch.tensor(0.0, dtype=torch.float64)
    b1 = torch.tensor(1.0, dtype=torch.float64)

    # True optimal coefficients (B.1) at (t, b0=0, b1=1):
    c1_true = (1.0 - alpha * 0.0) * 1.0 * np.exp((Px - 2.0 * Pnl) * (T - 0.3))
    c2_true = -0.5 * alpha * 1.0 * np.exp(2.0 * (Px + Pxx - Pnl) * (T - 0.3))
    a_star_true = (s2 ** 2) / (s1 ** 2 + s2 ** 2) - (r1 - r2) / (s1 ** 2 + s2 ** 2) * (
        1.0 + c1_true / (2.0 * c2_true * float(x))
    )

    with torch.no_grad():
        _, _, a_psi = Q.coefficients(t, x, b0, b1)
        c0, c1, c2 = J.coefficients(t, b0, b1)

    print(f"a_psi(ground truth)       = {a_psi.item(): .6f}")
    print(f"a*(analytical, B.1)       = {a_star_true: .6f}")
    print(f"|diff|                    = {abs(a_psi.item() - a_star_true):.2e}")
    print(f"J_theta c1 vs B.1 c1      = {c1.item(): .6f}  vs  {c1_true: .6f}")
    print(f"J_theta c2 vs B.1 c2      = {c2.item(): .6f}  vs  {c2_true: .6f}")

    # Confirm autograd wiring: a tiny scalar computation should produce
    # non-zero gradients on all 3 theta and all 5 psi parameters.
    a_action = torch.tensor(0.4, dtype=torch.float64)
    loss = J(t, x, b0, b1) + Q(t, x, b0, b1, a_action)
    loss.backward()
    theta_grads = [p.grad for p in (J.theta_Px, J.theta_Pxx, J.theta_Pnl)]
    psi_grads = [p.grad for p in (Q.psi_a0, Q.psi_a1, Q.psi_sv, Q.psi_ce1, Q.psi_ce2)]
    print("theta grads non-zero:", all(g is not None and g.abs() > 0 for g in theta_grads))
    print("psi   grads non-zero:", all(g is not None and g.abs() > 0 for g in psi_grads))
