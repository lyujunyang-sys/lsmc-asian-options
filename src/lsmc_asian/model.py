"""Market model: Vasicek short rate and a lognormal asset, simulated exactly on a time grid.

Under the risk-neutral measure

    dr_t = kappa * (theta - r_t) dt + sigma_r dW^r_t
    dS_t = (r_t - q) S_t dt + sigma_s S_t dW^S_t,      d<W^r, W^S>_t = rho dt.

Over one step of length dt, given r_t, the triple

    (r_{t+dt},  I = integral_t^{t+dt} r_u du,  W^S_{t+dt} - W^S_t)

is jointly Gaussian with known mean and covariance, so the short rate, the
pathwise discount factor exp(-integral r) and the asset are sampled without
discretisation error. Monte Carlo prices therefore converge to the exact
model price, which the closed forms in ``analytic.py`` check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def vasicek_b(kappa: float, tau):
    """B(tau) = (1 - exp(-kappa*tau)) / kappa = integral_0^tau exp(-kappa*v) dv."""
    return -np.expm1(-kappa * np.asarray(tau, dtype=float)) / kappa


def _integrated_variance_factor(x):
    """f(x) = 1 - 2(1-e^-x)/x + (1-e^-2x)/(2x), so Var(I over dt) = sigma_r^2/kappa^2 * dt * f(kappa*dt).

    The direct formula cancels badly for small x; below 1e-3 the Taylor series is used.
    """
    x = float(x)
    if x < 1e-3:
        return x * x / 3.0 - x**3 / 4.0 + 7.0 * x**4 / 60.0
    return 1.0 + 2.0 * np.expm1(-x) / x - np.expm1(-2.0 * x) / (2.0 * x)


def psd_factor(cov: np.ndarray) -> np.ndarray:
    """A matrix L with L @ L.T == cov, valid for singular PSD matrices (e.g. sigma_r = 0 or |rho| = 1)."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    return vecs * np.sqrt(vals)


@dataclass(frozen=True)
class Paths:
    """Simulated paths on the grid ``times`` (shape (N+1,)); every array has shape (M, N+1).

    ``discount[:, j]`` is exp(-integral_0^{t_j} r_u du), the realised stochastic discount factor.
    """

    times: np.ndarray
    spot: np.ndarray
    short_rate: np.ndarray
    discount: np.ndarray

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot.shape[1] - 1


@dataclass(frozen=True)
class MarketModel:
    """Parameters of the Vasicek + lognormal asset model (annualised, rates as decimals)."""

    s0: float
    sigma_s: float
    r0: float
    kappa: float
    theta: float
    sigma_r: float
    rho: float = 0.0
    dividend_yield: float = 0.0

    def __post_init__(self):
        if self.s0 <= 0:
            raise ValueError("s0 must be positive")
        if self.sigma_s < 0 or self.sigma_r < 0:
            raise ValueError("volatilities must be non-negative")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive (mean reversion)")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")

    def step_covariance(self, dt: float) -> np.ndarray:
        """Covariance of (r_{t+dt}, integral of r over the step, W^S increment) given r_t."""
        k, sr, rho = self.kappa, self.sigma_r, self.rho
        b1 = float(vasicek_b(k, dt))  # integral_0^dt e^{-k v} dv
        b2 = -np.expm1(-2.0 * k * dt) / (2.0 * k)  # integral_0^dt e^{-2 k v} dv
        var_r = sr**2 * b2
        var_i = sr**2 / k**2 * dt * _integrated_variance_factor(k * dt)
        cov_ri = sr**2 / k * (b1 - b2)
        cov_rw = rho * sr * b1
        cov_iw = rho * sr * (dt - b1) / k
        return np.array(
            [
                [var_r, cov_ri, cov_rw],
                [cov_ri, var_i, cov_iw],
                [cov_rw, cov_iw, dt],
            ]
        )

    def simulate(
        self,
        maturity: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = False,
    ) -> Paths:
        """Simulate ``n_paths`` paths on an equally spaced grid of ``n_steps`` steps.

        With ``antithetic=True`` the second half of the paths uses the negated
        normals of the first half; path i and path i + n_paths/2 form a pair.
        """
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be positive")
        if antithetic and n_paths % 2:
            raise ValueError("antithetic sampling needs an even number of paths")

        dt = maturity / n_steps
        times = np.linspace(0.0, maturity, n_steps + 1)
        factor = psd_factor(self.step_covariance(dt))
        decay = np.exp(-self.kappa * dt)
        b1 = float(vasicek_b(self.kappa, dt))
        log_drift = -(self.dividend_yield + 0.5 * self.sigma_s**2) * dt
        n_draw = n_paths // 2 if antithetic else n_paths

        r = np.empty((n_paths, n_steps + 1))
        int_r = np.zeros((n_paths, n_steps + 1))
        log_s = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = self.r0
        for j in range(n_steps):
            z = rng.standard_normal((n_draw, 3))
            if antithetic:
                z = np.concatenate([z, -z])
            eps = z @ factor.T
            rj = r[:, j]
            r[:, j + 1] = self.theta + (rj - self.theta) * decay + eps[:, 0]
            step_int = self.theta * dt + (rj - self.theta) * b1 + eps[:, 1]
            int_r[:, j + 1] = int_r[:, j] + step_int
            log_s[:, j + 1] = log_s[:, j] + step_int + log_drift + self.sigma_s * eps[:, 2]

        return Paths(times=times, spot=self.s0 * np.exp(log_s), short_rate=r, discount=np.exp(-int_r))
