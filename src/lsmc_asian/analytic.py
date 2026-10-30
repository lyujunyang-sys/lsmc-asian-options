"""Closed-form results under the Vasicek + lognormal model.

The log asset price and the integrated short rate are jointly Gaussian, so for
any payoff of the form (exp(Y) - K)^+ paid at T, where Y is a linear
combination of log prices on the monitoring grid and X = integral_0^T r_u du,

    E[exp(-X) (exp(Y) - K)^+]

has a Black-type closed form. This gives exact prices for the discretely
monitored geometric-average Asian option and the European option with
stochastic rates and rate/asset correlation. They serve as benchmarks for the
simulation and as the control variate for the arithmetic-average option.

Monitoring convention used throughout the package: averages are taken over
t_i = i * T / N, i = 1..N (the initial price S_0 is not included).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .model import MarketModel, vasicek_b


def zero_coupon_bond(model: MarketModel, maturity: float) -> float:
    """Vasicek zero-coupon bond price P(0, T) = A(0,T) exp(-B(0,T) r0)."""
    k, theta, s = model.kappa, model.theta, model.sigma_r
    b = float(vasicek_b(k, maturity))
    log_a = (theta - s**2 / (2 * k**2)) * (b - maturity) - s**2 * b**2 / (4 * k)
    return float(np.exp(log_a - b * model.r0))


def black_scholes(s0, strike, maturity, rate, sigma, dividend_yield=0.0, kind="call") -> float:
    """Black-Scholes price with a constant rate (used as a reference in tests)."""
    vol = sigma * np.sqrt(maturity)
    d1 = (np.log(s0 / strike) + (rate - dividend_yield) * maturity) / vol + 0.5 * vol
    d2 = d1 - vol
    df, dq = np.exp(-rate * maturity), np.exp(-dividend_yield * maturity)
    if kind == "call":
        return float(s0 * dq * norm.cdf(d1) - strike * df * norm.cdf(d2))
    if kind == "put":
        return float(strike * df * norm.cdf(-d2) - s0 * dq * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")


@dataclass(frozen=True)
class GaussianStructure:
    """Means and covariances of I(t_i) = integral_0^{t_i} r and W^S(t_i) on a grid of times."""

    times: np.ndarray
    mean_int_rate: np.ndarray
    cov_ii: np.ndarray  # Cov(I(t_a), I(t_b))
    cov_iw: np.ndarray  # Cov(I(t_a), W^S(t_b))
    cov_ww: np.ndarray  # Cov(W^S(t_a), W^S(t_b)) = min(t_a, t_b)

    @classmethod
    def build(cls, model: MarketModel, times) -> "GaussianStructure":
        t = np.asarray(times, dtype=float)
        k, theta, sr = model.kappa, model.theta, model.sigma_r
        mean_i = theta * t + (model.r0 - theta) * vasicek_b(k, t)

        s, u = np.meshgrid(t, t, indexing="ij")
        m = np.minimum(s, u)
        decay = np.exp(-k * np.abs(s - u))
        b2 = -np.expm1(-2 * k * m) / (2 * k)
        cov_ii = sr**2 / k**2 * (m - (1 + decay) * vasicek_b(k, m) + decay * b2)

        # integral_0^{min(s,u)} B(s - v) dv, with s the time of I and u the time of W
        g = np.where(
            s <= u,
            (s - vasicek_b(k, s)) / k,
            u / k + (np.exp(-k * s) - np.exp(-k * (s - u))) / k**2,
        )
        cov_iw = model.rho * sr * g
        return cls(times=t, mean_int_rate=mean_i, cov_ii=cov_ii, cov_iw=cov_iw, cov_ww=m)

    def log_price_mean(self, model: MarketModel) -> np.ndarray:
        drift = (model.dividend_yield + 0.5 * model.sigma_s**2) * self.times
        return np.log(model.s0) + self.mean_int_rate - drift

    def moments(self, model: MarketModel, weights) -> tuple[float, float, float, float, float]:
        """(mu_X, var_X, mu_Y, var_Y, cov_XY) for X = I(t_last), Y = sum_i w_i log S(t_i)."""
        w = np.asarray(weights, dtype=float)
        sig = model.sigma_s
        mu_x = float(self.mean_int_rate[-1])
        var_x = float(self.cov_ii[-1, -1])
        mu_y = float(w @ self.log_price_mean(model))
        var_y = float(w @ self.cov_ii @ w + 2 * sig * (w @ self.cov_iw @ w) + sig**2 * (w @ self.cov_ww @ w))
        cov_xy = float(self.cov_ii[-1] @ w + sig * (self.cov_iw[-1] @ w))
        return mu_x, var_x, mu_y, var_y, cov_xy


def lognormal_option_price(mu_x, var_x, mu_y, var_y, cov_xy, strike, kind="call") -> float:
    """E[exp(-X) (exp(Y) - K)^+] (call) or E[exp(-X) (K - exp(Y))^+] (put) for jointly Gaussian (X, Y)."""
    if var_y <= 0:
        raise ValueError("Y must have positive variance")
    sd_y = np.sqrt(var_y)
    c = np.log(strike)
    bond = np.exp(-mu_x + 0.5 * var_x)  # E[exp(-X)]
    fwd = np.exp(mu_y - mu_x + 0.5 * (var_y + var_x - 2 * cov_xy))  # E[exp(Y - X)]
    d_asset = (mu_y + var_y - cov_xy - c) / sd_y
    d_bond = (mu_y - cov_xy - c) / sd_y
    if kind == "call":
        return float(fwd * norm.cdf(d_asset) - strike * bond * norm.cdf(d_bond))
    if kind == "put":
        return float(strike * bond * norm.cdf(-d_bond) - fwd * norm.cdf(-d_asset))
    raise ValueError("kind must be 'call' or 'put'")


def monitoring_times(maturity: float, n_steps: int) -> np.ndarray:
    return maturity * np.arange(1, n_steps + 1) / n_steps


def geometric_asian_price(model: MarketModel, strike: float, maturity: float, n_steps: int, kind="call") -> float:
    """Exact price of the discretely monitored geometric-average Asian option (payoff at T)."""
    gs = GaussianStructure.build(model, monitoring_times(maturity, n_steps))
    w = np.full(n_steps, 1.0 / n_steps)
    return lognormal_option_price(*gs.moments(model, w), strike, kind)


def european_price(model: MarketModel, strike: float, maturity: float, kind="call") -> float:
    """Exact European option price with Vasicek rates and rate/asset correlation."""
    gs = GaussianStructure.build(model, [maturity])
    return lognormal_option_price(*gs.moments(model, [1.0]), strike, kind)


def discounted_average_forward(model: MarketModel, maturity: float, n_steps: int) -> float:
    """E[exp(-integral_0^T r) * A_T] for the arithmetic average A_T; gives Asian put-call parity

    C_arith - P_arith = discounted_average_forward - K * P(0, T).
    """
    gs = GaussianStructure.build(model, monitoring_times(maturity, n_steps))
    sig = model.sigma_s
    mu_y = gs.log_price_mean(model)
    var_y = np.diag(gs.cov_ii) + 2 * sig * np.diag(gs.cov_iw) + sig**2 * gs.times
    cov_xy = gs.cov_ii[-1] + sig * gs.cov_iw[-1]
    mu_x, var_x = gs.mean_int_rate[-1], gs.cov_ii[-1, -1]
    return float(np.mean(np.exp(mu_y - mu_x + 0.5 * (var_y + var_x - 2 * cov_xy))))
