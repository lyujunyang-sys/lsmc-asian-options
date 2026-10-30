"""Parameter estimation from market data.

* Vasicek: the exact discretisation of the SDE on a step dt is the AR(1)

      r_{t+dt} = a + b r_t + e_t,   e_t ~ N(0, s^2),
      b = exp(-kappa dt),  a = theta (1 - b),  s^2 = sigma^2 (1 - b^2) / (2 kappa),

  so OLS of r_{t+dt} on r_t (which is also the conditional MLE) gives
  kappa = -ln(b)/dt, theta = a/(1-b), sigma = s * sqrt(2 kappa / (1 - b^2)).
  Mean reversion is identified only if 0 < b < 1; otherwise ``NotMeanReverting``
  is raised instead of returning a meaningless kappa. Standard errors come from
  a parametric bootstrap. kappa is known to be estimated with a large upward
  bias and wide uncertainty on samples of a few years, which the bootstrap shows.

* Asset volatility: annualised standard deviation of daily log returns, over
  the full sample and over a rolling window (252 trading days by default).

* Correlation: between daily log returns of the asset and daily changes of the rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


class NotMeanReverting(ValueError):
    """The AR(1) slope is outside (0, 1), so the series shows no mean reversion to fit."""


@dataclass(frozen=True)
class VasicekFit:
    kappa: float
    theta: float
    sigma: float
    ar_intercept: float
    ar_slope: float
    residual_sd: float
    n_obs: int
    dt: float
    kappa_se: float | None = None
    theta_se: float | None = None
    sigma_se: float | None = None
    bootstrap_reps: int = 0
    bootstrap_failures: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _ar1_fit(r: np.ndarray, dt: float) -> tuple[float, float, float, float, float, float]:
    x, y = r[:-1], r[1:]
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    sd = float(np.sqrt(resid @ resid / (len(x) - 2)))
    if not 0.0 < slope < 1.0:
        raise NotMeanReverting(
            f"AR(1) slope b = {slope:.6f} is not in (0, 1): no mean reversion in this sample"
        )
    kappa = -np.log(slope) / dt
    theta = intercept / (1.0 - slope)
    sigma = sd * np.sqrt(2.0 * kappa / (1.0 - slope**2))
    return float(kappa), float(theta), float(sigma), float(intercept), float(slope), sd


def fit_vasicek(rates, dt: float = 1.0 / 252, n_bootstrap: int = 0, rng: np.random.Generator | None = None) -> VasicekFit:
    """Fit Vasicek to an equally spaced series of short rates (decimals, e.g. 0.03 for 3%)."""
    r = np.asarray(rates, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError("need at least 30 observations")
    kappa, theta, sigma, a, b, sd = _ar1_fit(r, dt)
    fit = dict(kappa=kappa, theta=theta, sigma=sigma, ar_intercept=a, ar_slope=b, residual_sd=sd, n_obs=int(r.size), dt=dt)
    if n_bootstrap <= 0:
        return VasicekFit(**fit)

    rng = rng if rng is not None else np.random.default_rng(0)
    draws, failures = [], 0
    for _ in range(n_bootstrap):
        sim = np.empty_like(r)
        sim[0] = r[0]
        shocks = rng.standard_normal(r.size - 1) * sd
        for t in range(1, r.size):
            sim[t] = a + b * sim[t - 1] + shocks[t - 1]
        try:
            draws.append(_ar1_fit(sim, dt)[:3])
        except NotMeanReverting:
            failures += 1
    draws = np.array(draws)
    se = draws.std(axis=0, ddof=1) if len(draws) > 1 else [np.nan] * 3
    return VasicekFit(
        **fit,
        kappa_se=float(se[0]),
        theta_se=float(se[1]),
        sigma_se=float(se[2]),
        bootstrap_reps=n_bootstrap,
        bootstrap_failures=failures,
    )


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def annualised_volatility(prices: pd.Series, periods_per_year: int = 252) -> float:
    return float(log_returns(prices).std(ddof=1) * np.sqrt(periods_per_year))


def rolling_volatility(prices: pd.Series, window: int = 252, periods_per_year: int = 252) -> pd.Series:
    return log_returns(prices).rolling(window).std(ddof=1).dropna() * np.sqrt(periods_per_year)


def return_rate_correlation(prices: pd.Series, rates: pd.Series) -> float:
    """Correlation of daily asset log returns with daily rate changes, on dates present in both series."""
    joined = pd.concat([log_returns(prices), rates.diff()], axis=1, join="inner").dropna()
    return float(joined.corr().iloc[0, 1])


def calibrate_market(
    prices: pd.Series,
    rates: pd.Series,
    valuation_date: str | None = None,
    vol_window: int = 252,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict:
    """Estimate all model inputs from price and rate series indexed by date.

    Everything uses data up to and including ``valuation_date`` (default: the
    last common date). Returns a JSON-serialisable dict whose ``market`` entry
    can be passed straight to ``MarketModel(**market)``.
    """
    end = pd.Timestamp(valuation_date) if valuation_date else min(prices.index.max(), rates.index.max())
    prices = prices.loc[:end].dropna()
    rates = rates.loc[:end].dropna()
    val_date = prices.index.max()

    vas = fit_vasicek(rates.to_numpy(), dt=1.0 / 252, n_bootstrap=n_bootstrap, rng=np.random.default_rng(seed))
    roll = rolling_volatility(prices, vol_window)
    market = {
        "s0": float(prices.iloc[-1]),
        "sigma_s": float(roll.iloc[-1]),
        "r0": float(rates.loc[:val_date].iloc[-1]),
        "kappa": vas.kappa,
        "theta": vas.theta,
        "sigma_r": vas.sigma,
        "rho": return_rate_correlation(prices, rates),
        "dividend_yield": 0.0,
    }
    return {
        "valuation_date": val_date.strftime("%Y-%m-%d"),
        "market": market,
        "diagnostics": {
            "price_obs": int(prices.size),
            "price_first_date": prices.index.min().strftime("%Y-%m-%d"),
            "rate_obs": int(rates.size),
            "rate_first_date": rates.index.min().strftime("%Y-%m-%d"),
            "sigma_s_full_sample": annualised_volatility(prices),
            "sigma_s_rolling_window": vol_window,
            "sigma_s_rolling_mean": float(roll.mean()),
            "sigma_s_rolling_min": float(roll.min()),
            "sigma_s_rolling_max": float(roll.max()),
            "vasicek": vas.to_dict(),
            "bootstrap_seed": seed,
        },
    }
