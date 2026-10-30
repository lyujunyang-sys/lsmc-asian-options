"""Monte Carlo pricing of European-style payoffs, with antithetic variates and a control variate.

Every price is reported with its standard error. With antithetic sampling the
two members of a pair are averaged first, so the standard error is computed
from independent pair means. The control variate for an arithmetic-average
Asian option is the geometric-average option with the same strike and
monitoring dates, whose exact price is known (``analytic.geometric_asian_price``);
both are discounted with the same pathwise discount factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analytic import geometric_asian_price
from .model import MarketModel
from .products import AsianOption


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo estimate with its standard error."""

    price: float
    std_error: float
    n_samples: int
    extra: dict = field(default_factory=dict)

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        return self.price - z * self.std_error, self.price + z * self.std_error


def pair_means(samples: np.ndarray, antithetic: bool) -> np.ndarray:
    """Collapse antithetic pairs (i, i + M/2) into their means; otherwise return the samples."""
    if not antithetic:
        return samples
    half = samples.shape[0] // 2
    return 0.5 * (samples[:half] + samples[half:])


def mean_and_error(samples: np.ndarray) -> tuple[float, float]:
    n = samples.shape[0]
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n))


def control_variate_estimate(y: np.ndarray, x: np.ndarray, x_mean: float) -> Estimate:
    """Estimate E[y] using x (with known mean ``x_mean``) as a control; beta = Cov(y, x) / Var(x).

    beta is estimated from the same sample, the usual choice; the resulting bias
    is O(1/n) and negligible next to the standard error at these sample sizes.
    """
    cov = np.cov(y, x, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    adjusted = y - beta * (x - x_mean)
    price, se = mean_and_error(adjusted)
    plain_var = cov[0, 0]
    vrf = plain_var / adjusted.var(ddof=1) if adjusted.var(ddof=1) > 0 else np.inf
    return Estimate(price, se, y.shape[0], {"beta": float(beta), "variance_reduction": float(vrf)})


def discounted_samples(model, product, maturity, n_steps, n_paths, rng, antithetic=False, batch_size=20_000):
    """Discounted payoffs D_T * payoff for ``product`` and, for Asian options, the geometric control.

    Paths are generated in batches so that large ``n_paths`` fit in memory. The
    random draws are consumed batch by batch, so the exact numbers depend on
    ``batch_size``; keep it fixed (it is part of each experiment config) when
    reproducing a run.
    """
    if antithetic and (n_paths % 2 or batch_size % 2):
        raise ValueError("antithetic sampling needs even n_paths and batch_size")
    ys, xs = [], []
    is_asian = isinstance(product, AsianOption)
    control = AsianOption(product.strike, product.kind, "geometric") if is_asian else None
    remaining = n_paths
    while remaining > 0:
        m = min(batch_size, remaining)
        paths = model.simulate(maturity, n_steps, m, rng, antithetic=antithetic)
        d_t = paths.discount[:, -1]
        ys.append(pair_means(d_t * product.payoff(paths), antithetic))
        if is_asian:
            xs.append(pair_means(d_t * control.payoff(paths), antithetic))
        remaining -= m
    y = np.concatenate(ys)
    x = np.concatenate(xs) if is_asian else None
    return y, x


def price_european(
    model: MarketModel,
    product,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = False,
    control_variate: bool = False,
    batch_size: int = 20_000,
) -> Estimate:
    """Price a payoff received at maturity (no early exercise) by Monte Carlo.

    ``control_variate`` is available for arithmetic-average Asian options.
    """
    y, x = discounted_samples(model, product, maturity, n_steps, n_paths, rng, antithetic, batch_size)
    if control_variate:
        if not (isinstance(product, AsianOption) and product.average == "arithmetic"):
            raise ValueError("the control variate applies to arithmetic-average Asian options")
        exact = geometric_asian_price(model, product.strike, maturity, n_steps, product.kind)
        est = control_variate_estimate(y, x, exact)
        return Estimate(est.price, est.std_error, n_paths, {**est.extra, "antithetic": antithetic})
    price, se = mean_and_error(y)
    return Estimate(price, se, n_paths, {"antithetic": antithetic})
