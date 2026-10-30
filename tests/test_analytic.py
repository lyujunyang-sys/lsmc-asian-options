import numpy as np
import pytest
from scipy.stats import norm

from lsmc_asian.analytic import (
    GaussianStructure,
    black_scholes,
    discounted_average_forward,
    european_price,
    geometric_asian_price,
    monitoring_times,
    zero_coupon_bond,
)
from lsmc_asian.model import MarketModel
from lsmc_asian.montecarlo import price_european
from lsmc_asian.products import AsianOption, VanillaOption

STOCH = MarketModel(s0=100.0, sigma_s=0.25, r0=0.03, kappa=0.4, theta=0.05, sigma_r=0.04, rho=-0.5)


def constant_rate(r=0.05, sigma=0.2, q=0.0):
    return MarketModel(s0=100.0, sigma_s=sigma, r0=r, kappa=1.0, theta=r, sigma_r=0.0, dividend_yield=q)


def kemna_vorst_discrete(s0, k, T, r, sigma, n, q=0.0):
    """Geometric average over t_i = iT/n, i = 1..n, with a constant rate: ln G is normal."""
    mu = np.log(s0) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (mu - np.log(k) + var) / np.sqrt(var)
    d2 = d1 - np.sqrt(var)
    return np.exp(-r * T) * (np.exp(mu + 0.5 * var) * norm.cdf(d1) - k * norm.cdf(d2))


def test_bond_formula_equals_gaussian_moment():
    for T in (0.25, 1.0, 5.0):
        gs = GaussianStructure.build(STOCH, [T])
        via_moments = np.exp(-gs.mean_int_rate[-1] + 0.5 * gs.cov_ii[-1, -1])
        assert zero_coupon_bond(STOCH, T) == pytest.approx(via_moments, rel=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_european_reduces_to_black_scholes(kind, strike):
    m = constant_rate(q=0.01)
    expected = black_scholes(100.0, strike, 1.5, 0.05, 0.2, 0.01, kind)
    assert european_price(m, strike, 1.5, kind) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("n", [1, 12, 252])
def test_geometric_asian_reduces_to_kemna_vorst(n):
    m = constant_rate(q=0.02)
    expected = kemna_vorst_discrete(100, 95, 1.0, 0.05, 0.2, n, 0.02)
    assert geometric_asian_price(m, 95.0, 1.0, n) == pytest.approx(expected, rel=1e-10)


def test_european_put_call_parity_with_stochastic_rates():
    T, K = 2.0, 105.0
    lhs = european_price(STOCH, K, T, "call") - european_price(STOCH, K, T, "put")
    assert lhs == pytest.approx(STOCH.s0 - K * zero_coupon_bond(STOCH, T), rel=1e-10)


def test_geometric_asian_put_call_parity():
    T, N, K = 1.0, 24, 100.0
    gs = GaussianStructure.build(STOCH, monitoring_times(T, N))
    mx, vx, my, vy, cxy = gs.moments(STOCH, np.full(N, 1 / N))
    fwd = np.exp(my - mx + 0.5 * (vy + vx - 2 * cxy))
    diff = geometric_asian_price(STOCH, K, T, N, "call") - geometric_asian_price(STOCH, K, T, N, "put")
    assert diff == pytest.approx(fwd - K * zero_coupon_bond(STOCH, T), rel=1e-10)


def test_covariances_match_simulation():
    T, N = 1.0, 6
    gs = GaussianStructure.build(STOCH, monitoring_times(T, N))
    p = STOCH.simulate(T, N, 400_000, np.random.default_rng(10))
    integ = -np.log(p.discount[:, 1:])
    # recover W^S(t) from the log price: ln S = ln S0 + I - (q + s^2/2) t + s W
    w = (np.log(p.spot[:, 1:] / STOCH.s0) - integ + 0.5 * STOCH.sigma_s**2 * gs.times) / STOCH.sigma_s
    np.testing.assert_allclose(integ.mean(axis=0), gs.mean_int_rate, atol=2e-4)
    np.testing.assert_allclose(np.cov(integ.T), gs.cov_ii, atol=5e-5, rtol=0.03)
    cross = np.cov(np.hstack([integ, w]).T)[:N, N:]
    np.testing.assert_allclose(cross, gs.cov_iw, atol=6e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_closed_forms_match_monte_carlo(kind):
    T, N, K, M = 1.0, 12, 100.0, 400_000
    geo = price_european(STOCH, AsianOption(K, kind, "geometric"), T, N, M, np.random.default_rng(11), antithetic=True)
    assert abs(geo.price - geometric_asian_price(STOCH, K, T, N, kind)) < 4 * geo.std_error
    eur = price_european(STOCH, VanillaOption(K, kind), T, N, M, np.random.default_rng(12), antithetic=True)
    assert abs(eur.price - european_price(STOCH, K, T, kind)) < 4 * eur.std_error


def test_arithmetic_asian_parity_matches_forward():
    # pathwise (A - K)^+ - (K - A)^+ = A - K, so C - P = E[D_T A] - K P(0, T)
    T, N, K = 1.0, 12, 100.0
    p = STOCH.simulate(T, N, 400_000, np.random.default_rng(13))
    d = p.discount[:, -1]
    diff = d * (AsianOption(K, "call").payoff(p) - AsianOption(K, "put").payoff(p))
    target = discounted_average_forward(STOCH, T, N) - K * zero_coupon_bond(STOCH, T)
    assert abs(diff.mean() - target) < 4 * diff.std() / np.sqrt(diff.size)


def test_geometric_call_cheaper_than_arithmetic_call():
    T, N, K = 1.0, 12, 100.0
    arith = price_european(STOCH, AsianOption(K, "call"), T, N, 100_000, np.random.default_rng(14),
                           antithetic=True, control_variate=True)
    assert geometric_asian_price(STOCH, K, T, N) < arith.price
