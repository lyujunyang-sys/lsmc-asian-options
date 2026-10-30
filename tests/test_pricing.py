import numpy as np
import pytest

from lsmc_asian.analytic import black_scholes, geometric_asian_price
from lsmc_asian.lsmc import apply_policy, basis_matrix, fit_policy, monomial_exponents, price_lsmc
from lsmc_asian.model import MarketModel
from lsmc_asian.montecarlo import control_variate_estimate, price_european
from lsmc_asian.products import AsianOption, VanillaOption, running_average

BASE = MarketModel(s0=100.0, sigma_s=0.2, r0=0.03, kappa=0.1, theta=0.05, sigma_r=0.02)


def test_running_average():
    spot = np.array([[100.0, 1.0, 3.0, 8.0]])
    np.testing.assert_allclose(running_average(spot, "arithmetic")[0, 1:], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(running_average(spot, "geometric")[0, 1:], [1.0, np.sqrt(3.0), 24 ** (1 / 3)])


def test_monomial_basis():
    assert len(monomial_exponents(3, 2)) == 10  # 1, 3 linear, 6 quadratic
    x = np.array([[2.0, 3.0]])
    b = basis_matrix(x, monomial_exponents(2, 2))
    np.testing.assert_allclose(sorted(b[0]), sorted([1, 2, 3, 4, 6, 9]))


def test_control_variate_is_unbiased_and_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_000)
    y = 2.0 + 3.0 * x + 0.1 * rng.standard_normal(x.size)
    est = control_variate_estimate(y, x, 0.0)
    assert abs(est.price - 2.0) < 4 * est.std_error
    assert est.extra["beta"] == pytest.approx(3.0, abs=0.01)
    assert est.extra["variance_reduction"] > 500


def test_variance_reduction_on_asian_option():
    arith = AsianOption(100.0, "call")
    plain = price_european(BASE, arith, 1.0, 40, 20_000, np.random.default_rng(1))
    both = price_european(BASE, arith, 1.0, 40, 20_000, np.random.default_rng(2), antithetic=True, control_variate=True)
    assert (plain.std_error / both.std_error) ** 2 > 100
    assert abs(plain.price - both.price) < 4 * plain.std_error


def test_european_pricing_is_reproducible():
    arith = AsianOption(100.0, "call")
    a = price_european(BASE, arith, 1.0, 12, 4_000, np.random.default_rng(3), antithetic=True)
    b = price_european(BASE, arith, 1.0, 12, 4_000, np.random.default_rng(3), antithetic=True)
    assert a.price == b.price and a.std_error == b.std_error


def test_lsmc_american_put_longstaff_schwartz_benchmark():
    # Longstaff & Schwartz (2001), Table 1: S0 = 36, K = 40, r = 6%, sigma = 20%, T = 1,
    # 50 exercise dates per year; finite-difference American value 4.478.
    m = MarketModel(s0=36.0, sigma_s=0.2, r0=0.06, kappa=1.0, theta=0.06, sigma_r=0.0)
    res = price_lsmc(m, VanillaOption(40.0, "put"), 1.0, 50, 50_000, np.random.default_rng(4),
                     degree=3, antithetic=True, n_paths_out_of_sample=50_000)
    assert abs(res.in_sample.price - 4.478) < 0.03
    assert abs(res.out_of_sample.price - 4.478) < 0.03
    assert abs(res.european.price - black_scholes(36, 40, 1.0, 0.06, 0.2, kind="put")) < 4 * res.european.std_error


def test_lsmc_call_without_dividends_is_never_exercised_early():
    # with q = 0 and positive rates, early exercise of a vanilla call is never optimal
    m = MarketModel(s0=100.0, sigma_s=0.2, r0=0.05, kappa=1.0, theta=0.05, sigma_r=0.0)
    res = price_lsmc(m, VanillaOption(100.0, "call"), 1.0, 20, 20_000, np.random.default_rng(5),
                     antithetic=True, n_paths_out_of_sample=20_000)
    bs = black_scholes(100, 100, 1.0, 0.05, 0.2)
    assert abs(res.out_of_sample.price - bs) < 4 * res.out_of_sample.std_error + 0.05


def test_bermudan_asian_worth_at_least_european():
    res = price_lsmc(BASE, AsianOption(100.0, "call"), 1.0, 40, 20_000, np.random.default_rng(6),
                     antithetic=True, n_paths_out_of_sample=20_000)
    assert res.out_of_sample.price > res.european.price - 3 * res.out_of_sample.std_error
    # out-of-sample (feasible policy) should not exceed in-sample by more than noise
    assert res.out_of_sample.price < res.in_sample.price + 4 * res.in_sample.std_error


def test_policy_without_exercise_reproduces_european():
    paths = BASE.simulate(1.0, 12, 5_000, np.random.default_rng(7))
    product = AsianOption(100.0, "call")
    policy, _ = fit_policy(paths, product)
    policy.rules = {j: None for j in policy.rules}  # never exercise early
    cash = apply_policy(policy, paths, product)
    np.testing.assert_allclose(cash, paths.discount[:, -1] * product.payoff(paths))


def test_lsmc_control_variate_tightens_error():
    product = AsianOption(100.0, "call")
    a = price_lsmc(BASE, product, 1.0, 40, 10_000, np.random.default_rng(8), antithetic=True, n_paths_out_of_sample=10_000)
    b = price_lsmc(BASE, product, 1.0, 40, 10_000, np.random.default_rng(8), antithetic=True,
                   n_paths_out_of_sample=10_000, control_variate=True)
    assert b.out_of_sample.std_error < a.out_of_sample.std_error / 2


def test_geometric_closed_form_used_as_control_matches_product():
    # the control in price_european is the geometric option with the same strike and dates
    T, N, K = 1.0, 40, 100.0
    geo = price_european(BASE, AsianOption(K, "call", "geometric"), T, N, 100_000, np.random.default_rng(9), antithetic=True)
    assert abs(geo.price - geometric_asian_price(BASE, K, T, N)) < 4 * geo.std_error
