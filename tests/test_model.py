import numpy as np
import pytest

from lsmc_asian.analytic import zero_coupon_bond
from lsmc_asian.model import MarketModel, psd_factor

BASE = dict(s0=100.0, sigma_s=0.2, r0=0.03, kappa=0.5, theta=0.05, sigma_r=0.03, rho=-0.4)


def test_step_covariance_is_psd_and_factor_reproduces_it():
    m = MarketModel(**BASE)
    for dt in (1 / 252, 0.1, 1.0):
        cov = m.step_covariance(dt)
        assert np.all(np.linalg.eigvalsh(cov) > -1e-15)
        f = psd_factor(cov)
        np.testing.assert_allclose(f @ f.T, cov, atol=1e-15)


def test_integrated_variance_is_continuous_across_series_switch():
    # the small kappa*dt Taylor series and the closed form must agree where they meet (x = 1e-3)
    lo = MarketModel(**{**BASE, "kappa": 0.252 * (1 - 1e-9)}).step_covariance(1 / 252)[1, 1]
    hi = MarketModel(**{**BASE, "kappa": 0.252 * (1 + 1e-9)}).step_covariance(1 / 252)[1, 1]
    assert abs(lo - hi) / hi < 1e-6


def test_short_rate_moments_match_vasicek():
    m = MarketModel(**BASE)
    T, n = 2.0, 20
    p = m.simulate(T, n, 200_000, np.random.default_rng(1))
    r_t = p.short_rate[:, -1]
    mean = m.theta + (m.r0 - m.theta) * np.exp(-m.kappa * T)
    var = m.sigma_r**2 / (2 * m.kappa) * (1 - np.exp(-2 * m.kappa * T))
    assert abs(r_t.mean() - mean) < 4 * np.sqrt(var / r_t.size)
    assert abs(r_t.var() / var - 1) < 0.02


def test_discount_factor_matches_bond_price():
    m = MarketModel(**BASE)
    p = m.simulate(1.0, 12, 200_000, np.random.default_rng(2))
    d = p.discount[:, -1]
    assert abs(d.mean() - zero_coupon_bond(m, 1.0)) < 4 * d.std() / np.sqrt(d.size)


def test_discounted_asset_is_a_martingale():
    m = MarketModel(**BASE)
    p = m.simulate(1.0, 12, 200_000, np.random.default_rng(3))
    x = p.discount * p.spot
    for j in (1, 6, 12):
        assert abs(x[:, j].mean() - m.s0) < 4 * x[:, j].std() / np.sqrt(x.shape[0])


def test_zero_rate_vol_gives_deterministic_rate():
    m = MarketModel(**{**BASE, "sigma_r": 0.0, "theta": 0.03})
    p = m.simulate(1.0, 10, 1000, np.random.default_rng(4))
    np.testing.assert_allclose(p.short_rate, 0.03)
    np.testing.assert_allclose(p.discount[:, -1], np.exp(-0.03))


def test_antithetic_pairs_are_mirrored():
    m = MarketModel(**BASE)
    p = m.simulate(1.0, 5, 10, np.random.default_rng(5), antithetic=True)
    mean_path = m.theta + (m.r0 - m.theta) * np.exp(-m.kappa * p.times)
    dev = p.short_rate - mean_path
    np.testing.assert_allclose(dev[:5], -dev[5:], atol=1e-15)


def test_same_seed_same_paths():
    m = MarketModel(**BASE)
    a = m.simulate(1.0, 8, 100, np.random.default_rng(7))
    b = m.simulate(1.0, 8, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(a.spot, b.spot)


@pytest.mark.parametrize("bad", [dict(s0=-1), dict(kappa=0.0), dict(sigma_s=-0.1), dict(rho=1.5)])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):
        MarketModel(**{**BASE, **bad})


def test_antithetic_needs_even_paths():
    with pytest.raises(ValueError):
        MarketModel(**BASE).simulate(1.0, 4, 11, np.random.default_rng(0), antithetic=True)
