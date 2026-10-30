import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lsmc_asian.calibration import NotMeanReverting, annualised_volatility, calibrate_market, fit_vasicek
from lsmc_asian.experiments import ROOT, load_config, resolve_market, rng_for, run
from lsmc_asian.model import MarketModel


def simulate_vasicek(kappa, theta, sigma, r0, n, dt, rng):
    b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - b * b) / (2 * kappa))
    r = np.empty(n)
    r[0] = r0
    for t in range(1, n):
        r[t] = theta + (r[t - 1] - theta) * b + sd * rng.standard_normal()
    return r


def test_vasicek_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    r = simulate_vasicek(kappa=2.0, theta=0.04, sigma=0.01, r0=0.02, n=252 * 60, dt=1 / 252, rng=rng)
    fit = fit_vasicek(r, dt=1 / 252, n_bootstrap=100, rng=np.random.default_rng(1))
    assert abs(fit.kappa - 2.0) < 4 * fit.kappa_se
    assert abs(fit.theta - 0.04) < 4 * fit.theta_se
    assert fit.sigma == pytest.approx(0.01, rel=0.02)


def test_series_without_mean_reversion_is_rejected():
    # rates drifting away from any level (AR(1) slope > 1), like the 3-month yield in 2015-2023
    rng = np.random.default_rng(2)
    r = np.empty(2000)
    r[0] = 0.001
    for t in range(1, r.size):
        r[t] = 1.002 * r[t - 1] + 1e-5 * rng.standard_normal()
    with pytest.raises(NotMeanReverting):
        fit_vasicek(r)


def test_volatility_estimator():
    rng = np.random.default_rng(3)
    logp = np.cumsum(0.25 / np.sqrt(252) * rng.standard_normal(252 * 40))
    prices = pd.Series(100 * np.exp(logp))
    assert annualised_volatility(prices) == pytest.approx(0.25, rel=0.02)


def test_calibrate_market_on_synthetic_series():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2010-01-01", periods=252 * 30)
    rates = pd.Series(simulate_vasicek(1.5, 0.03, 0.01, 0.03, dates.size, 1 / 252, rng), index=dates)
    prices = pd.Series(100 * np.exp(np.cumsum(0.2 / np.sqrt(252) * rng.standard_normal(dates.size))), index=dates)
    out = calibrate_market(prices, rates, n_bootstrap=20)
    MarketModel(**out["market"])  # must be a valid model
    assert out["valuation_date"] == dates[-1].strftime("%Y-%m-%d")
    assert out["market"]["sigma_s"] == pytest.approx(0.2, rel=0.15)
    assert abs(out["market"]["rho"]) < 0.05


def test_committed_calibration_file_is_valid():
    path = ROOT / "data" / "calibrated" / "sp500_2015-2023_DGS10.json"
    calib = json.loads(path.read_text(encoding="utf-8"))
    MarketModel(**calib["market"])
    assert calib["valuation_date"] == "2023-12-29"


def test_rng_for_is_deterministic_and_label_specific():
    assert rng_for(1, "a").random() == rng_for(1, "a").random()
    assert rng_for(1, "a").random() != rng_for(1, "b").random()


def test_every_config_resolves():
    for path in sorted((ROOT / "configs").glob("*.json")):
        model, strike, _ = resolve_market(load_config(path))
        assert strike > 0 and model.s0 > 0


def test_small_experiment_runs_end_to_end(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "appendix_baseline.json")
    cfg["name"] = "smoke"
    cfg["contract"]["kinds"] = ["call"]
    cfg["simulation"].update(n_paths=2000, n_paths_out_of_sample=2000, batch_size=2000)
    table = run(cfg, out_root=tmp_path)["pricing"]
    assert (tmp_path / "smoke" / "pricing.csv").exists()
    assert (tmp_path / "smoke" / "summary.json").exists()
    exact = table.loc[table["method"] == "closed form", "price"]
    assert len(exact) == 2 and (exact > 0).all()
