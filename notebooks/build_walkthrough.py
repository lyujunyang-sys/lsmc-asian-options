"""Build notebooks/walkthrough.ipynb from the cells below (then execute it with nbconvert).

    python notebooks/build_walkthrough.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# LSMC pricing of Asian options under Vasicek rates: walkthrough

This notebook goes through the method of the paper step by step with the `lsmc_asian` package:

1. the model and its exact simulation,
2. checks against closed-form prices,
3. the European-exercise arithmetic Asian option and variance reduction,
4. the Bermudan-exercise Asian option priced by least-squares Monte Carlo,
5. the S&P 500 empirical application.

Every random number comes from a seeded generator, so re-running the notebook gives the same output."""),
    code("""import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsmc_asian import (AsianOption, MarketModel, VanillaOption, black_scholes, european_price,
                        geometric_asian_price, price_european, price_lsmc, zero_coupon_bond)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SEED = 20240914
pd.set_option("display.float_format", "{:.4f}".format)"""),
    md("""## 1. Model

Risk-neutral dynamics:

$$dr_t = \\kappa(\\theta - r_t)\\,dt + \\sigma_r\\,dW^r_t, \\qquad \\frac{dS_t}{S_t} = (r_t - q)\\,dt + \\sigma_S\\,dW^S_t, \\qquad d\\langle W^r, W^S\\rangle_t = \\rho\\,dt.$$

Over each step the new short rate, the integral of the rate over the step and the Brownian increment of the asset are jointly Gaussian, so the simulation is exact (no Euler error) and the realised discount factor $D_t = e^{-\\int_0^t r_u du}$ comes out of the same draw. Base case: the generic parameters of the paper's appendix."""),
    code("""m = MarketModel(s0=100.0, sigma_s=0.20, r0=0.03, kappa=0.10, theta=0.05, sigma_r=0.02)
T, N, K = 1.0, 40, 100.0
paths = m.simulate(T, N, 10_000, np.random.default_rng(SEED))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(paths.times, paths.spot[:10].T, lw=1)
axes[0].set(title="asset paths", xlabel="years")
axes[1].plot(paths.times, 100 * paths.short_rate[:10].T, lw=1)
axes[1].set(title="short rate (%)", xlabel="years")
plt.tight_layout()"""),
    md("""## 2. Checks against closed forms

The log asset price and the integrated rate are jointly Gaussian, which gives exact prices for the zero-coupon bond, the European option and the discretely monitored geometric-average Asian option, all with stochastic rates. The simulation must reproduce them within Monte Carlo error."""),
    code("""rng = np.random.default_rng(SEED + 1)
d_T = paths.discount[:, -1]
geo_mc = price_european(m, AsianOption(K, "call", "geometric"), T, N, 200_000, rng, antithetic=True)
eur_mc = price_european(m, VanillaOption(K, "call"), T, N, 200_000, rng, antithetic=True)
checks = pd.DataFrame([
    ["zero-coupon bond P(0,T)", zero_coupon_bond(m, T), d_T.mean(), d_T.std() / np.sqrt(d_T.size)],
    ["geometric Asian call", geometric_asian_price(m, K, T, N), geo_mc.price, geo_mc.std_error],
    ["European call", european_price(m, K, T), eur_mc.price, eur_mc.std_error],
], columns=["quantity", "closed form", "Monte Carlo", "std error"])
checks["z"] = (checks["Monte Carlo"] - checks["closed form"]) / checks["std error"]
checks"""),
    md("""With $\\sigma_r = 0$ the rate is deterministic and the closed forms reduce to Black-Scholes and to the discrete Kemna-Vorst formula (both covered by the test suite)."""),
    code("""flat = MarketModel(s0=100, sigma_s=0.2, r0=0.05, kappa=1.0, theta=0.05, sigma_r=0.0)
european_price(flat, 100, 1.0), black_scholes(100, 100, 1.0, 0.05, 0.2)"""),
    md("""## 3. Arithmetic Asian option, European exercise

No closed form exists, so it is priced by Monte Carlo. Two variance-reduction techniques:

* **antithetic variates**: each normal draw is also used with its sign flipped;
* **control variate**: the geometric-average option on the same paths, whose exact price is known. The two payoffs are almost perfectly correlated.

The variance reduction factor is (plain standard error / reduced standard error)$^2$, i.e. how many times more plain paths would be needed for the same accuracy."""),
    code("""arith = AsianOption(K, "call", "arithmetic")
rows = []
for label, anti, cv in [("plain", False, False), ("antithetic", True, False),
                        ("control variate", False, True), ("antithetic + control variate", True, True)]:
    est = price_european(m, arith, T, N, 10_000, np.random.default_rng(SEED + 2), antithetic=anti, control_variate=cv)
    rows.append([label, est.price, est.std_error])
vr = pd.DataFrame(rows, columns=["method", "price", "std error"])
vr["variance reduction"] = (vr["std error"].iloc[0] / vr["std error"]) ** 2
vr"""),
    md("""## 4. Bermudan-exercise Asian option: least-squares Monte Carlo

If the holder may exercise on any monitoring date $t_j$ and receive $(A_j - K)^+$, with $A_j$ the average to date, the price is an optimal-stopping problem. LSMC (Longstaff and Schwartz, 2001) works backwards: on each date it regresses the realised, pathwise-discounted future cash flow of the in-the-money paths on polynomials in the state $(S_j, A_j, r_j)$, and exercises where the immediate payoff beats the fitted continuation value.

The in-sample estimate reuses the fitting paths, so its bias can go either way. The out-of-sample estimate applies the frozen rule to independent paths; a feasible rule cannot beat the optimal one, so it is a lower bound on the price up to Monte Carlo noise, and it is the number to report.

First, the engine on the standard benchmark of the original paper: an American put, $S_0=36$, $K=40$, $r=6\\%$, $\\sigma=20\\%$, $T=1$, 50 exercise dates. The finite-difference value is 4.478."""),
    code("""bench = MarketModel(s0=36, sigma_s=0.2, r0=0.06, kappa=1.0, theta=0.06, sigma_r=0.0)
res = price_lsmc(bench, VanillaOption(40, "put"), 1.0, 50, 50_000, np.random.default_rng(SEED + 3),
                 degree=3, antithetic=True, n_paths_out_of_sample=50_000)
print(f"in-sample {res.in_sample.price:.4f} ± {res.in_sample.std_error:.4f}, "
      f"out-of-sample {res.out_of_sample.price:.4f} ± {res.out_of_sample.std_error:.4f}  (reference 4.478)")"""),
    md("""Now the Bermudan arithmetic Asian call under Vasicek rates, with and without the geometric control variate:"""),
    code("""rows = []
for cv in (False, True):
    res = price_lsmc(m, arith, T, N, 10_000, np.random.default_rng(SEED + 4), antithetic=True,
                     n_paths_out_of_sample=10_000, control_variate=cv)
    for which, est in [("in-sample", res.in_sample), ("out-of-sample", res.out_of_sample), ("European exercise", res.european)]:
        rows.append(["with CV" if cv else "no CV", which, est.price, est.std_error])
pd.DataFrame(rows, columns=["control variate", "estimate", "price", "std error"])"""),
    md("""## 5. S&P 500 application

`scripts/download_data.py` fetches S&P 500 closes (Yahoo Finance) and Treasury yields (FRED) for 2015 to 2023; `scripts/calibrate.py` estimates the inputs. The calibrated file is committed, so this section runs without downloading anything."""),
    code("""calib = json.loads((ROOT / "data/calibrated/sp500_2015-2023_DGS10.json").read_text())
vas = calib["diagnostics"]["vasicek"]
print("valuation date", calib["valuation_date"])
pd.Series(calib["market"]).to_frame("value").assign(
    std_error=pd.Series({"kappa": vas["kappa_se"], "theta": vas["theta_se"], "sigma_r": vas["sigma_se"]}))"""),
    md("""The mean-reversion speed is poorly determined on nine years of data (its bootstrap standard error is about twice the estimate). The 3-month yield cannot be fitted at all on this window: it trends from 0% to 5.4% and the AR(1) slope exceeds one. Like the paper, the calibration therefore uses the 10-year yield as the rate input.

If the raw data has been downloaded, plot it:"""),
    code("""raw = ROOT / "data/raw"
if (raw / "sp500_close.csv").exists():
    from lsmc_asian.calibration import rolling_volatility
    from lsmc_asian.data import load_rate, load_sp500
    px, r10 = load_sp500(raw / "sp500_close.csv"), load_rate(raw / "fred_DGS10.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    px.plot(ax=axes[0], title="S&P 500")
    (100 * rolling_volatility(px)).plot(ax=axes[1], title="rolling 252-day volatility (%)")
    (100 * r10).plot(ax=axes[2], title="10-year Treasury yield (%)")
    plt.tight_layout()
else:
    print("run scripts/download_data.py to plot the raw series")"""),
    md("""Prices at the calibrated parameters and at the parameter values of the paper's Table 6-1, as produced by `scripts/run_experiment.py` (committed in `results/`):"""),
    code("""cols = ["product", "method", "price", "std_error", "ci95_low", "ci95_high"]
for name in ["paper_table_6_1", "sp500_calibrated"]:
    print(name)
    display(pd.read_csv(ROOT / "results" / name / "pricing.csv")[cols])"""),
    md("""## Reproducing everything

```
python scripts/download_data.py
python scripts/calibrate.py
python scripts/run_experiment.py configs/*.json
python scripts/make_figures.py
pytest
```"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
out = Path(__file__).with_name("walkthrough.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
