# Data dictionary

Every file the code reads or writes, with its columns, units and origin.

## Raw data (`data/raw/`, downloaded, not committed)

Created by `python scripts/download_data.py` (default window 2015-01-01 to 2023-12-31, the sample used in the paper). The S&P 500 index series is licensed by S&P Dow Jones Indices, so it is downloaded rather than redistributed. `data/manifest.json` records the source, retrieval time, row count, date range and SHA-256 of each file used for the committed results, so a fresh download can be compared with it.

### `sp500_close.csv`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `date` | date (YYYY-MM-DD) | | NYSE trading day |
| `close` | float | index points | S&P 500 (`^GSPC`) daily close from Yahoo Finance, rounded to 4 decimals. Price index: dividends are not reinvested. |

Window used: 2,264 trading days, 2015-01-02 to 2023-12-29.

### `fred_DGS10.csv`, `fred_DGS3MO.csv`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `date` | date (YYYY-MM-DD) | | US business day |
| `value_pct` | float | percent per year | Market yield on US Treasury securities at 10-year (`DGS10`) or 3-month (`DGS3MO`) constant maturity, from FRED (Board of Governors of the Federal Reserve System, H.15). Days FRED marks as missing (`.`) are dropped. |

Window used: 2,251 business days, 2015-01-02 to 2023-12-29. The loader divides by 100 (3.88 becomes 0.0388).

## Calibrated inputs (`data/calibrated/*.json`, committed)

Created by `python scripts/calibrate.py`. One file per rate series that admits a Vasicek fit; file name `sp500_<start year>-<end year>_<series>.json`.

| Field | Unit | How it is obtained |
| --- | --- | --- |
| `valuation_date` | date | last trading day with a price (2023-12-29) |
| `market.s0` | index points | S&P 500 close on the valuation date |
| `market.sigma_s` | decimal per year | standard deviation of daily log returns over the last 252 trading days, times sqrt(252) |
| `market.r0` | decimal per year | rate on the valuation date |
| `market.kappa` | 1 / year | Vasicek mean-reversion speed, from the exact AR(1) discretisation: kappa = -ln(b) / dt |
| `market.theta` | decimal per year | Vasicek long-run mean, theta = a / (1 - b) |
| `market.sigma_r` | decimal per sqrt(year) | Vasicek rate volatility, sigma = s * sqrt(2 kappa / (1 - b^2)) |
| `market.rho` | | correlation of daily S&P 500 log returns with daily rate changes, on common dates |
| `market.dividend_yield` | decimal per year | set to 0, as in the paper (see the README's limitations) |
| `diagnostics.sigma_s_full_sample` | decimal per year | volatility over the whole window |
| `diagnostics.sigma_s_rolling_mean/min/max` | decimal per year | summary of the rolling 252-day volatility |
| `diagnostics.vasicek.ar_intercept`, `ar_slope` | | OLS a, b of r_{t+1} = a + b r_t + e |
| `diagnostics.vasicek.residual_sd` | decimal | s, standard deviation of e |
| `diagnostics.vasicek.dt` | years | 1/252: one step per business-day observation |
| `diagnostics.vasicek.*_se` | as the parameter | parametric-bootstrap standard errors (`bootstrap_reps` samples, seed `bootstrap_seed`; `bootstrap_failures` samples had no mean reversion and were dropped) |
| `rate_series`, `data_window`, `data_manifest` | | provenance |

Rate proxy. The paper uses the 10-year Treasury yield as the interest-rate input, and so does `sp500_calibrated`. The 3-month yield, the more natural proxy for a short rate, rose from 0.02% to 5.4% over 2015 to 2023 with no reversion (AR(1) slope 1.0009 > 1), so Vasicek cannot be fitted to it on this window and `calibrate.py` reports that instead of writing a file. The mean-reversion speed is weakly identified even for the 10-year yield (kappa = 0.26, bootstrap standard error 0.50), a well-known property of short samples.

## Experiment configs (`configs/*.json`, the seed files)

| Field | Description |
| --- | --- |
| `name` | output folder under `results/` |
| `description` | what the experiment is |
| `seed` | master seed. Each table row draws from `numpy.random.SeedSequence([seed, crc32(row label)])`, so rows are independent and individually reproducible. |
| `market` | model parameters (`s0`, `sigma_s`, `r0`, `kappa`, `theta`, `sigma_r`, `rho`, `dividend_yield`), or instead |
| `market_from` | path of a calibration file, optionally with `market_overrides` |
| `contract.maturity` | years |
| `contract.n_steps` | number of monitoring (and exercise) dates, equally spaced; the average is over t_1..t_N, excluding S_0 |
| `contract.strike` or `contract.strike_moneyness` | absolute strike, or strike / s0 (rounded to cents) |
| `contract.kinds` | `call` and/or `put` |
| `simulation.n_paths` | paths per estimate (training paths for LSMC) |
| `simulation.n_paths_out_of_sample` | independent paths for the out-of-sample LSMC estimate |
| `simulation.antithetic` | antithetic pairs in LSMC |
| `simulation.batch_size` | paths simulated at a time in European-exercise Monte Carlo (affects the order of random draws, so keep it fixed when reproducing) |
| `simulation.lsmc_degree` | total degree of the polynomial basis in the standardised state (S_t, A_t, r_t) |
| `convergence`, `sensitivity` | grids for those studies |
| `tables` | which of `pricing`, `convergence`, `sensitivity` to produce |

## Results (`results/<name>/`, committed)

### `pricing.csv`

| Column | Description |
| --- | --- |
| `product` | contract; "European exercise" pays at T only, "Bermudan exercise" may be exercised on any monitoring date, receiving the average to date minus the strike |
| `method` | `closed form`; `MC` (plain); `MC + antithetic`; `MC + control variate` (geometric Asian with exact price as the control); `MC + antithetic + control variate`; `LSMC[, + control variate], in-sample / out-of-sample` |
| `price` | estimate, in the units of `s0` |
| `std_error` | Monte Carlo standard error (0 for closed forms) |
| `ci95_low`, `ci95_high` | price ± 1.96 standard errors |
| `n_samples` | simulated paths (for LSMC, per path set) |
| `runtime_s` | wall-clock seconds on the machine in `summary.json`; for LSMC rows it covers fitting and out-of-sample valuation together |
| `variance_reduction` | (plain-MC standard error / this standard error)^2 at the same number of paths |
| `z_vs_exact` | (MC price - closed form) / standard error; should look like a standard normal draw |
| `early_exercise_premium` | LSMC out-of-sample price minus the European-exercise price on the training paths |

### `convergence.csv`

| Column | Description |
| --- | --- |
| `study` | `paths` (vary the number of paths) or `degree` (vary the LSMC basis degree) |
| `method`, `n_paths`, `degree`, `price`, `std_error`, `runtime_s` | as above |

### `sensitivity.csv`

| Column | Description |
| --- | --- |
| `parameter`, `value` | parameter changed from the base case (`base` = no change) and its new value |
| `strike` | strike used |
| `asian_european`, `asian_european_se` | arithmetic Asian call, European exercise, MC + antithetic + control variate |
| `asian_bermudan_oos`, `asian_bermudan_oos_se` | arithmetic Asian call, Bermudan exercise, LSMC out-of-sample with control variate |
| `geometric_exact` | geometric Asian call, closed form |
| `european_vanilla_exact` | European call with Vasicek rates, closed form |
| `bond_P0T` | Vasicek zero-coupon bond price P(0, T) |

### `summary.json`

The config as run, the resolved model parameters and strike, P(0, T), total runtime, and the software versions and platform.
