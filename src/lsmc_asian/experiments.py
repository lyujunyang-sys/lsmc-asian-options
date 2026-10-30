"""Experiment configs (JSON, one seed per file) and the tables they produce.

Each table row gets its own random generator, derived from the config seed
and the row label, so any single row can be re-run on its own and adding a
row never changes the others.
"""

from __future__ import annotations

import json
import platform
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from . import __version__
from .analytic import european_price, geometric_asian_price, zero_coupon_bond
from .lsmc import price_lsmc
from .model import MarketModel
from .montecarlo import Estimate, price_european
from .products import AsianOption, VanillaOption

ROOT = Path(__file__).resolve().parents[2]


def rng_for(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(label.encode())]))


def load_config(path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg["_path"] = Path(path).as_posix()
    return cfg


def resolve_market(cfg: dict) -> tuple[MarketModel, float, dict]:
    """MarketModel and strike from a config; ``market_from`` points to a calibration file."""
    info: dict = {}
    if "market_from" in cfg:
        calib = json.loads((ROOT / cfg["market_from"]).read_text(encoding="utf-8"))
        market = {**calib["market"], **cfg.get("market_overrides", {})}
        info["valuation_date"] = calib["valuation_date"]
    else:
        market = cfg["market"]
    model = MarketModel(**market)
    contract = cfg["contract"]
    strike = contract["strike"] if "strike" in contract else round(model.s0 * contract["strike_moneyness"], 2)
    return model, float(strike), info


def _row(product, method, est: Estimate | None, price=None, runtime=None, **extra) -> dict:
    row = {"product": product, "method": method}
    if est is None:
        row.update(price=price, std_error=0.0, ci95_low=price, ci95_high=price, n_samples=None)
    else:
        lo, hi = est.interval()
        row.update(price=est.price, std_error=est.std_error, ci95_low=lo, ci95_high=hi, n_samples=est.n_samples)
    row["runtime_s"] = runtime
    row.update(extra)
    return row


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def pricing_table(cfg: dict) -> pd.DataFrame:
    """Prices of the contracts compared in the paper, each with every applicable method."""
    model, strike, _ = resolve_market(cfg)
    T, N = cfg["contract"]["maturity"], cfg["contract"]["n_steps"]
    sim = cfg["simulation"]
    M, M_oos = sim["n_paths"], sim.get("n_paths_out_of_sample", sim["n_paths"])
    batch, degree, seed = sim.get("batch_size", 20_000), sim.get("lsmc_degree", 2), cfg["seed"]
    rows = []

    for kind in cfg["contract"].get("kinds", ["call"]):
        arith = AsianOption(strike, kind, "arithmetic")
        geo = AsianOption(strike, kind, "geometric")
        vanilla = VanillaOption(strike, kind)
        tag = f"{kind}"

        plain, t_plain = _timed(lambda: price_european(model, arith, T, N, M, rng_for(seed, f"arith-plain-{tag}"), batch_size=batch))
        rows.append(_row(f"arithmetic Asian {kind} (European exercise)", "MC", plain, runtime=t_plain))
        for label, anti, cv in [("MC + antithetic", True, False), ("MC + control variate", False, True), ("MC + antithetic + control variate", True, True)]:
            est, t = _timed(lambda: price_european(model, arith, T, N, M, rng_for(seed, f"arith-{label}-{tag}"), antithetic=anti, control_variate=cv, batch_size=batch))
            rows.append(_row(f"arithmetic Asian {kind} (European exercise)", label, est, runtime=t,
                             variance_reduction=(plain.std_error / est.std_error) ** 2))

        exact_geo, t = _timed(lambda: geometric_asian_price(model, strike, T, N, kind))
        rows.append(_row(f"geometric Asian {kind} (European exercise)", "closed form", None, price=exact_geo, runtime=t))
        est, t = _timed(lambda: price_european(model, geo, T, N, M, rng_for(seed, f"geo-mc-{tag}"), antithetic=True, batch_size=batch))
        rows.append(_row(f"geometric Asian {kind} (European exercise)", "MC + antithetic", est, runtime=t,
                         z_vs_exact=(est.price - exact_geo) / est.std_error))

        exact_eur, t = _timed(lambda: european_price(model, strike, T, kind))
        rows.append(_row(f"European {kind}", "closed form", None, price=exact_eur, runtime=t))
        est, t = _timed(lambda: price_european(model, vanilla, T, N, M, rng_for(seed, f"eur-mc-{tag}"), antithetic=True, batch_size=batch))
        rows.append(_row(f"European {kind}", "MC + antithetic", est, runtime=t,
                         z_vs_exact=(est.price - exact_eur) / est.std_error))

        for label, cv in [("LSMC", False), ("LSMC + control variate", True)]:
            res, t = _timed(lambda: price_lsmc(model, arith, T, N, M, rng_for(seed, f"lsmc-{label}-{tag}"), degree=degree,
                                               antithetic=sim.get("antithetic", True), n_paths_out_of_sample=M_oos, control_variate=cv))
            product = f"arithmetic Asian {kind} (Bermudan exercise)"
            rows.append(_row(product, f"{label}, in-sample", res.in_sample, runtime=t))
            rows.append(_row(product, f"{label}, out-of-sample", res.out_of_sample, runtime=t,
                             early_exercise_premium=res.early_exercise_premium))

    table = pd.DataFrame(rows)
    table.attrs["discount_factor_T"] = zero_coupon_bond(model, T)
    return table


def convergence_table(cfg: dict) -> pd.DataFrame:
    """Price and standard error against the number of paths, and LSMC against the basis degree."""
    model, strike, _ = resolve_market(cfg)
    T, N = cfg["contract"]["maturity"], cfg["contract"]["n_steps"]
    seed, conv = cfg["seed"], cfg["convergence"]
    arith = AsianOption(strike, "call", "arithmetic")
    rows = []
    for M in conv["n_paths"]:
        for label, anti, cv in [("MC", False, False), ("MC + antithetic + control variate", True, True)]:
            est, t = _timed(lambda: price_european(model, arith, T, N, M, rng_for(seed, f"conv-{label}-{M}"), antithetic=anti, control_variate=cv))
            rows.append({"study": "paths", "method": label, "n_paths": M, "degree": None, "price": est.price, "std_error": est.std_error, "runtime_s": t})
    for M in conv["lsmc_n_paths"]:
        res, t = _timed(lambda: price_lsmc(model, arith, T, N, M, rng_for(seed, f"conv-lsmc-{M}"), antithetic=True, n_paths_out_of_sample=M))
        for which, est in [("LSMC in-sample", res.in_sample), ("LSMC out-of-sample", res.out_of_sample)]:
            rows.append({"study": "paths", "method": which, "n_paths": M, "degree": 2, "price": est.price, "std_error": est.std_error, "runtime_s": t})
    M = conv["degree_n_paths"]
    for d in conv["degrees"]:
        res, t = _timed(lambda: price_lsmc(model, arith, T, N, M, rng_for(seed, f"conv-degree-{d}"), degree=d, antithetic=True, n_paths_out_of_sample=M))
        for which, est in [("LSMC in-sample", res.in_sample), ("LSMC out-of-sample", res.out_of_sample)]:
            rows.append({"study": "degree", "method": which, "n_paths": M, "degree": d, "price": est.price, "std_error": est.std_error, "runtime_s": t})
    return pd.DataFrame(rows)


def sensitivity_table(cfg: dict) -> pd.DataFrame:
    """One-at-a-time changes of the rate parameters and the strike around the base case."""
    base, strike, _ = resolve_market(cfg)
    T, N = cfg["contract"]["maturity"], cfg["contract"]["n_steps"]
    seed, sim, sens = cfg["seed"], cfg["simulation"], cfg["sensitivity"]
    M = sim["n_paths"]
    rows = []
    cases = [("base", None, None)]
    for name, values in sens.items():
        cases += [(name, name, v) for v in values]
    for label, name, value in cases:
        params = {k: getattr(base, k) for k in ("s0", "sigma_s", "r0", "kappa", "theta", "sigma_r", "rho", "dividend_yield")}
        k_ = strike
        if name == "strike":
            k_ = float(value)
        elif name is not None:
            params[name] = float(value)
        model = MarketModel(**params)
        arith = AsianOption(k_, "call", "arithmetic")
        tag = f"{label}-{value}"
        eur = price_european(model, arith, T, N, M, rng_for(seed, f"sens-eur-{tag}"), antithetic=True, control_variate=True)
        res = price_lsmc(model, arith, T, N, M, rng_for(seed, f"sens-lsmc-{tag}"), antithetic=True, n_paths_out_of_sample=M, control_variate=True)
        rows.append({
            "parameter": label, "value": value if name else None, "strike": k_,
            "asian_european": eur.price, "asian_european_se": eur.std_error,
            "asian_bermudan_oos": res.out_of_sample.price, "asian_bermudan_oos_se": res.out_of_sample.std_error,
            "geometric_exact": geometric_asian_price(model, k_, T, N),
            "european_vanilla_exact": european_price(model, k_, T),
            "bond_P0T": zero_coupon_bond(model, T),
        })
    return pd.DataFrame(rows)


TABLES = {"pricing": pricing_table, "convergence": convergence_table, "sensitivity": sensitivity_table}


def run(cfg: dict, out_root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Run every table listed in ``cfg["tables"]``; write CSVs and a run summary under results/<name>/."""
    out = (out_root or ROOT / "results") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    model, strike, info = resolve_market(cfg)
    tables, started = {}, time.perf_counter()
    for name in cfg.get("tables", ["pricing"]):
        table = TABLES[name](cfg)
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.6g", lineterminator="\n")
        tables[name] = table
    summary = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_file": cfg.get("_path"),
        "resolved_market": {k: getattr(model, k) for k in ("s0", "sigma_s", "r0", "kappa", "theta", "sigma_r", "rho", "dividend_yield")},
        "strike": strike,
        **info,
        "zero_coupon_bond_P0T": zero_coupon_bond(model, cfg["contract"]["maturity"]),
        "total_runtime_s": round(time.perf_counter() - started, 2),
        "environment": {
            "lsmc_asian": __version__, "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__, "platform": platform.platform(),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return tables
