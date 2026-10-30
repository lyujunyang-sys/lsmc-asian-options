"""Estimate the model inputs from data/raw/ and write data/calibrated/<name>.json.

    python scripts/calibrate.py                 # 10-year yield (as in the paper) and 3-month yield

A rate series that shows no mean reversion over the window (AR(1) slope >= 1)
cannot be fitted by Vasicek; this is reported and no file is written for it.
"""

import argparse
import json
from pathlib import Path

from lsmc_asian.calibration import NotMeanReverting, calibrate_market
from lsmc_asian.data import load_rate, load_sp500

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", nargs="+", default=["DGS10", "DGS3MO"])
    ap.add_argument("--valuation-date", default=None, help="default: last trading day in the data")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20230101)
    args = ap.parse_args()

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    window = manifest["window"]
    prices = load_sp500(ROOT / "data" / "raw" / "sp500_close.csv")
    out_dir = ROOT / "data" / "calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    for series in args.rates:
        rates = load_rate(ROOT / "data" / "raw" / f"fred_{series}.csv")
        try:
            calib = calibrate_market(prices, rates, args.valuation_date, n_bootstrap=args.bootstrap, seed=args.seed)
        except NotMeanReverting as exc:
            print(f"{series}: not calibrated - {exc}")
            continue
        calib["rate_series"] = series
        calib["data_window"] = window
        calib["data_manifest"] = "data/manifest.json"
        name = f"sp500_{window['start'][:4]}-{window['end'][:4]}_{series}.json"
        (out_dir / name).write_text(json.dumps(calib, indent=2) + "\n", encoding="utf-8")
        m, v = calib["market"], calib["diagnostics"]["vasicek"]
        print(f"{series}: S0={m['s0']:.2f} sigma_s={m['sigma_s']:.4f} r0={m['r0']:.4f} "
              f"kappa={m['kappa']:.4f}+-{v['kappa_se']:.4f} theta={m['theta']:.4f}+-{v['theta_se']:.4f} "
              f"sigma_r={m['sigma_r']:.5f} rho={m['rho']:.3f} -> data/calibrated/{name}")


if __name__ == "__main__":
    main()
