"""Download the raw market data into data/raw/ and record it in data/manifest.json.

    python scripts/download_data.py                      # 2015-01-01 .. 2023-12-31, as in the paper
    python scripts/download_data.py --start 2010-01-01 --end 2024-12-31
"""

import argparse
from pathlib import Path

from lsmc_asian.data import download_fred, download_sp500, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--rates", nargs="+", default=["DGS10", "DGS3MO"], help="FRED series ids")
    args = ap.parse_args()

    raw = ROOT / "data" / "raw"
    records = [download_sp500(args.start, args.end, raw)]
    records += [download_fred(s, args.start, args.end, raw) for s in args.rates]
    for r in records:
        r["file"] = Path(r["file"]).resolve().relative_to(ROOT).as_posix()
        print(f"{r['file']}: {r['rows']} rows, {r['first_date']} .. {r['last_date']}")
    write_manifest(records, ROOT / "data" / "manifest.json", args.start, args.end)


if __name__ == "__main__":
    main()
