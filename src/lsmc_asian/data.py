"""Download and load the market data used in the empirical section.

* S&P 500 index (^GSPC) daily close from Yahoo Finance, via ``yfinance``.
* US Treasury constant-maturity yields (FRED series DGS10, DGS3MO), in percent.

Raw files go to ``data/raw/`` (not committed; S&P index data is licensed by
S&P Dow Jones Indices). ``data/manifest.json`` records where each file came
from, when, its date range, row count and SHA-256, so a later download can be
checked against the one used for the committed results. See
``docs/data_dictionary.md`` for every column.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
YAHOO_TICKER = "^GSPC"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, source: str, frame: pd.DataFrame) -> dict:
    return {
        "file": path.as_posix(),
        "source": source,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(len(frame)),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
        "sha256": _sha256(path),
    }


def download_sp500(start: str, end: str, out_dir: Path) -> dict:
    """Daily close of ^GSPC for [start, end] (inclusive) -> sp500_close.csv (date, close)."""
    import yfinance as yf  # only needed for downloading

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(YAHOO_TICKER, start=start, end=end_exclusive, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data for ^GSPC")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance >= 0.2.48 returns a column per ticker
        close = close.iloc[:, 0]
    frame = pd.DataFrame({"date": close.index.strftime("%Y-%m-%d"), "close": close.round(4).to_numpy()})
    path = Path(out_dir) / "sp500_close.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return _record(path, f"Yahoo Finance via yfinance, ticker {YAHOO_TICKER}, field Close", frame)


def download_fred(series: str, start: str, end: str, out_dir: Path) -> dict:
    """FRED series (percent, daily) -> fred_<series>.csv (date, value_pct); missing days are dropped."""
    url = FRED_URL.format(series=series, start=start, end=end)
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    raw = pd.read_csv(io.StringIO(text), na_values=".")
    raw.columns = ["date", "value_pct"]
    frame = raw.dropna().reset_index(drop=True)
    path = Path(out_dir) / f"fred_{series}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return _record(path, url, frame)


def write_manifest(records: list[dict], path: Path, start: str, end: str) -> None:
    manifest = {"window": {"start": start, "end": end}, "files": records}
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_sp500(path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date")["close"].astype(float).rename("sp500")


def load_rate(path) -> pd.Series:
    """FRED yield file -> Series of decimal rates (3.88% -> 0.0388) indexed by date."""
    frame = pd.read_csv(path, parse_dates=["date"])
    return (frame.set_index("date")["value_pct"].astype(float) / 100.0).rename(Path(path).stem)
