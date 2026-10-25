"""Load NSE end-of-day data into wide panels (dates x symbols).

Source: github.com/BennyThadikaran/eod2_data (built from NSE's daily reports).
Prices are adjusted for splits and bonuses, not for dividends, rights or demergers.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Symbols that look like ETFs / index funds / bonds rather than operating companies
NON_EQUITY_PATTERN = re.compile(
    r"(BEES|ETF|NIFTY|SENSEX|GOLD|SILVER|LIQUID|GILT|BHARAT22|CPSE|NV20|MON100|MAFANG|"
    r"HNGSNG|NASDAQ|BOND|SDL|^MOM\d|^MO[A-Z]*\d)",
    re.IGNORECASE,
)


def _read_one(path: Path, start: str) -> tuple[str, pd.DataFrame | None]:
    sym = path.stem.upper()
    try:
        df = pd.read_csv(path, usecols=["Date", "Close", "Volume", "DLV_QTY"], parse_dates=["Date"])
    except Exception as exc:  # malformed file
        log.warning("skip %s: %s", sym, exc)
        return sym, None
    df = df[df["Date"] >= start]
    if df.empty:
        return sym, None
    df = df.drop_duplicates("Date", keep="last").set_index("Date")
    return sym, df


def equity_symbols(data_dir: Path, isin_map_path: Path) -> tuple[list[str], dict]:
    """Return operating-company symbols (ISIN starting INE) and the symbol->ISIN map."""
    sym2isin = json.loads(Path(isin_map_path).read_text()).get("sym2isin", {})
    syms = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        s = p.stem.upper()
        isin = sym2isin.get(s)
        if isin is not None and not isin.startswith("INE"):
            continue  # INF = mutual fund / ETF units
        if isin is None and NON_EQUITY_PATTERN.search(s):
            continue
        syms.append(s)
    return syms, sym2isin


def load_panel(cfg: dict) -> dict[str, pd.DataFrame | pd.Series]:
    """Return {'close','volume','deliv'} wide panels for equities plus 'bench' close series.

    Panels are cached as parquet so later runs load in seconds.
    """
    dcfg = cfg["data"]
    data_dir, cache = Path(dcfg["dir"]), Path(dcfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"start{dcfg['start']}"
    files = {k: cache / f"{k}_{tag}.parquet" for k in ("close", "volume", "deliv", "bench")}
    newest_csv = max(p.stat().st_mtime for p in data_dir.glob("*.csv"))
    if all(f.exists() and f.stat().st_mtime > newest_csv for f in files.values()):
        log.info("loading cached panels")
        out = {k: pd.read_parquet(f) for k, f in files.items()}
        out["bench"] = _clean_series(out["bench"].iloc[:, 0], dcfg["bad_tick_threshold"])
        return out

    syms, _ = equity_symbols(data_dir, Path(dcfg["isin_map"]))
    log.info("reading %d equity files", len(syms))
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = dict(ex.map(lambda s: _read_one(data_dir / f"{s.lower()}.csv", dcfg["start"]), syms))
    frames = {s: f for s, f in frames.items() if f is not None}

    def wide(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: f[col] for s, f in frames.items()}).sort_index().astype("float32")

    close, volume, deliv = wide("Close"), wide("Volume"), wide("DLV_QTY")
    _, bench_df = _read_one(data_dir / f"{dcfg['benchmark'].lower()}.csv", dcfg["start"])
    bench = bench_df["Close"].reindex(close.index).ffill()

    for k, v in (("close", close), ("volume", volume), ("deliv", deliv)):
        v.to_parquet(files[k])
    bench.to_frame("close").to_parquet(files["bench"])
    log.info("panel: %d dates x %d symbols", *close.shape)
    bench = _clean_series(bench, dcfg["bad_tick_threshold"])
    return {"close": close, "volume": volume, "deliv": deliv, "bench": bench}


def _clean_series(s: pd.Series, threshold: float) -> pd.Series:
    """Rebuild a price series from returns with split-like gaps removed (e.g. NIFTYBEES 1:10 in 2019)."""
    r = s.pct_change(fill_method=None)
    r = r.mask(r.abs() > threshold, 0.0).fillna(0.0)
    return (1 + r).cumprod() * s.dropna().iloc[0]


def clean_returns(close: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Daily returns with suspected unadjusted corporate actions removed.

    A one-day move beyond `threshold` (most NSE stocks have 20% circuit limits) is far more
    likely to be a demerger/rights gap in the data than a real trade, so it is set to 0.
    Returns (returns, clean price index starting at 1, number of ticks fixed).
    """
    ret = close.pct_change(fill_method=None)
    bad = ret.abs() > threshold
    n_bad = int(bad.sum().sum())
    ret = ret.mask(bad, 0.0)
    # price index: compound returns, keep NaN where the stock was not trading
    traded = close.notna()
    px = (1.0 + ret.fillna(0.0)).cumprod().where(traded).astype("float32")
    return ret.astype("float32"), px, n_bad
