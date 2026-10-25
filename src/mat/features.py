"""Point-in-time universe, features and labels.

Everything computed for date t uses only data up to and including t's close.
Labels (forward returns) start at t+1's close, i.e. we assume you trade the day after the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature catalogue: name -> one-line description (used in the report)
FEATURES = {
    "ret_1m": "Return over the last month (21 days); short-term reversal signal",
    "ret_3m": "Return over the last 3 months",
    "ret_6m": "Return over the last 6 months",
    "mom_12_1": "12-month return skipping the latest month (classic momentum)",
    "vol_1m": "Annualised volatility, last month",
    "vol_3m": "Annualised volatility, last 3 months",
    "dist_52w_high": "Distance from the 52-week high",
    "dist_52w_low": "Distance above the 52-week low",
    "ma50_ma200": "50-day vs 200-day moving average (trend)",
    "px_vs_ma50": "Price vs its 50-day moving average",
    "rsi_14": "14-day RSI",
    "skew_3m": "Skewness of daily returns, 3 months (lottery-like behaviour)",
    "volume_surge": "5-day vs 63-day average volume",
    "liquidity": "Log median daily traded value, 3 months (size proxy)",
    "deliv_pct": "Delivery % of volume, last month (conviction buying vs intraday churn)",
    "deliv_change": "Change in delivery % vs the prior 3 months",
    "beta_1y": "1-year beta to the Nifty 50",
    "mkt_ret_3m": "Nifty 50 return, last 3 months (market regime)",
    "mkt_vol_3m": "Nifty 50 volatility, last 3 months (market regime)",
}
REGIME_FEATURES = ["mkt_ret_3m", "mkt_vol_3m"]  # same for every stock on a date; not ranked


def trading_dates(index: pd.DatetimeIndex, start: str, end: str | None, step: int) -> pd.DatetimeIndex:
    idx = index[(index >= start) & ((index <= end) if end else True)]
    return idx[::step]


def build_universe(close: pd.DataFrame, volume: pd.DataFrame, dates: pd.DatetimeIndex, ucfg: dict) -> pd.DataFrame:
    """Boolean (dates x symbols): the top-N most liquid tradeable stocks on each date."""
    tv = (close * volume).rolling(ucfg["liquidity_window"], min_periods=40).median()
    history = close.notna().rolling(252, min_periods=1).sum()
    ok = (
        close.notna()
        & (close >= ucfg["min_price"])
        & (history >= ucfg["min_history_days"])
    )
    tv_d = tv.loc[dates].where(ok.loc[dates])
    rank = tv_d.rank(axis=1, ascending=False)
    return (rank <= ucfg["size"]) & tv_d.notna()


def _rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def compute_features(
    panel: dict, px: pd.DataFrame, ret: pd.DataFrame, universe: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Long DataFrame indexed by (date, symbol) with raw features and the forward-return label."""
    dates = universe.index
    close, volume, deliv, bench = panel["close"], panel["volume"], panel["deliv"], panel["bench"]
    pxf = px.ffill()
    cols = {}

    def take(name, wide):
        cols[name] = wide.loc[dates].where(universe).stack(future_stack=True)

    take("ret_1m", pxf / pxf.shift(21) - 1)
    take("ret_3m", pxf / pxf.shift(63) - 1)
    take("ret_6m", pxf / pxf.shift(126) - 1)
    take("mom_12_1", pxf.shift(21) / pxf.shift(252) - 1)
    take("vol_1m", ret.rolling(21, min_periods=15).std() * np.sqrt(252))
    take("vol_3m", ret.rolling(63, min_periods=40).std() * np.sqrt(252))
    hi = pxf.rolling(252, min_periods=120).max()
    lo = pxf.rolling(252, min_periods=120).min()
    take("dist_52w_high", pxf / hi - 1)
    take("dist_52w_low", pxf / lo - 1)
    ma50 = pxf.rolling(50, min_periods=40).mean()
    ma200 = pxf.rolling(200, min_periods=150).mean()
    take("ma50_ma200", ma50 / ma200 - 1)
    take("px_vs_ma50", pxf / ma50 - 1)
    take("rsi_14", _rsi(pxf))
    take("skew_3m", ret.rolling(63, min_periods=40).skew())
    take("volume_surge", volume.rolling(5, min_periods=3).mean() / volume.rolling(63, min_periods=40).mean())
    take("liquidity", np.log1p((close * volume).rolling(63, min_periods=40).median()))
    dpct = deliv.rolling(21, min_periods=10).sum() / volume.rolling(21, min_periods=10).sum()
    dpct_prev = deliv.shift(21).rolling(63, min_periods=30).sum() / volume.shift(21).rolling(63, min_periods=30).sum()
    take("deliv_pct", dpct)
    take("deliv_change", dpct - dpct_prev)
    bret = bench.pct_change(fill_method=None)
    r64 = ret.astype("float64")
    mx = r64.rolling(252, min_periods=150).mean()
    mb = bret.rolling(252, min_periods=150).mean()
    mxb = r64.mul(bret, axis=0).rolling(252, min_periods=150).mean()
    beta = (mxb - mx.mul(mb, axis=0)).div(bret.rolling(252, min_periods=150).var(ddof=0), axis=0)
    take("beta_1y", beta)

    df = pd.DataFrame(cols)
    df.index.names = ["date", "symbol"]
    # keep only (date, stock) pairs that are in that date's universe
    members = universe.stack(future_stack=True)
    df = df[members.reindex(df.index, fill_value=False).to_numpy()]
    mret = (bench / bench.shift(63) - 1).loc[dates]
    mvol = (bret.rolling(63).std() * np.sqrt(252)).loc[dates]
    d = df.index.get_level_values("date")
    df["mkt_ret_3m"] = mret.reindex(d).to_numpy()
    df["mkt_vol_3m"] = mvol.reindex(d).to_numpy()

    # Label: forward return from close(t+1) to close(t+1+h). Delisted -> last traded price.
    fwd = pxf.shift(-(1 + horizon)) / pxf.shift(-1) - 1
    df["fwd_ret"] = fwd.loc[dates].where(universe).stack(future_stack=True).reindex(df.index).to_numpy()
    df = df.replace([np.inf, -np.inf], np.nan).astype("float32")
    return df


def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks per date (robust to outliers and regime shifts)."""
    feats = [c for c in FEATURES if c not in REGIME_FEATURES]
    out = df[feats].groupby(level="date").rank(pct=True).astype("float32")
    for c in REGIME_FEATURES:
        out[c] = df[c]
    out["fwd_ret"] = df["fwd_ret"]
    out["target"] = df["fwd_ret"].groupby(level="date").rank(pct=True).astype("float32")
    return out
