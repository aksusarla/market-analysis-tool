"""Performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def perf_stats(nav: pd.Series, rf: float, bench: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    daily = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    vol = daily.std() * np.sqrt(252)
    downside = daily[daily < 0].std() * np.sqrt(252)
    dd = nav / nav.cummax() - 1
    out = {
        "CAGR %": 100 * cagr,
        "Volatility %": 100 * vol,
        "Sharpe": (daily.mean() * 252 - rf) / vol if vol else np.nan,
        "Sortino": (daily.mean() * 252 - rf) / downside if downside else np.nan,
        "Max drawdown %": 100 * dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Final value of Rs 1L (Rs L)": nav.iloc[-1] / nav.iloc[0],
    }
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        m_s = nav.resample("ME").last().pct_change().dropna()
        m_b = b.resample("ME").last().pct_change().reindex(m_s.index)
        out["Months beating Nifty %"] = 100 * (m_s > m_b).mean()
    return out


def yearly_returns(navs: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, nav in navs.items():
        y = nav.resample("YE").last()
        first = nav.iloc[0]
        r = y.pct_change()
        r.iloc[0] = y.iloc[0] / first - 1
        rows[name] = 100 * r
    df = pd.DataFrame(rows)
    df.index = df.index.year
    return df


def summarize(result: dict, rf: float) -> pd.DataFrame:
    bench = result["navs"]["nifty50"]
    rows = {}
    for name, nav in result["navs"].items():
        s = perf_stats(nav, rf, None if name == "nifty50" else bench)
        if name in result["periods"]:
            p = result["periods"][name]
            s["Avg turnover / rebalance %"] = 100 * p["turnover"].iloc[1:].mean() if len(p) > 1 else np.nan
            s["Cost drag % / yr"] = 100 * p["cost"].sum() / ((nav.index[-1] - nav.index[0]).days / 365.25)
        if name in result["ics"] and len(result["ics"][name]):
            ic = result["ics"][name]
            s["Mean IC"] = ic["ic"].mean()
            s["IC t-stat"] = ic["ic"].mean() / ic["ic"].std() * np.sqrt(len(ic))
            s["Top-bottom decile spread % / period"] = 100 * ic["spread"].mean()
        rows[name] = s
    return pd.DataFrame(rows).T
