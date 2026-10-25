"""Synthetic-data tests for the parts most likely to hide bugs: costs, timing and look-ahead."""
import numpy as np
import pandas as pd

from mat.backtest import simulate
from mat.data import clean_returns
from mat.features import build_universe, compute_features

IDX = pd.bdate_range("2015-01-01", periods=600)


def _panel(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, size=(len(IDX), n))
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=IDX, columns=[f"S{i}" for i in range(n)])
    vol = pd.DataFrame(rng.integers(1e5, 1e6, size=close.shape), index=IDX, columns=close.columns).astype(float)
    bench = close.mean(axis=1)
    return {"close": close, "volume": vol, "deliv": vol * 0.5, "bench": bench}


def test_simulate_flat_prices_only_loses_costs():
    px = pd.DataFrame(10.0, index=IDX[:100], columns=["A", "B"])
    hold = {IDX[0]: ["A", "B"], IDX[50]: ["A", "B"]}
    nav, periods = simulate(hold, px, cost_bps=25)
    # first rebalance buys 100% (25bp); second keeps the same stocks (no turnover)
    assert np.isclose(nav.iloc[-1], 1 - 0.0025)
    assert np.isclose(periods["turnover"].iloc[1], 0.0)


def test_simulate_trades_day_after_signal():
    px = pd.DataFrame({"A": [10.0] * 5 + [20.0] * 5}, index=IDX[:10])
    # signal on day 4 (price 10) -> entry at day 5's close (20): the jump must NOT be captured
    nav, _ = simulate({IDX[4]: ["A"]}, px, cost_bps=0)
    assert np.isclose(nav.iloc[-1], 1.0)


def test_features_do_not_use_future_prices():
    p = _panel()
    ret, px, _ = clean_returns(p["close"], 0.35)
    dates = IDX[300::21]
    uni = build_universe(p["close"], p["volume"], dates, {"liquidity_window": 63, "min_price": 1, "min_history_days": 200, "size": 30})
    base = compute_features(p, px, ret, uni, horizon=21)
    # wreck every price after the 10th feature date; features up to that date must be identical
    cut = dates[10]
    p2 = {k: v.copy() for k, v in p.items()}
    p2["close"].loc[p2["close"].index > cut] *= 3.0
    p2["volume"].loc[p2["volume"].index > cut] *= 7.0
    ret2, px2, _ = clean_returns(p2["close"], 10.0)
    ret, px, _ = clean_returns(p["close"], 10.0)
    a = compute_features(p, px, ret, uni, 21).drop(columns=["fwd_ret"])
    b = compute_features(p2, px2, ret2, uni, 21).drop(columns=["fwd_ret"])
    sel = a.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(a[sel], b[sel], check_exact=False, rtol=1e-4)


def test_features_only_contain_universe_members():
    p = _panel(n=40)
    ret, px, _ = clean_returns(p["close"], 0.35)
    dates = IDX[300::21]
    uni = build_universe(p["close"], p["volume"], dates, {"liquidity_window": 63, "min_price": 1, "min_history_days": 200, "size": 10})
    df = compute_features(p, px, ret, uni, 21)
    per_date = df.groupby(level="date").size()
    assert (per_date <= 10).all() and len(per_date) == len(dates)


def test_clean_returns_removes_split_gap():
    close = pd.DataFrame({"A": [100, 101, 10.1, 10.2]}, index=IDX[:4])
    ret, px, n = clean_returns(close, 0.35)
    assert n == 1 and np.isclose(px["A"].iloc[-1] / px["A"].iloc[0], 1.02, atol=1e-3)
