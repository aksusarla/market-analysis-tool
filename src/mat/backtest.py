"""Walk-forward training, strategy scoring and portfolio simulation."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import clean_returns, load_panel
from .features import FEATURES, build_universe, compute_features, rank_features, trading_dates
from .models import RankModel

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- dataset
def prepare_dataset(cfg: dict, panel: dict | None = None) -> dict:
    """Build (or load from cache) the ranked feature table plus clean prices."""
    panel = panel or load_panel(cfg)
    ret, px, n_bad = clean_returns(panel["close"], cfg["data"]["bad_tick_threshold"])
    idx = panel["close"].index
    h = cfg["model"]["horizon"]
    warm = idx[min(260, len(idx) - 1)]  # need ~1y of history before first feature date
    sample = trading_dates(idx, str(warm.date()), None, cfg["features"]["sample_every"])
    rebal = trading_dates(idx, cfg["backtest"]["start"], cfg["backtest"]["end"], h)
    dates = sample.union(rebal).union(idx[-1:])

    key = json.dumps(["features-v2", cfg["data"], cfg["universe"], h, cfg["features"], str(idx[-1].date())], sort_keys=True)
    cache = Path(cfg["data"]["cache_dir"]) / f"features_{hashlib.md5(key.encode()).hexdigest()[:10]}.parquet"
    if cache.exists():
        ranked = pd.read_parquet(cache)
    else:
        universe = build_universe(panel["close"], panel["volume"], dates, cfg["universe"])
        log.info("computing features on %d dates (avg universe %.0f)", len(dates), universe.sum(axis=1).mean())
        raw = compute_features(panel, px, ret, universe, h)
        ranked = rank_features(raw)
        ranked["raw_mom_12_1"] = raw["mom_12_1"]
        ranked["raw_vol_3m"] = raw["vol_3m"]
        ranked.to_parquet(cache)
    return {"panel": panel, "px": px, "ranked": ranked, "sample": sample, "rebal": rebal, "n_bad_ticks": n_bad}


# --------------------------------------------------------------------------- scoring
def walk_forward_ml(ds: dict, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-sample ML scores on rebalance dates, retraining on a rolling window.

    A training row (date d) is only used once its label is fully known: d + 1 + horizon < T,
    where T is the retrain date. This embargo prevents look-ahead leakage.
    """
    mcfg = cfg["model"]
    h = mcfg["horizon"]
    df = ds["ranked"]
    idx = ds["panel"]["close"].index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    feats = list(FEATURES)
    sample_set = set(ds["sample"])
    train = df[df.index.get_level_values("date").isin(sample_set)].dropna(subset=["target"])
    tdates = train.index.get_level_values("date")
    tpos = pos.reindex(tdates).to_numpy()

    rebal = ds["rebal"]
    retrain_at, last = [], None
    for t in rebal:
        if last is None or (t - last).days >= 30.4 * mcfg["retrain_every_months"]:
            retrain_at.append(t)
            last = t

    scores, importances = [], []
    model = None
    for t in rebal:
        if t in retrain_at:
            lo = t - pd.DateOffset(years=mcfg["train_years"])
            m = (tpos + 1 + h < pos[t]) & (tdates >= lo)
            Xtr, ytr = train.loc[m, feats], train.loc[m, "target"]
            if len(Xtr) < 5000:
                log.warning("skip %s: only %d training rows", t.date(), len(Xtr))
                continue
            model = RankModel(mcfg["type"], mcfg).fit(Xtr, ytr)
            importances.append(model.importance().rename(t))
            log.info("retrained %s on %d rows", t.date(), len(Xtr))
        if model is None:
            continue
        Xt = df.xs(t, level="date")[feats]
        scores.append(pd.Series(model.predict(Xt), index=pd.MultiIndex.from_product([[t], Xt.index])))
    s = pd.concat(scores) if scores else pd.Series(dtype=float)
    s.index.names = ["date", "symbol"]
    return s, pd.DataFrame(importances)


def rule_scores(ds: dict) -> dict[str, pd.Series]:
    df = ds["ranked"]
    df = df[df.index.get_level_values("date").isin(set(ds["rebal"]))]
    return {
        "momentum_12_1": df["raw_mom_12_1"],
        "low_vol": -df["raw_vol_3m"],
        # The "momentum radar" rule: strong 12-1 momentum, near 52w highs, rising trend
        "rule_trend": df[["mom_12_1", "dist_52w_high", "ma50_ma200"]].mean(axis=1, skipna=False),
        "universe_ew": pd.Series(1.0, index=df.index),
    }


def top_k(scores: pd.Series, k: int | None, buffer: float = 1.0) -> dict[pd.Timestamp, list[str]]:
    """Pick the top-k each rebalance. With buffer > 1, a stock already held is kept while it
    still ranks inside the top (buffer * k): this cuts churn from names hovering near the cut-off."""
    out, held = {}, []
    for t, s in scores.dropna().groupby(level="date"):
        s = s.droplevel("date")
        if k is None:
            out[t] = list(s.index)
            continue
        rank = s.rank(ascending=False, method="first")
        keep = [x for x in held if x in rank.index and rank[x] <= buffer * k]
        keep = sorted(keep, key=lambda x: rank[x])[:k]
        new = [x for x in rank.sort_values().index if x not in keep][: k - len(keep)]
        held = keep + new
        out[t] = held
    return out


# --------------------------------------------------------------------------- simulation
def simulate(holdings: dict, px: pd.DataFrame, cost_bps: float) -> tuple[pd.Series, pd.DataFrame]:
    """Equal-weight at each rebalance, buy-and-hold between rebalances, costs on turnover.

    Signal on date t (close) -> trade at close of t+1. Returns daily NAV and per-period stats.
    """
    pxf = px.ffill()
    idx = pxf.index
    dates = sorted(holdings)
    entries = [idx.get_loc(t) + 1 for t in dates if idx.get_loc(t) + 1 < len(idx)]
    nav_parts, rows = [], []
    nav, w_old = 1.0, pd.Series(dtype=float)
    for i, e in enumerate(entries):
        end = entries[i + 1] if i + 1 < len(entries) else len(idx) - 1
        if end <= e:
            break
        syms = [s for s in holdings[dates[i]] if pd.notna(pxf.iat[e, pxf.columns.get_loc(s)])]
        if not syms:
            continue
        w_new = pd.Series(1.0 / len(syms), index=syms)
        turnover = w_new.sub(w_old, fill_value=0).abs().sum()
        cost = turnover * cost_bps / 1e4
        sub = pxf.iloc[e : end + 1][syms]
        rel = sub / sub.iloc[0]
        path = rel.mean(axis=1) * nav * (1 - cost)
        if i == 0:
            nav_parts.append(pd.Series([nav], index=[idx[e]]))
        nav_parts.append(path.iloc[1:])
        period_ret = path.iloc[-1] / nav - 1
        rows.append({"decision": dates[i], "entry": idx[e], "exit": idx[end], "n": len(syms),
                     "turnover": turnover, "cost": cost, "ret": period_ret})
        nav = float(path.iloc[-1])
        last = rel.iloc[-1]
        w_old = last / last.sum()
    return pd.concat(nav_parts), pd.DataFrame(rows)


def information_coefficient(scores: pd.Series, ranked: pd.DataFrame) -> pd.DataFrame:
    """Per-date Spearman IC and top-minus-bottom decile spread of forward returns."""
    fwd = ranked["fwd_ret"].reindex(scores.index)
    rows = []
    for t, s in scores.groupby(level="date"):
        d = pd.DataFrame({"s": s.droplevel("date"), "f": fwd.xs(t, level="date")}).dropna()
        if len(d) < 30:
            continue
        ic = d["s"].rank().corr(d["f"].rank())
        q = pd.qcut(d["s"].rank(method="first"), 10, labels=False)
        rows.append({"date": t, "ic": ic, "spread": d.loc[q == 9, "f"].mean() - d.loc[q == 0, "f"].mean()})
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- orchestration
def run_backtest(cfg: dict, ds: dict | None = None) -> dict:
    ds = ds or prepare_dataset(cfg)
    bcfg = cfg["backtest"]
    k = bcfg["top_k"]
    strat_scores = rule_scores(ds)
    importances = pd.DataFrame()
    if "ml" in bcfg["strategies"]:
        ml, importances = walk_forward_ml(ds, cfg)
        strat_scores["ml"] = ml
    # Align every strategy to the dates the ML model can trade so comparisons are fair
    common = set(strat_scores["ml"].index.get_level_values("date")) if "ml" in strat_scores else None

    navs, periods, ics = {}, {}, {}
    for name in bcfg["strategies"]:
        sc = strat_scores[name]
        if common is not None:
            sc = sc[sc.index.get_level_values("date").isin(common)]
        hold = top_k(sc, None if name == "universe_ew" else k, bcfg.get("buffer", 1.0))
        navs[name], periods[name] = simulate(hold, ds["px"], bcfg["cost_bps_per_side"])
        if name != "universe_ew":
            ics[name] = information_coefficient(sc, ds["ranked"])

    first = min(n.index[0] for n in navs.values())
    bench = ds["panel"]["bench"].loc[first:]
    navs["nifty50"] = bench / bench.iloc[0]
    return {"navs": navs, "periods": periods, "ics": ics, "importances": importances,
            "scores": strat_scores, "ds": ds}


def current_picks(cfg: dict, ds: dict, n: int = 30) -> pd.DataFrame:
    """Train on all labelled history and score the latest date."""
    mcfg = cfg["model"]
    h = mcfg["horizon"]
    df = ds["ranked"]
    idx = ds["panel"]["close"].index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    feats = list(FEATURES)
    last = idx[-1]
    train = df[df.index.get_level_values("date").isin(set(ds["sample"]))].dropna(subset=["target"])
    td = train.index.get_level_values("date")
    m = (pos.reindex(td).to_numpy() + 1 + h < len(idx)) & (td >= last - pd.DateOffset(years=mcfg["train_years"]))
    model = RankModel(mcfg["type"], mcfg).fit(train.loc[m, feats], train.loc[m, "target"])
    X = df.xs(last, level="date")
    out = pd.DataFrame({"ml_score": model.predict(X[feats])}, index=X.index)
    out["rule_trend"] = X[["mom_12_1", "dist_52w_high", "ma50_ma200"]].mean(axis=1)
    raw_px = ds["panel"]["close"].loc[last]
    out["close"] = raw_px.reindex(out.index)
    out["mom_12_1_%"] = (X["raw_mom_12_1"] * 100).round(1)
    out["vol_3m_%"] = (X["raw_vol_3m"] * 100).round(1)
    for c in ["dist_52w_high", "deliv_change", "volume_surge"]:
        out[f"{c}_rank"] = X[c].round(2)
    out = out.sort_values("ml_score", ascending=False)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.attrs["as_of"] = str(last.date())
    return out.head(n)
