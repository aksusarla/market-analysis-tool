"""Charts and CSV outputs for a backtest run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .features import FEATURES
from .metrics import summarize, yearly_returns

# Fixed categorical order (validated palette); colour follows the strategy, never its rank
COLORS = {
    "ml": "#2a78d6",
    "rule_trend": "#eb6834",
    "momentum_12_1": "#1baf7a",
    "low_vol": "#eda100",
    "universe_ew": "#e87ba4",
    "nifty50": "#52514e",
}
LABELS = {
    "ml": "ML model (LightGBM/Ridge)",
    "rule_trend": "Rule: trend score",
    "momentum_12_1": "Rule: 12-1 momentum",
    "low_vol": "Rule: low volatility",
    "universe_ew": "All universe stocks (equal weight)",
    "nifty50": "Nifty 50 (NIFTYBEES)",
}
SHORT = {
    "ret_1m": "1-month return", "ret_3m": "3-month return", "ret_6m": "6-month return",
    "mom_12_1": "12-1 momentum", "vol_1m": "Volatility (1m)", "vol_3m": "Volatility (3m)",
    "dist_52w_high": "Distance from 52w high", "dist_52w_low": "Distance above 52w low",
    "ma50_ma200": "50 vs 200-day average", "px_vs_ma50": "Price vs 50-day average", "rsi_14": "RSI (14)",
    "skew_3m": "Return skew (3m)", "volume_surge": "Volume surge (5d vs 63d)", "liquidity": "Liquidity / size",
    "deliv_pct": "Delivery %", "deliv_change": "Change in delivery %", "beta_1y": "Beta to Nifty (1y)",
    "mkt_ret_3m": "Market regime: Nifty 3m return", "mkt_vol_3m": "Market regime: Nifty 3m volatility",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def write_report(result: dict, cfg: dict, out_dir: str | Path) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rf = cfg["backtest"]["risk_free"]
    summary = summarize(result, rf)
    summary.round(3).to_csv(out / "summary.csv")
    yearly_returns(result["navs"]).round(2).to_csv(out / "yearly_returns.csv")
    pd.DataFrame(result["navs"]).to_csv(out / "nav_daily.csv")
    for name, p in result["periods"].items():
        p.to_csv(out / f"periods_{name}.csv", index=False)

    navs = result["navs"]
    order = [n for n in COLORS if n in navs]

    # 1. growth of Rs 1 (log scale)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    ends = []
    for n in order:
        nav = navs[n] / navs[n].iloc[0]
        ax.plot(nav.index, nav, color=COLORS[n], lw=2.4 if n in ("ml", "nifty50") else 1.6, label=LABELS[n])
        ends.append((float(nav.iloc[-1]), n, nav.index[-1]))
    # end labels, nudged apart when two strategies finish at almost the same value
    ends.sort()
    last_y = None
    for val, n, x in ends:
        y = val if last_y is None or val / last_y > 1.09 else last_y * 1.09
        ax.annotate(f"{val:.1f}x", (x, y), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=MUTED, va="center")
        last_y = y
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    _style(ax, "Growth of Rs 1, after costs (log scale)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "equity_curves.png")
    plt.close(fig)

    # 2. drawdowns: ML vs Nifty only (two series keeps it readable)
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=130)
    for n in ("ml", "nifty50"):
        if n in navs:
            dd = 100 * (navs[n] / navs[n].cummax() - 1)
            ax.plot(dd.index, dd, color=COLORS[n], lw=1.6, label=LABELS[n])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style(ax, "Drawdown from previous peak")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "drawdowns.png")
    plt.close(fig)

    # 3. rolling 12-period IC of the ML model
    if "ml" in result["ics"] and len(result["ics"]["ml"]):
        ic = result["ics"]["ml"]["ic"]
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=130)
        ax.bar(ic.index, ic, width=18, color=COLORS["ml"], alpha=0.35, label="IC each rebalance")
        ax.plot(ic.index, ic.rolling(12).mean(), color=COLORS["ml"], lw=2, label="12-period average")
        ax.axhline(0, color=MUTED, lw=0.8)
        _style(ax, "ML model: rank correlation between predicted and actual returns (IC)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "ic_over_time.png")
        plt.close(fig)

    # 4. feature importance (average across retrains)
    imp = result["importances"]
    if len(imp):
        s = imp.abs().div(imp.abs().sum(axis=1), axis=0).mean().sort_values()
        s.to_csv(out / "feature_importance.csv", header=["share"])
        fig, ax = plt.subplots(figsize=(8, 5.6), dpi=130)
        ax.barh(range(len(s)), 100 * s.values, color=COLORS["ml"], height=0.62)
        ax.set_yticks(range(len(s)), [SHORT.get(i, i) for i in s.index], fontsize=8.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        _style(ax, "What the model relies on (share of importance)")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png")
        plt.close(fig)
    return summary
