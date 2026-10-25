# Market Analysis Tool

An open-source research tool for Indian equities (NSE): a machine-learning stock ranker, rule-based
baselines, and an honest walk-forward backtest on 12+ years of real market data.

> **Educational research only.** Backtests describe the past, not the future. Nothing here is
> investment advice, and the tool does not issue buy or sell recommendations.

## What it does

| Command | Output |
|---|---|
| `python -m mat backtest` | Walk-forward backtest of the ML model vs four rule-based strategies and the Nifty 50, after costs. Charts + CSVs in `results/` |
| `python -m mat sweep` | Runs a parameter grid (model, horizon, portfolio size, holding buffer, universe size) and writes `results/sweep.csv` |
| `python -m mat picks` | Trains on all history and ranks today's universe: `results/picks_<date>.csv` |

Override any setting from the command line: `python -m mat backtest --set model.type=ridge --set backtest.top_k=20`

## Quick start

```bash
git clone https://github.com/<you>/market-analysis-tool.git && cd market-analysis-tool
pip install -r requirements.txt
bash scripts/download_data.sh          # ~420 MB of NSE daily data, one-time
export PYTHONPATH=src
python -m mat backtest -v              # ~3 minutes on a laptop
```

## Findings (NSE, Jan 2014 to Oct 2026, after 25 bp/side costs)

Default config: Ridge model, top 500 stocks by liquidity, hold 20, rebalance quarterly, keep a holding
until it drops out of the top 40.

| Strategy | CAGR | Sharpe | Max drawdown | Rs 1 became |
|---|---|---|---|---|
| **ML model (Ridge)** | **26.7%** | **1.19** | **-30.6%** | 20.7x |
| Rule: trend score (momentum + near 52w high + 50/200 DMA) | 36.9% | 1.14 | -45.9% | 55.7x |
| Rule: 12-1 momentum | 28.5% | 0.85 | -66.6% | 24.9x |
| Rule: low volatility | 13.3% | 0.58 | -32.7% | 5.0x |
| Top 500 equal weight (fair benchmark) | 16.6% | 0.57 | -59.8% | 7.2x |
| Nifty 50 (NIFTYBEES, price only) | 11.6% | 0.39 | -36.3% | 4.1x |

**32-configuration sweep** (model x horizon x portfolio size x buffer x universe, `results/sweep.csv`):

- **Wider universe wins everywhere.** Top-500 beat top-200 in all 16 pairs by 4-10 points a year.
- **Simple beats complex here.** Ridge averaged a higher Sharpe than LightGBM (0.72 vs 0.63) and won the top 4 slots.
- **Costs decide the winner.** Monthly LightGBM had the strongest raw signal but 4%+ a year of cost drag.
  Quarterly rebalancing plus a holding buffer cut drag to ~1.3% a year.
- **The ML model's edge is risk, not return.** The trend rule makes more money; the ML model has the best
  Sharpe and the smallest drawdown of any equity strategy, including the Nifty 50.

**Read with care:** the headline config is the best of 32, so expect live results to be weaker
(selection bias), and survivorship bias inflates every strategy's absolute numbers (see limitations).

## How it works

```
NSE daily data ──> clean prices ──> point-in-time universe ──> 19 features ──> rank-normalise
 (3,200+ stocks)   (split/demerger    (top 500 by liquidity      (momentum, trend,     per date
                    gaps removed)      on each date)              volatility, beta,
                                                                  volume, delivery %,
                                                                  market regime)
        ──> walk-forward model (LightGBM or Ridge, retrained every 6 months on a rolling 6-year window)
        ──> pick top-K each rebalance ──> simulate: trade next day, equal weight, 25 bp per side costs
```

**Guards against fooling yourself**

- **No look-ahead.** Features on date *t* use data up to *t*'s close only; trades happen at *t+1*'s close.
  A training sample is used only once its label period has fully ended (an embargo). `tests/` includes a
  test that corrupts all future prices and checks past features are unchanged.
- **Point-in-time universe.** Membership is decided from liquidity *on that date*, not today's index list.
- **Costs.** 25 bp per side (STT, stamp duty, exchange fees, GST, slippage) charged on actual turnover.
- **Fair benchmarks.** Every strategy is compared with an equal-weight portfolio of the same universe,
  which shares the same data quirks, not only with the Nifty 50.

**Known limitations**

- **Survivorship bias.** The data source drops stocks a year after delisting, so failed companies are
  missing. This inflates absolute returns for every strategy; compare strategies with each other and with
  `universe_ew`, not with the Nifty 50.
- Prices are adjusted for splits and bonuses only. Dividends are excluded (understates returns by ~1-2%/yr),
  and one-day moves above 35% are treated as unadjusted corporate actions and zeroed.
- Price and volume features only. Fundamentals (results, promoter holding) are not yet point-in-time.

## Project layout

```
config/default.yaml      all parameters + sweep grid
src/mat/data.py          load NSE CSVs into panels, ETF filtering, corporate-action cleaning
src/mat/features.py      universe, 19 features, forward-return labels
src/mat/models.py        LightGBM / Ridge rank models
src/mat/backtest.py      walk-forward training, strategies, portfolio simulation, IC
src/mat/metrics.py       CAGR, Sharpe, Sortino, drawdown, hit rate, turnover
src/mat/report.py        charts and CSVs
tests/                   cost, timing and look-ahead tests
.github/workflows/       weekly refresh of charts and picks
```

## Data

End-of-day NSE data from [eod2_data](https://github.com/BennyThadikaran/eod2_data), built from NSE's
published daily reports (OHLCV + delivery quantities, 1995 onward).
