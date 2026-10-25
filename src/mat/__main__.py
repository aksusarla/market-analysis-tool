"""Command line entry point.

    python -m mat backtest                 # full backtest + charts into results/
    python -m mat sweep                    # parameter grid from config -> results/sweep.csv
    python -m mat picks                    # latest ML ranking -> results/picks_<date>.csv
    python -m mat backtest --set backtest.top_k=20 --set model.type=ridge
"""
from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

import pandas as pd

from .backtest import current_picks, prepare_dataset, run_backtest
from .config import load_config, with_overrides
from .data import load_panel
from .metrics import summarize
from .report import write_report

DISCLAIMER = "Educational research tool. Past performance does not predict future returns. Not investment advice."


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["backtest", "sweep", "picks"])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.set)
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.command == "backtest":
        res = run_backtest(cfg)
        summary = write_report(res, cfg, out)
        pd.set_option("display.width", 200)
        print(summary.round(2).to_string())
        print(f"\nData fixes: {res['ds']['n_bad_ticks']} suspected corporate-action gaps zeroed")
    elif args.command == "picks":
        ds = prepare_dataset(cfg)
        picks = current_picks(cfg, ds, n=50)
        path = out / f"picks_{picks.attrs['as_of']}.csv"
        picks.to_csv(path)
        print(f"As of {picks.attrs['as_of']}:\n{picks.head(30).round(3).to_string()}\nsaved {path}")
    else:  # sweep
        grid = cfg["sweep"]["grid"]
        keys = list(grid)
        panel = load_panel(cfg)
        rows, ds_cache = [], {}
        for combo in itertools.product(*grid.values()):
            change = dict(zip(keys, combo))
            c = with_overrides(cfg, change)
            c["backtest"]["strategies"] = ["ml", "rule_trend", "universe_ew"]
            dkey = (c["universe"]["size"], c["model"]["horizon"])
            if dkey not in ds_cache:
                ds_cache[dkey] = prepare_dataset(c, panel)
            res = run_backtest(c, ds_cache[dkey])
            s = summarize(res, c["backtest"]["risk_free"])
            for strat in ("ml", "rule_trend", "universe_ew", "nifty50"):
                rows.append({**change, "strategy": strat, **s.loc[strat].to_dict()})
            pd.DataFrame(rows).round(3).to_csv(out / "sweep.csv", index=False)  # save as we go
            print(change, f"ML CAGR {s.loc['ml', 'CAGR %']:.1f}%  Sharpe {s.loc['ml', 'Sharpe']:.2f}", flush=True)
        df = pd.DataFrame(rows)
        df.round(3).to_csv(out / "sweep.csv", index=False)
        print(df[df.strategy == "ml"].round(2).to_string(index=False))
    print(f"\n{DISCLAIMER}\nDone in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
