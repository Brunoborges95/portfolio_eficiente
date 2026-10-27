"""Runs a predefined backtest suite and saves the results to results/<suite>_<end>/.

    python run_backtests.py --suite model                  # 2021 to today
    python run_backtests.py --suite analysts               # point-in-time analyst data only
    python run_backtests.py --suite analysts_bdr           # BDRs with each month's consensus, from 2018
    python run_backtests.py --suite fundamentals           # CVM factor score of each date, from 2016
    python run_backtests.py --suite estimators             # expected-return estimators, 2021 to today
    python run_backtests.py --suite model --lookback 252 --rebalance 63 --cost 0.002

Saves metrics.csv, values.csv (daily value of R$1) and chart.html. See backtesting.suite
for what each suite compares.
"""
import argparse
from datetime import date
from pathlib import Path

import plotly.graph_objects as go

import backtesting


def main():
    parser = argparse.ArgumentParser(description="Run a predefined backtest suite.")
    parser.add_argument("--suite", choices=["model", "estimators", "analysts", "analysts_bdr", "fundamentals"],
                        default="model")
    parser.add_argument("--start", default=None, help="first rebalance (default: 2021-01-04 for 'model' and "
                                                     "'estimators', the day after the first snapshot for "
                                                     "'analysts', 2020-01-02 for 'analysts_bdr', 2016-01-04 "
                                                     "for 'fundamentals')")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--lookback", type=int, default=504, help="trading days of history per decision")
    parser.add_argument("--rebalance", type=int, default=21, help="trading days between rebalances")
    parser.add_argument("--cost", type=float, default=0.001, help="cost per unit of turnover (0.001 = 0.1%%)")
    parser.add_argument("--name", default=None, help="results folder name (default: <suite>_<end>)")
    args = parser.parse_args()
    first_snapshot = backtesting.pd.Timestamp(backtesting.universe.list_snapshots()[0])
    default_start = {"model": "2021-01-04", "estimators": "2021-01-04",
                     "analysts": first_snapshot + backtesting.pd.Timedelta(days=1),
                     "analysts_bdr": "2020-01-02", "fundamentals": "2016-01-04"}
    start = args.start or default_start[args.suite]

    out = backtesting.run_suite(backtesting.suite(args.suite), start, args.end, args.lookback, args.rebalance, args.cost)
    folder = Path(__file__).resolve().parent / "results" / (args.name or f"{args.suite}_{args.end}")
    folder.mkdir(parents=True, exist_ok=True)
    out["metrics"].to_csv(folder / "metrics.csv")
    backtesting.pd.DataFrame(out["values"]).to_csv(folder / "values.csv")
    fig = go.Figure([go.Scatter(x=v.index, y=v, name=k, mode="lines") for k, v in out["values"].items()])
    fig.update_layout(title=f"Backtest '{args.suite}': value of R$1", xaxis_title="Date", yaxis_title="Value")
    fig.write_html(folder / "chart.html")
    print(out["metrics"].round(3).to_string())
    print(f"Saved to {folder}")


if __name__ == "__main__":
    main()
