"""Builds today's universe snapshot and refreshes the point-in-time data behind the backtests.

1. CVM filings (fundamentals): the first run downloads 2011 to today (~550 MB, 5 minutes);
   later runs re-read the current and previous years.
2. Today's universe snapshot: B3 listing, liquidity, analyst consensus and fundamentals,
   saved to datasets/<today>/universe.csv.
3. The month-end analyst consensus of the BDRs' US companies since 2012, rebuilt from the
   brokers' rating changes (datasets/reference/analyst_history_us.csv.gz).

Run it regularly (e.g. weekly): each snapshot is kept, and the backtests use the latest one
published before each rebalance date, so the B3 analyst data they use was known at the time.

    python build_dataset.py
    python build_dataset.py --workers 4   # fewer parallel requests if Yahoo rate-limits
"""
import argparse

import analysts
import fundamentals
import universe


def main():
    parser = argparse.ArgumentParser(description="Build today's universe snapshot.")
    parser.add_argument("--workers", type=int, default=8, help="parallel requests to Yahoo Finance")
    args = parser.parse_args()
    fundamentals.update_table()
    fundamentals.ticker_table()
    df = universe.build_universe(workers=args.workers)
    bdrs = df[(df["asset_class"] == "BDRs") & (df["sub_class"] == "BDR") & df["underlying"].notna()]
    reference = bdrs.drop_duplicates("underlying").set_index("underlying")["rec_mean"]
    analysts.build_us_history(bdrs["underlying"].unique(), reference=reference, workers=args.workers)


if __name__ == "__main__":
    main()
