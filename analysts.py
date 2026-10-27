"""Sell-side analyst consensus: data, features, filters, ranking score and return projections.

How the evidence shapes the defaults (the tests are in the notebook):
- Ratings: the literature finds that the level of the consensus predicts little and that
  changes, downgrades most of all, predict more (Womack, 1996; Jegadeesh, Kim, Krische &
  Lee, 2004). On this project's 2025-09-10 B3 snapshot, "strong_buy" stocks did worse than
  "buy" and "hold" ones over the next 13 months and the "underperform" ones did worst. On
  10 years of point-in-time consensus for the BDRs' underlyings, the rating level had a
  small positive rank IC (~0.05 at 12 months) and revisions none.
  -> exclude Underperform/Sell, do not restrict the universe to "strong_buy", and keep the
     downgrade filter optional.
- Target prices are optimistic on average, so the absolute upside is a poor forecast; the
  upside relative to peers carries more information (Da & Schaumburg, 2011), though little
  here (IC ~0.04 at 12 months).
  -> projections are relative to peers and get a low default confidence.
- Disagreement among analysts predicted lower returns in Diether, Malloy & Scherbina (2002)
  but higher ones in the recent US sample (partly survivorship), so it is not penalized.
- A consensus of one or two analysts is noise.
  -> require a minimum coverage and shrink the score of lightly covered names.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from market_data import CDI, TESOURO_PREFIX

US_HISTORY_PATH = Path(__file__).resolve().parent / "datasets" / "reference" / "analyst_history_us.csv.gz"
# Broker rating labels -> 1 (strong buy) ... 5 (sell), like Yahoo's consensus scale
GRADE_SCORES = {
    1: ["strong buy", "top pick", "conviction buy"],
    2: ["buy", "outperform", "overweight", "market outperform", "sector outperform", "positive",
        "accumulate", "add", "long-term buy", "speculative buy", "moderate buy", "outperformer",
        "above average", "sector overweight", "trading buy"],
    3: ["hold", "neutral", "market perform", "sector perform", "equal-weight", "equal weight",
        "in-line", "inline", "peer perform", "sector weight", "perform", "mixed", "fair value",
        "market weight", "sector neutral", "average", "sector market weight"],
    4: ["underperform", "underweight", "reduce", "market underperform", "sector underperform",
        "negative", "moderate sell", "below average", "sector underweight"],
    5: ["sell", "strong sell"],
}
_GRADE_LOOKUP = {label: score for score, labels in GRADE_SCORES.items() for label in labels}

RATINGS = ["strong_buy", "buy", "hold", "underperform", "sell"]
RATING_BINS = [0, 1.5, 2.5, 3.5, 4.5, 5.01]  # Yahoo's consensus mean (1-5) -> rating
COVERED_CLASSES = ("Stocks", "BDRs")

# Yahoo Finance quote fields -> snapshot columns
INFO_FIELDS = {
    "recommendationKey": "rec_key",
    "recommendationMean": "rec_mean",
    "numberOfAnalystOpinions": "n_analysts",
    "targetMeanPrice": "target_mean",
    "targetMedianPrice": "target_median",
    "targetHighPrice": "target_high",
    "targetLowPrice": "target_low",
    "currentPrice": "analyst_price",
}
ANALYST_COLUMNS = list(INFO_FIELDS.values()) + ["rec_mean_now", "rec_mean_3m", "n_trend"]

# Weights of the composite score (z-scores within each asset class)
SCORE_WEIGHTS = {"consensus": 0.4, "rel_upside": 0.4, "revision": 0.2, "dispersion": 0.0}

# Expected range of n normal draws in standard deviations, to turn the high-low spread of
# price targets into a standard deviation
_RANGE_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 8: 2.847, 10: 3.078,
             15: 3.472, 20: 3.735, 30: 4.086, 50: 4.498}


def extract_info(info):
    """Analyst fields of a Yahoo Finance quote."""
    row = {col: info.get(key) for key, col in INFO_FIELDS.items()}
    if row["rec_key"] in (None, "none"):
        row["rec_key"] = None
    return row


def _consensus(counts):
    """Mean rating on Yahoo's 1 (strong buy) to 5 (strong sell) scale from rating counts."""
    n = counts[["strongBuy", "buy", "hold", "sell", "strongSell"]].to_numpy(dtype=float)
    total = n.sum()
    return (n @ np.arange(1, 6)) / total if total else np.nan, total


def fetch_trends(tickers, workers=8):
    """Consensus rating now and three months ago, from the monthly rating counts."""
    def one(ticker):
        try:
            df = yf.Ticker(ticker).recommendations
        except Exception:
            return ticker, {}
        if df is None or df.empty or "period" not in df:
            return ticker, {}
        df = df.set_index("period")
        if "0m" not in df.index:
            return ticker, {}
        now, n = _consensus(df.loc["0m"])
        before = _consensus(df.loc["-3m"])[0] if "-3m" in df.index else np.nan
        return ticker, {"rec_mean_now": now, "rec_mean_3m": before, "n_trend": n}

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, tickers))


def grade_score(grade):
    return _GRADE_LOOKUP.get(re.sub(r"\s+", " ", str(grade or "")).strip().lower(), np.nan)


def fetch_upgrades(tickers, workers=8):
    """Yahoo's history of rating changes and price targets, broker by broker, of each US ticker."""
    def one(t):
        try:
            return t, yf.Ticker(t).upgrades_downgrades
        except Exception:
            return t, None
    with ThreadPoolExecutor(workers) as ex:
        return {t: ud for t, ud in ex.map(one, tickers) if ud is not None and len(ud)}


def broker_actions(upgrades):
    """One ticker's broker actions with the date, the broker, the rating on the 1-5 scale and
    the price target."""
    actions = upgrades.reset_index().rename(columns={"GradeDate": "date"})
    actions["date"] = pd.to_datetime(actions["date"]).dt.tz_localize(None)
    actions["score"] = actions["ToGrade"].map(grade_score)
    return actions.dropna(subset=["score"]).sort_values("date")


def consensus_at(actions, when, window_days=365):
    """Consensus on `when` from each broker's latest action of the previous `window_days`:
    (mean rating, number of brokers, their price targets)."""
    recent = actions[(actions["date"] <= when) & (actions["date"] > when - pd.Timedelta(days=window_days))]
    last = recent.groupby("Firm").tail(1)
    targets = last.loc[last["currentPriceTarget"] > 0, "currentPriceTarget"]
    return last["score"].mean(), len(last), targets


def build_us_history(underlyings, start="2012-01-31", end=None, reference=None, workers=8, log=print):
    """Month-end analyst consensus of the BDRs' underlying US companies, rebuilt from the
    brokers' rating changes and price targets known at each month end, saved to
    US_HISTORY_PATH. Backtests use it so that a BDR's analyst data is the one of each date.

    Brokers' own labels give a consensus on a different scale from Yahoo's (its mean was
    0.44 higher and its spread narrower, so no company reached "strong buy"). With
    `reference` (Yahoo's current consensus by ticker), the rebuilt values are mapped onto
    Yahoo's scale by a linear fit on the latest month (the raw values are kept), so the rating
    filters mean the same in backtests as in the app."""
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    log(f"Downloading broker histories of {len(underlyings)} US companies...")
    upgrades = fetch_upgrades(sorted(set(underlyings)), workers)
    prices = yf.download(sorted(upgrades), start=pd.Timestamp(start) - pd.Timedelta(days=10), end=end,
                         auto_adjust=True, progress=False)["Close"]
    months = pd.date_range(start, end, freq="ME")
    rows = []
    for ticker, ud in upgrades.items():
        if ticker not in prices:
            continue
        actions, px = broker_actions(ud), prices[ticker].dropna()
        for d in months:
            rating, n, targets = consensus_at(actions, d)
            if n == 0 or px.loc[:d].empty:
                continue
            before = consensus_at(actions, d - pd.Timedelta(days=91))[0]
            rows.append({"date": d, "underlying": ticker, "rec_mean": rating, "n_analysts": n,
                         "target_mean": targets.mean() if len(targets) else np.nan,
                         "target_high": targets.max() if len(targets) else np.nan,
                         "target_low": targets.min() if len(targets) else np.nan,
                         "analyst_price": px.loc[:d].iloc[-1], "rec_mean_3m": before})
    history = calibrate_scale(pd.DataFrame(rows), reference, log)
    US_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(US_HISTORY_PATH, index=False)
    log(f"Saved {len(history):,} month-end consensus records to {US_HISTORY_PATH}")
    return history


def calibrate_scale(history, reference=None, log=print, min_pairs=20):
    """Maps the rebuilt consensus onto Yahoo's scale with a linear fit on the latest month
    (keeping the raw values in *_raw columns); unchanged without enough reference values."""
    history = history.copy()
    history["rec_mean_raw"], history["rec_mean_3m_raw"] = history["rec_mean"], history["rec_mean_3m"]
    if reference is None or history.empty:
        return history
    last = history[history["date"] == history["date"].max()].set_index("underlying")
    pairs = pd.DataFrame({"raw": last["rec_mean_raw"], "yahoo": reference}).dropna()
    if len(pairs) < min_pairs:
        return history
    slope, intercept = np.polyfit(pairs["raw"], pairs["yahoo"], 1)
    log(f"Rating scale calibrated on {len(pairs)} companies: Yahoo = {intercept:.2f} + {slope:.2f} x rebuilt "
        f"(correlation {pairs['raw'].corr(pairs['yahoo']):.2f})")
    for col in ("rec_mean", "rec_mean_3m"):
        history[col] = (intercept + slope * history[col + "_raw"]).clip(1.0, 5.0)
    return history


def load_us_history():
    if not US_HISTORY_PATH.exists():
        return None
    return pd.read_csv(US_HISTORY_PATH, parse_dates=["date"])


def apply_us_history(df, when, history, max_age_days=45):
    """Replaces the analyst data of the BDRs in a universe snapshot with the consensus known
    on `when` (the latest month end before it); BDRs without coverage then lose their
    analyst data. Recomputes the features and scores."""
    when = pd.Timestamp(when)
    known = history[(history["date"] <= when) & (history["date"] > when - pd.Timedelta(days=max_age_days))]
    latest = known.sort_values("date").groupby("underlying").tail(1).set_index("underlying")
    out = df.copy()
    bdr = out["asset_class"] == "BDRs"
    # Legacy snapshots do not store the underlying company: use the cached BDR mapping
    bdr_map = US_HISTORY_PATH.parent / "bdr_underlying.csv"
    out["underlying"] = out["underlying"].astype(object) if "underlying" in out else None
    if bdr_map.exists():
        mapping = pd.read_csv(bdr_map).dropna(subset=["underlying"]).set_index("bdr")["underlying"]
        out.loc[bdr, "underlying"] = out.loc[bdr, "underlying"].fillna(out.loc[bdr, "symbol"].map(mapping))
    cols = ["rec_mean", "n_analysts", "target_mean", "target_high", "target_low", "analyst_price", "rec_mean_3m"]
    for col in cols:
        out.loc[bdr, col] = out.loc[bdr, "underlying"].map(latest[col]) if len(latest) else np.nan
    out.loc[bdr, "rec_mean_now"] = out.loc[bdr, "rec_mean"]
    rated = bdr & out["rec_mean"].notna()
    out.loc[bdr, "rec_key"] = None
    if rated.any():
        out.loc[rated, "rec_key"] = pd.cut(out.loc[rated, "rec_mean"], RATING_BINS, labels=RATINGS).astype(str)
    return composite_score(add_features(out))


def add_features(df):
    """Upside to the price target, dispersion of targets, recent rating revision, and the
    analysts' 12-month total return projection (upside + dividend yield), also relative to peers."""
    out = df.copy()
    for col in ANALYST_COLUMNS + ["dividend_yield", "sector"]:  # legacy snapshots lack some
        if col not in out:
            out[col] = np.nan
    # Yahoo sometimes omits the consensus while the monthly rating counts exist
    out["rec_mean"] = pd.to_numeric(out["rec_mean"], errors="coerce").fillna(out["rec_mean_now"])
    missing = out["rec_key"].isna() & out["rec_mean"].notna()
    if missing.any():
        out.loc[missing, "rec_key"] = pd.cut(out.loc[missing, "rec_mean"], RATING_BINS, labels=RATINGS).astype(str)
    price = pd.to_numeric(out["analyst_price"], errors="coerce").where(lambda x: x > 0)
    out["upside"] = (out["target_mean"] / price - 1).clip(-0.9, 2.0)
    out["dispersion"] = (out["target_high"] - out["target_low"]) / price
    # Positive revision: the consensus moved toward "buy" over the last three months
    out["revision"] = out["rec_mean_3m"] - out["rec_mean_now"]
    out["projected_return"] = out["upside"] + out["dividend_yield"].fillna(0.0)
    out["rel_upside"] = np.nan
    has_view = out["projected_return"].notna() & out["asset_class"].isin(COVERED_CLASSES)
    for _, grp in out[has_view].groupby("asset_class"):
        peers = grp["projected_return"].median()
        rel = grp["projected_return"] - peers
        # Within a sector when it has enough covered names, which is closer to the evidence
        for _, sec in grp.groupby(grp["sector"].fillna("")):
            if len(sec) >= 5:
                rel.loc[sec.index] = sec["projected_return"] - sec["projected_return"].median()
        out.loc[rel.index, "rel_upside"] = rel
    return out


def composite_score(df, weights=None, prior_n=3):
    """Weighted z-scores of the consensus, peer-relative upside, revision and dispersion,
    computed within each asset class and shrunk toward zero when coverage is thin
    (multiplied by n / (n + prior_n))."""
    weights = SCORE_WEIGHTS if weights is None else weights
    out = df.copy()
    out["analyst_score"] = np.nan
    features = {
        "consensus": 3 - out["rec_mean"],
        "rel_upside": out["rel_upside"],
        "revision": out["revision"],
        "dispersion": out["dispersion"],
    }
    n = out["n_analysts"].fillna(0)
    covered = (n >= 1) & out["rec_mean"].notna() & out["asset_class"].isin(COVERED_CLASSES)
    for _, idx in out[covered].groupby("asset_class").groups.items():
        total = pd.Series(0.0, index=idx)
        for name, weight in weights.items():
            x = pd.to_numeric(features[name].loc[idx], errors="coerce")
            if x.std() > 0:
                total += weight * ((x - x.mean()) / x.std()).clip(-3, 3).fillna(0.0)
        out.loc[idx, "analyst_score"] = total * n[idx] / (n[idx] + prior_n)
    return out


@dataclass(frozen=True)
class FilterRules:
    """Universe filters. Analyst rules apply to Stocks and BDRs (BDRs use the consensus on
    the underlying company); FII rules apply to FIIs."""
    classes: tuple = ("Stocks", "BDRs", "FIIs", "Fixed income")
    min_adtv: float = 500_000.0  # average daily traded value, R$
    ratings: tuple = ("strong_buy", "buy", "hold")
    min_analysts: int = 3
    min_upside: float | None = None  # e.g. 0.0 keeps names whose target is above the price
    max_downgrade: float | None = None  # e.g. 0.25: consensus worsening (1-5 scale) in 3 months
    max_dispersion: float | None = None  # (high - low target) / price
    include_uncovered: bool = False  # keep Stocks/BDRs without enough analysts
    fii_min_dy: float | None = None
    fii_max_pvp: float | None = None
    max_per_class: int | None = 25  # keep the best ranked of each class
    rank_by: str = "auto"  # "auto", "score" (analyst), "fundamentals" (CVM factor score) or "liquidity"


# Ranking within each class; assets without the score fall back to liquidity (avg_volume,
# in shares, only exists in legacy snapshots, which have no adtv). "auto" ranks stocks by the
# fundamental score, which predicted returns far better than the consensus in the notebook,
# and the rest by liquidity: ranking BDRs by the analyst score did worse than by liquidity in
# the point-in-time backtest (2020-2026).
RANK_KEYS = {
    "auto": ["fundamental_score", "adtv", "avg_volume"],
    "score": ["analyst_score", "adtv", "avg_volume"],
    "fundamentals": ["fundamental_score", "adtv", "avg_volume"],
    "liquidity": ["adtv", "avg_volume"],
}


def off_exchange(symbols):
    """The CDI and Tesouro Direto bonds: no traded volume, redeemable any day."""
    symbols = pd.Series(symbols)
    return (symbols == CDI) | symbols.str.startswith(TESOURO_PREFIX)


def apply_filters(df, rules):
    """Adds a 'status' column: 'selected', or the first rule that excluded the asset."""
    out = df.copy()
    for col in ANALYST_COLUMNS + ["adtv", "avg_volume", "analyst_score", "upside", "revision",
                                  "dispersion", "dividend_yield", "price_to_book", "fundamental_score"]:
        if col not in out:
            out[col] = np.nan
    status = pd.Series("selected", index=out.index)

    def exclude(mask, reason):
        status[(status == "selected") & mask.fillna(False).astype(bool)] = reason

    exclude(~out["asset_class"].isin(rules.classes), "asset class")
    exempt = off_exchange(out["symbol"]).set_axis(out.index)
    if out["adtv"].notna().any():
        exclude((~(out["adtv"] >= rules.min_adtv)) & ~exempt, "liquidity")
    analyst = out["asset_class"].isin(COVERED_CLASSES)
    # Legacy snapshots only have the stocks' rating (the BDR history can still fill the BDRs' counts)
    counted = out.groupby("asset_class", dropna=False)["n_analysts"].transform(lambda s: s.notna().any()).astype(bool)
    covered = (out["n_analysts"] >= rules.min_analysts).where(counted, out["rec_key"].notna())
    if not rules.include_uncovered:
        exclude(analyst & ~covered, "analyst coverage")
    rated = analyst & covered
    exclude(rated & ~out["rec_key"].isin(rules.ratings), "consensus rating")
    if rules.min_upside is not None:
        exclude(rated & (out["upside"] < rules.min_upside), "upside to target")
    if rules.max_downgrade is not None:
        exclude(rated & (out["revision"] < -rules.max_downgrade), "recent downgrade")
    if rules.max_dispersion is not None:
        exclude(rated & (out["dispersion"] > rules.max_dispersion), "analyst disagreement")
    fii = out["asset_class"] == "FIIs"
    if rules.fii_min_dy is not None:
        exclude(fii & (out["dividend_yield"] < rules.fii_min_dy), "dividend yield")
    if rules.fii_max_pvp is not None:
        exclude(fii & (out["price_to_book"] > rules.fii_max_pvp), "P/VP")
    if rules.max_per_class:
        selected = out[(status == "selected") & ~exempt]
        keys = RANK_KEYS[rules.rank_by]
        for _, grp in selected.groupby("asset_class"):
            ranked = grp.sort_values(keys, ascending=False, na_position="last")
            exclude(out.index.to_series().isin(ranked.index[rules.max_per_class:]), "ranking cut")
    out["status"] = status
    return out


def _target_std(dispersion, n):
    keys = np.array(list(_RANGE_D2))
    d2 = np.interp(np.log(np.clip(n, 2, 50)), np.log(keys), list(_RANGE_D2.values()))
    return dispersion / d2


def projections(df, prior_annual, relative=True, analyst_std=1.0, fii_std=0.10, min_analysts=3,
                use_analysts=True, use_fii_yield=True):
    """12-month expected-return views for the Black-Litterman blend (optimizer.black_litterman).

    - One view per Stock/BDR with at least `min_analysts`: the target-price upside plus the
      dividend yield. With `relative` (default), the view only tilts the asset's prior by how
      much its projection beats its peers', which removes the analysts' average optimism.
      Its uncertainty is `analyst_std` plus the standard error of the consensus target. The
      default of 100% a year reflects how little the consensus predicted in this project's
      10-year point-in-time test on the BDRs' underlyings (rank IC of about 0.04).
    - One view on the equal-weighted basket of FIIs: it returns their median dividend yield
      (carry with a flat price), with uncertainty `fii_std`. The yield is used for the level
      of the class only: across funds it barely predicted returns and the highest yields did
      worst (yield traps), in the notebook's 2022-2026 test.
    df: universe rows indexed by symbol; prior_annual: annual expected returns of the assets
    being optimized. Returns (P, q, sd): view portfolios (views x assets), annual expected
    returns and annual standard deviations.
    """
    rows = df.reindex(prior_annual.index)
    P, q, sd = {}, {}, {}
    if use_analysts:
        n = rows["n_analysts"].fillna(0)
        ok = rows["asset_class"].isin(COVERED_CLASSES) & (n >= min_analysts) & rows["projected_return"].notna()
        if relative:
            ok &= rows["rel_upside"].notna()
        for s in rows.index[ok]:
            r = rows.loc[s]
            sd_mean = _target_std(r["dispersion"], r["n_analysts"]) / np.sqrt(r["n_analysts"])
            P[s] = pd.Series({s: 1.0})
            q[s] = prior_annual[s] + r["rel_upside"] if relative else r["projected_return"]
            sd[s] = float(np.hypot(analyst_std, np.nan_to_num(sd_mean)))
    if use_fii_yield:
        fiis = rows.index[(rows["asset_class"] == "FIIs") & rows["dividend_yield"].gt(0)]
        if len(fiis):
            P["FII carry"] = pd.Series(1 / len(fiis), index=fiis)
            q["FII carry"] = float(rows.loc[fiis, "dividend_yield"].median())
            sd["FII carry"] = fii_std
    P = pd.DataFrame(P).T.reindex(columns=prior_annual.index).fillna(0.0)
    return P, pd.Series(q, dtype=float).reindex(P.index), pd.Series(sd, dtype=float).reindex(P.index)
