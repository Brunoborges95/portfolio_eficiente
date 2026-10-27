import numpy as np
import pandas as pd

import analysts
from market_data import CDI


def make_universe():
    rows = [
        # symbol, class, rec_key, rec_mean, n, target_mean, high, low, price, rec_3m, adtv, sector
        ("AAA3", "Stocks", "strong_buy", 1.3, 12, 30.0, 36.0, 24.0, 20.0, 1.6, 5e7, "Energy"),
        ("BBB3", "Stocks", "buy", 2.1, 8, 22.0, 26.0, 18.0, 20.0, 2.0, 3e7, "Energy"),
        ("CCC3", "Stocks", "hold", 3.0, 10, 21.0, 25.0, 15.0, 20.0, 2.6, 2e7, "Banks"),
        ("DDD3", "Stocks", "underperform", 3.8, 9, 17.0, 20.0, 12.0, 20.0, 3.8, 2e7, "Banks"),
        ("EEE3", "Stocks", None, None, 1, 25.0, 25.0, 25.0, 20.0, None, 2e7, "Banks"),
        ("FFF3", "Stocks", "buy", 2.0, 6, 24.0, 30.0, 20.0, 20.0, 2.0, 1e5, "Banks"),
        ("GGG34", "BDRs", "buy", 1.9, 30, 110.0, 150.0, 80.0, 100.0, 1.9, 2e6, "Tech"),
    ]
    cols = ["symbol", "asset_class", "rec_key", "rec_mean", "n_analysts", "target_mean", "target_high",
            "target_low", "analyst_price", "rec_mean_3m", "adtv", "sector"]
    df = pd.DataFrame(rows, columns=cols)
    df["rec_mean_now"] = df["rec_mean"]
    df.loc[df.symbol == "BBB3", ["rec_key", "rec_mean"]] = [None, None]  # Yahoo omitted the consensus
    fii = pd.DataFrame({"symbol": ["HHH11", "III11"], "asset_class": "FIIs", "dividend_yield": [0.10, 0.14],
                        "adtv": [2e6, 3e6]})
    cdi = pd.DataFrame({"symbol": [CDI, "TD-IPCA2035"], "asset_class": ["Fixed income", "Fixed income"]})
    return pd.concat([df, fii, cdi], ignore_index=True)


def test_features():
    df = analysts.add_features(make_universe()).set_index("symbol")
    assert np.isclose(df.loc["AAA3", "upside"], 0.5)
    assert np.isclose(df.loc["AAA3", "dispersion"], 0.6)
    assert np.isclose(df.loc["AAA3", "revision"], 0.3)  # 1.6 -> 1.3: upgraded
    # The consensus is rebuilt from the rating counts when Yahoo omits it
    assert df.loc["BBB3", "rec_mean"] == 2.1 and df.loc["BBB3", "rec_key"] == "buy"
    stocks = df[df.asset_class == "Stocks"]
    # Relative upside is measured against the peer group (class, or sector with 5+ names)
    assert np.isclose(stocks["rel_upside"].median(), 0.0)


def test_filters_funnel():
    df = analysts.composite_score(analysts.add_features(make_universe()))
    rules = analysts.FilterRules(classes=("Stocks", "FIIs", "Fixed income"), max_per_class=None)
    status = analysts.apply_filters(df, rules).set_index("symbol")["status"]
    assert status["GGG34"] == "asset class"
    assert status["FFF3"] == "liquidity"
    assert status["EEE3"] == "analyst coverage"  # a single analyst
    assert status["DDD3"] == "consensus rating"
    assert status[[CDI, "TD-IPCA2035"]].eq("selected").all()  # off exchange: no liquidity requirement
    assert status[["AAA3", "BBB3", "CCC3", "HHH11", "III11"]].eq("selected").all()
    loose = analysts.FilterRules(classes=("Stocks",), include_uncovered=True, max_per_class=None)
    assert analysts.apply_filters(df, loose).set_index("symbol").loc["EEE3", "status"] == "selected"
    top1 = analysts.FilterRules(classes=("Stocks", "Fixed income"), max_per_class=1)
    kept = analysts.apply_filters(df, top1)
    best = df[df.symbol.isin(["AAA3", "BBB3", "CCC3"])].sort_values("analyst_score").symbol.iloc[-1]
    # The CDI and Tesouro bonds have no volume to rank by and are never cut
    assert kept.loc[kept.status == "selected", "symbol"].tolist() == [best, CDI, "TD-IPCA2035"]


def test_legacy_coverage_is_decided_per_class():
    # Old snapshots have the stocks' rating only, while the BDR history fills the BDRs' counts
    df = analysts.composite_score(analysts.add_features(make_universe()))
    df.loc[df.asset_class == "Stocks", "n_analysts"] = np.nan
    rules = analysts.FilterRules(classes=("Stocks", "BDRs"), max_per_class=None)
    status = analysts.apply_filters(df, rules).set_index("symbol")["status"]
    assert status["AAA3"] == "selected" and status["GGG34"] == "selected"
    assert status["DDD3"] == "consensus rating" and status["EEE3"] == "analyst coverage"


def test_score_shrinks_with_low_coverage():
    df = analysts.add_features(make_universe())
    few = analysts.composite_score(df, prior_n=3).set_index("symbol")["analyst_score"]
    many = analysts.composite_score(df, prior_n=0).set_index("symbol")["analyst_score"]
    assert abs(few["AAA3"]) < abs(many["AAA3"])
    assert few["AAA3"] > few["DDD3"]


def test_consensus_from_broker_actions():
    ud = pd.DataFrame({
        "Firm": ["A", "B", "A", "C"],
        "ToGrade": ["Buy", "Hold", "Sell", "Outperform"],
        "currentPriceTarget": [120.0, 100.0, 80.0, 0.0],
    }, index=pd.Index(pd.to_datetime(["2024-01-10", "2024-02-10", "2024-06-10", "2023-01-10"]), name="GradeDate"))
    actions = analysts.broker_actions(ud)
    rating, n, targets = analysts.consensus_at(actions, pd.Timestamp("2024-07-01"))
    # Broker A's latest action (sell) replaces its buy; C's action is older than a year
    assert n == 2 and np.isclose(rating, (5 + 3) / 2)
    assert sorted(targets) == [80.0, 100.0]
    assert analysts.consensus_at(actions, pd.Timestamp("2024-03-01"))[1] == 2  # A (buy) and B


def test_calibrate_scale_maps_onto_the_reference():
    rng = np.random.default_rng(0)
    names = [f"T{i}" for i in range(40)]
    raw = pd.Series(rng.uniform(2.0, 3.0, 40), index=names)
    history = pd.DataFrame({"date": pd.Timestamp("2026-09-30"), "underlying": names,
                            "rec_mean": raw.to_numpy(), "rec_mean_3m": raw.to_numpy()})
    yahoo = -1.4 + 1.4 * raw  # Yahoo's scale: lower and wider
    out = analysts.calibrate_scale(history, yahoo, log=lambda *_: None).set_index("underlying")
    assert np.allclose(out["rec_mean"], yahoo[out.index])
    assert np.allclose(out["rec_mean_raw"], raw[out.index])
    unchanged = analysts.calibrate_scale(history, yahoo.iloc[:5], log=lambda *_: None)  # too few pairs
    assert np.allclose(unchanged["rec_mean"], history["rec_mean"])


def test_bdr_analyst_data_is_point_in_time():
    df = analysts.add_features(make_universe())
    df.loc[df.symbol == "GGG34", "underlying"] = "GGG"
    history = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31", "2020-02-29"]), "underlying": "GGG",
        "rec_mean": [3.8, 1.4], "n_analysts": [5, 6], "target_mean": [90.0, 130.0], "target_high": [100.0, 150.0],
        "target_low": [80.0, 110.0], "analyst_price": [100.0, 100.0], "rec_mean_3m": [3.5, 3.8],
    })
    jan = analysts.apply_us_history(df, "2020-02-15", history).set_index("symbol").loc["GGG34"]
    assert jan["rec_key"] == "underperform" and np.isclose(jan["upside"], -0.1)
    feb = analysts.apply_us_history(df, "2020-03-05", history).set_index("symbol").loc["GGG34"]
    assert feb["rec_key"] == "strong_buy" and feb["n_analysts"] == 6
    # Before any coverage, the snapshot's (future) consensus is not used
    old = analysts.apply_us_history(df, "2019-06-30", history).set_index("symbol").loc["GGG34"]
    assert pd.isna(old["rec_mean"]) and pd.isna(old["n_analysts"])
    # Legacy snapshots have no sector column
    legacy = analysts.apply_us_history(df.drop(columns="sector"), "2020-03-05", history).set_index("symbol")
    assert legacy.loc["GGG34", "rec_key"] == "strong_buy" and legacy["rel_upside"].notna().any()


def test_projections():
    df = analysts.add_features(make_universe()).set_index("symbol")
    prior = pd.Series(0.12, index=["AAA3", "CCC3", "HHH11", "III11", CDI])
    P, q, sd = analysts.projections(df, prior, relative=True, analyst_std=1.0)
    assert set(P.index) == {"AAA3", "CCC3", "FII carry"}
    assert np.isclose(q["AAA3"], 0.12 + df.loc["AAA3", "rel_upside"])
    assert (sd[["AAA3", "CCC3"]] >= 1.0).all()
    assert np.isclose(q["FII carry"], 0.12)  # median dividend yield of the FIIs
    assert np.isclose(P.loc["FII carry", ["HHH11", "III11"]].sum(), 1.0) and P.loc["FII carry", CDI] == 0
    _, q_abs, _ = analysts.projections(df, prior, relative=False, use_fii_yield=False)
    assert np.isclose(q_abs["AAA3"], 0.5)  # raw analyst projection
