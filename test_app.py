"""Integration tests with live data (Yahoo Finance, brapi.dev, Central Bank of Brazil).

    pytest -m network
"""
from datetime import date, timedelta

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import analysts
import backtesting
import market_data
import portfolio
import universe
from market_data import CDI

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def selected():
    df, _ = universe.read_universe(date.today())
    out = analysts.apply_filters(df, analysts.FilterRules(max_per_class=8))
    return out[out["status"] == "selected"].set_index("symbol")


@pytest.fixture(scope="module")
def prices(selected):
    end = date.today()
    history, _ = market_data.collect_prices(list(selected.index) + [CDI], end - timedelta(days=730), end)
    return history


def test_universe_snapshot_has_every_class():
    df, _ = universe.read_universe(date.today())
    assert len(df) > 100
    assert {"Stocks", "BDRs", "FIIs", "Fixed income"} <= set(df["asset_class"])
    assert df["symbol"].notna().all()


def test_prices_are_clean(prices):
    assert not prices.isna().any().any()
    assert (prices.pct_change().abs().max() < 1.0).all()
    assert CDI in prices


def test_central_bank_data():
    assert 0.0 < market_data.expected_cdi() < 0.5
    rates = market_data.fetch_cdi(date.today() - timedelta(days=30), date.today())
    assert len(rates) > 10 and (rates > 0).all()


def test_optimization_pipeline(prices, selected):
    for settings in [portfolio.ModelSettings(), portfolio.ModelSettings(projections=True, objective="max_ratio")]:
        problem, details = portfolio.build_problem(prices, selected, settings)
        w = portfolio.solve(problem, settings, details["rf"])
        assert np.isclose(w.sum(), 1) and (w >= 0).all()
        assert (w.drop(CDI, errors="ignore") <= settings.max_weight + 1e-6).all()
    stats, _ = problem.frontier(n_points=10)
    assert len(stats) >= 8


def test_app_pages_run():
    app = AppTest.from_file("app.py", default_timeout=600).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Generate Optimization").click().run()
    assert not app.exception and "po" in app.session_state
    results = AppTest.from_file("pages/Results.py", default_timeout=300)
    results.session_state["po"] = app.session_state["po"]
    assert not results.run().exception


def test_backtest_runs():
    stocks = analysts.FilterRules(classes=("Stocks", "Fixed income"), max_per_class=8)
    strategies = [backtesting.Strategy("model", stocks),
                  backtesting.Strategy("1/N", stocks, kind="equal")]
    out = backtesting.run_suite(strategies, date.today() - timedelta(days=200), date.today(),
                                lookback=126, log=lambda *_: None)
    assert {"model", "1/N", "CDI"} <= set(out["metrics"].index)
    assert out["values"]["model"].notna().all()
