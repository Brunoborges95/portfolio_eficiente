import numpy as np
import pandas as pd
import pytest

import backtesting


@pytest.fixture()
def prices():
    days = pd.bdate_range("2024-01-01", periods=60)
    return pd.DataFrame({"UP": 1.01 ** np.arange(60), "FLAT": np.ones(60)}, index=days)


def test_daily_rebalance_matches_closed_form(prices):
    half = lambda past, when: pd.Series(0.5, index=["UP", "FLAT"])
    res = backtesting.walk_forward(prices, half, prices.index[10], lookback=10, rebalance=1, cost=0.0)
    n = len(res["value"]) - 1
    assert np.isclose(res["value"].iloc[-1], 1.005 ** n)


def test_weights_drift_between_rebalances(prices):
    half = lambda past, when: pd.Series(0.5, index=["UP", "FLAT"])
    res = backtesting.walk_forward(prices, half, prices.index[10], lookback=10, rebalance=100, cost=0.0)
    k = len(res["value"]) - 1
    assert np.isclose(res["value"].iloc[-1], 0.5 * 1.01 ** k + 0.5)  # buy and hold


def test_no_look_ahead_and_costs(prices):
    seen = []

    def spy(past, when):
        assert past.index[-1] == when  # the decision only sees prices up to its own close
        seen.append(when)
        return pd.Series({"FLAT": 1.0})

    res = backtesting.walk_forward(prices, spy, prices.index[10], lookback=10, rebalance=5, cost=0.01)
    blocks = res["value"].index
    assert all(d < blocks[blocks > d][0] for d in seen)
    assert np.isclose(res["value"].iloc[-1], 0.99)  # one purchase, then no trades
    assert np.isclose(res["turnover"].iloc[0], 1.0) and np.allclose(res["turnover"].iloc[1:], 0.0)


def test_cash_when_nothing_is_investable(monkeypatch, prices):
    """A strategy whose filters select nothing on a date holds the CDI instead of failing."""
    p = prices.assign(CDI=1.0005 ** np.arange(len(prices)))
    empty = pd.DataFrame({"asset_class": []}, index=pd.Index([], name="symbol"))
    monkeypatch.setattr(backtesting, "selected_assets", lambda *a, **k: (empty, "2024-01-01"))
    strategy = backtesting.Strategy("none", kind="equal", pit_liquidity=False)
    w = backtesting._weight_fn(strategy, {}, {})(p.iloc[:20], p.index[19])
    assert w.to_dict() == {"CDI": 1.0}


def test_performance_metrics():
    days = pd.bdate_range("2024-01-01", periods=5)
    value = pd.Series([1.0, 1.1, 0.99, 1.089, 1.2], index=days)
    m = backtesting.performance(value)
    assert np.isclose(m["Total return"], 0.2)
    assert np.isclose(m["Max drawdown"], 0.99 / 1.1 - 1)


def test_legacy_model_runs(prices):
    rng = np.random.default_rng(1)
    noisy = pd.DataFrame(np.exp(np.cumsum(rng.normal(0.0005, 0.01, (120, 4)), axis=0)),
                         columns=list("ABCD"), index=pd.bdate_range("2024-01-01", periods=120))
    w = backtesting.legacy_weights(noisy, risk_value=5.0)
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all()
