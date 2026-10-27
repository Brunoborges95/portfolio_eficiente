import numpy as np
import pandas as pd
import pytest

import portfolio
from market_data import CDI


@pytest.fixture()
def inputs():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0.0006, [0.01, 0.02, 0.03], (300, 3)), columns=["AAA3", "BBB3", "CCC3"],
                           index=pd.bdate_range("2024-01-01", periods=300))
    returns[CDI] = 0.0004
    assets = pd.DataFrame({"asset_class": "Stocks"}, index=["AAA3", "BBB3", "CCC3"])
    return (1 + returns).cumprod(), assets


def test_risk_drift(inputs):
    prices, assets = inputs
    problem, details = portfolio.build_problem(prices, assets, portfolio.ModelSettings(max_weight=1.0))
    assert np.allclose(problem.drift, details["rf"])  # the CDI's rate for every asset
    settings = portfolio.ModelSettings(max_weight=1.0, risk_drift="expected")
    problem, details = portfolio.build_problem(prices, assets, settings)
    assert np.allclose(problem.drift, details["mu"][problem.assets])
    with pytest.raises(ValueError):
        portfolio.build_problem(prices, assets, portfolio.ModelSettings(max_weight=1.0, risk_drift="forecast"))


def test_budget_below_the_least_risky_portfolio_falls_back_to_it(inputs):
    prices, assets = inputs
    settings = portfolio.ModelSettings(risk_budget=-0.5, max_weight=1.0)
    problem, details = portfolio.build_problem(prices, assets, settings)
    assert np.allclose(portfolio.solve(problem, settings, details["rf"]), problem.min_risk())
