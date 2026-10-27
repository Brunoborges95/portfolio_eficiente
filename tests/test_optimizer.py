import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

import optimizer as opt


@pytest.fixture()
def returns():
    """Two years of daily returns for 8 risky assets plus a CDI-like riskless one."""
    rng = np.random.default_rng(42)
    n, k = 504, 8
    vols = np.linspace(0.008, 0.03, k)
    common = rng.standard_t(4, n)[:, None] * 0.006
    X = 0.0004 + common * np.linspace(0.5, 1.5, k) + rng.standard_t(4, (n, k)) * vols
    R = pd.DataFrame(X, columns=[f"A{i}" for i in range(k)],
                     index=pd.bdate_range("2023-01-02", periods=n))
    R["CDI"] = 0.0005
    return R


def test_scenario_probabilities_halflife():
    p = opt.scenario_probabilities(100, halflife=10)
    assert np.isclose(p.sum(), 1)
    assert np.isclose(p[-11] / p[-1], 0.5)
    assert np.allclose(opt.scenario_probabilities(5), 0.2)


def test_tail_risk_matches_mean_of_worst_losses():
    losses = np.arange(100, dtype=float)
    var, cvar = opt.tail_risk(losses, np.full(100, 0.01), 0.95)
    assert var == 95
    assert np.isclose(cvar, np.mean([95, 96, 97, 98, 99]))


def test_lp_risk_equals_empirical_cvar(returns):
    mu = opt.expected_returns(returns, method="historical", rf=returns["CDI"])
    prob = opt.CVaRProblem(returns, mu, h=1)
    w = prob.min_risk()
    # Solve the same LP directly to read its objective value
    res = linprog(prob._risk_coefs(), A_ub=prob._scenarios, b_ub=np.zeros(prob.S),
                  A_eq=np.concatenate([np.ones(prob.N), np.zeros(1 + prob.S)])[None, :], b_eq=[1],
                  bounds=[(0, 1)] * prob.N + [(None, None)] + [(0, None)] * prob.S, method="highs")
    assert np.isclose(prob.stats(w)["cvar"], res.fun, atol=1e-9)


def test_cvar_can_be_negative_when_every_scenario_is_a_gain():
    """The VaR variable must be free: with only gains, VaR and CVaR are negative."""
    R = pd.DataFrame({"A": np.linspace(0.001, 0.003, 50), "B": np.linspace(0.002, 0.001, 50)})
    mu = R.mean()
    prob = opt.CVaRProblem(R, mu, h=1)
    s = prob.stats(prob.min_risk())
    assert s["cvar"] < 0


def test_constraints_are_respected(returns):
    mu = opt.expected_returns(returns, rf=returns["CDI"])
    groups = pd.Series({**{f"A{i}": "Stocks" for i in range(4)},
                        **{f"A{i}": "BDRs" for i in range(4, 8)}, "CDI": "Fixed income"})
    prob = opt.CVaRProblem(returns, mu, max_weight=0.3, min_weight=0.03, groups=groups,
                           group_bounds={"BDRs": (0.2, 0.4), "Fixed income": (0.0, 0.5)},
                           uncapped=("CDI",))
    for w in [prob.min_risk(), prob.max_return(), prob.max_ratio(0.0005, exclude=("CDI",))]:
        assert np.isclose(w.sum(), 1)
        assert (w >= 0).all()
        assert (w.drop("CDI") <= 0.3 + 1e-7).all()
        assert w["CDI"] <= 0.5 + 1e-7
        bdr = w[[f"A{i}" for i in range(4, 8)]].sum()
        assert 0.2 - 1e-7 <= bdr <= 0.4 + 1e-7
        held = w[w > 0]
        assert (held >= 0.03 - 1e-9).all()


def test_frontier_is_monotonic(returns):
    mu = opt.expected_returns(returns, method="historical", rf=returns["CDI"])
    stats, W = opt.CVaRProblem(returns, mu).frontier(n_points=12)
    assert len(stats) >= 10
    assert (np.diff(stats["exp_return"]) > -1e-9).all()
    assert (np.diff(stats["cvar"]) > -1e-7).all()
    assert np.allclose(W.sum(axis=1), 1)


def test_max_return_respects_risk_budget(returns):
    mu = opt.expected_returns(returns, method="historical", rf=returns["CDI"])
    prob = opt.CVaRProblem(returns, mu)
    low = prob.stats(prob.min_risk())
    budget = low["cvar"] + 0.03
    s = prob.stats(prob.max_return(max_risk=budget))
    assert s["cvar"] <= budget + 1e-7
    assert s["exp_return"] >= low["exp_return"] - 1e-9
    with pytest.raises(opt.InfeasibleError):
        prob.max_return(max_risk=low["cvar"] - 0.05)


def test_fixed_drift_keeps_forecasts_out_of_the_risk(returns):
    mu = opt.expected_returns(returns, method="historical", rf=returns["CDI"])
    optimistic = mu + 0.002  # forecasts that would loosen the budget
    cdi = opt.CVaRProblem(returns, optimistic, drift=0.0005)
    w = cdi.min_risk()
    assert np.isclose(cdi.stats(w)["cvar"], opt.CVaRProblem(returns, mu, drift=0.0005).stats(w)["cvar"])
    budget = cdi.stats(w)["cvar"] + 0.01
    credited = opt.CVaRProblem(returns, optimistic)  # drift = the forecasts themselves
    assert cdi.stats(cdi.max_return(budget))["cvar"] <= budget + 1e-7
    # Measured without the optimistic drift, the portfolio that relied on it breaks the budget
    assert cdi.stats(credited.max_return(budget))["cvar"] > budget


def test_tangency_beats_other_portfolios(returns):
    mu = opt.expected_returns(returns, method="historical", rf=returns["CDI"])
    rf = 0.0005
    prob = opt.CVaRProblem(returns, mu, h=1)

    def ratio(w):
        w = np.asarray(w)
        _, cvar = opt.tail_risk(-prob.X_dev @ w, prob.p, prob.beta)
        return (prob.mu @ w - rf) / cvar

    best = ratio(prob.max_ratio(rf, exclude=("CDI",)).to_numpy())
    rng = np.random.default_rng(0)
    for _ in range(300):
        w = np.append(rng.dirichlet(np.ones(8)), 0.0)
        assert ratio(w) <= best + 1e-9


def test_risk_contributions_add_up_to_the_cvar(returns):
    mu = opt.expected_returns(returns, rf=returns["CDI"])
    prob = opt.CVaRProblem(returns, mu, h=1)
    w = prob.max_return(max_risk=prob.stats(prob.min_risk())["cvar"] + 0.01)
    shares = prob.risk_contributions(w)
    assert np.isclose(shares.sum(), 1.0)
    assert (shares[w == 0] == 0).all()
    # Unnormalized, the contributions are the dispersion CVaR itself
    _, cvar = opt.tail_risk(-prob.X_dev @ w.to_numpy(), prob.p, prob.beta)
    losses = -prob.X_dev @ w.to_numpy()
    order = np.argsort(losses)[::-1]
    in_tail = np.clip(0.05 - (np.cumsum(prob.p[order]) - prob.p[order]), 0, prob.p[order])
    assert np.isclose((in_tail @ losses[order]) / 0.05, cvar)


def test_expected_returns(returns):
    hist = opt.expected_returns(returns, method="historical", rf=returns["CDI"])
    shrunk = opt.expected_returns(returns, method="shrinkage", rf=returns["CDI"])
    assert np.isclose(hist["CDI"], 0.0005) and np.isclose(shrunk["CDI"], 0.0005)
    assert np.allclose(hist.drop("CDI"), returns.drop(columns="CDI").mean())
    assert shrunk.drop("CDI").std() < hist.drop("CDI").std()
    # A projected risk-free rate shifts every asset by the same amount
    proj = opt.expected_returns(returns, method="historical", rf=returns["CDI"], rf_now=0.0004)
    assert np.allclose(hist - proj, 0.0001)


def test_black_litterman(returns):
    prior = opt.expected_returns(returns, rf=returns["CDI"])
    assert np.allclose(opt.black_litterman(prior, returns, None, [], []), prior)
    P = pd.DataFrame({"A3": [1.0]}, index=["view"])
    sure = opt.black_litterman(prior, returns, P, [0.40], [1e-4])
    vague = opt.black_litterman(prior, returns, P, [0.40], [10.0])
    assert np.isclose(sure["A3"] * 252, 0.40, atol=1e-3)
    assert abs(vague["A3"] - prior["A3"]) < 0.01 * abs(sure["A3"] - prior["A3"])
    # Correlated assets move in the same direction as the view
    assert (sure.drop(["A3", "CDI"]) - prior.drop(["A3", "CDI"])).mean() > 0
    # A basket view sets the basket's mean, not each member's
    basket = pd.DataFrame({"A0": [0.5], "A1": [0.5]}, index=["basket"])
    post = opt.black_litterman(prior, returns, basket, [0.30], [1e-4])
    assert np.isclose((post["A0"] + post["A1"]) / 2 * 252, 0.30, atol=1e-3)


def test_correlated_views_do_not_add_up(returns):
    """Identical views on highly correlated assets: with correlated view errors, each asset
    moves about as much as it would with its view alone."""
    R = returns.copy()
    rng = np.random.default_rng(3)
    for c in ["B1", "B2", "B3"]:
        R[c] = R["A5"] + rng.normal(0, 0.002, len(R))  # near copies of A5
    prior = opt.expected_returns(R, rf=R["CDI"])
    one = pd.DataFrame({"B1": [1.0]}, index=["v"])
    three = pd.DataFrame(np.eye(3), index=["v1", "v2", "v3"], columns=["B1", "B2", "B3"])
    alone = opt.black_litterman(prior, R, one, [prior["B1"] * 252 + 0.3], [0.5])["B1"] - prior["B1"]
    q3 = [prior[c] * 252 + 0.3 for c in ["B1", "B2", "B3"]]
    together = opt.black_litterman(prior, R, three, q3, [0.5] * 3)["B1"] - prior["B1"]
    independent = opt.black_litterman(prior, R, three, q3, [0.5] * 3, correlated_views=False)["B1"] - prior["B1"]
    # Independent errors would triple the views' precision: a weight of a/(a + w/3) instead of a/(a + w)
    assert together < 1.15 * alone
    assert independent > 1.4 * alone


def test_infeasible_settings_raise(returns):
    mu = returns.mean()
    with pytest.raises(opt.InfeasibleError):
        opt.CVaRProblem(returns, mu, max_weight=0.05)
    groups = pd.Series({c: "Stocks" for c in returns.columns})
    with pytest.raises(opt.InfeasibleError):
        opt.CVaRProblem(returns, mu, groups=groups, group_bounds={"Crypto": (0.1, 0.2)})
