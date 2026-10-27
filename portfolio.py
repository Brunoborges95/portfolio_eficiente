"""Portfolio construction: from prices, universe rows and model settings to weights.

Shared by the app and the backtests, so both run exactly the same model.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import analysts
import market_data
import optimizer as opt
from market_data import CDI, TESOURO_PREFIX

OBJECTIVES = {
    "risk_budget": "Maximum return for a CVaR budget",
    "min_risk": "Minimum CVaR",
    "max_ratio": "Best return per unit of risk (tangency)",
}
RETURN_METHODS = {
    "historical": "Historical mean (momentum)",
    "shrinkage": "Shrinkage toward a common Sharpe ratio (James-Stein)",
}
RISK_DRIFTS = {
    "cdi": "The CDI's return only (conservative)",
    "expected": "Each asset's expected return",
}


@dataclass
class ModelSettings:
    """Model choices. Risk is the CVaR at `confidence` over `horizon` trading days."""
    confidence: float = 0.95
    horizon: int = 21
    # In the walk-forward tests the historical means did better than the shrunk ones: over one
    # or two years they carry momentum, which shrinkage toward a common Sharpe ratio removes
    return_method: str = "historical"
    # Drift in the CVaR: with "cdi" no asset is credited with more than the CDI's return, so the
    # budget does not rest on return forecasts; crediting the historical means ("expected") let
    # recent winners loosen it, and the realized CVaR reached about twice the budget
    risk_drift: str = "cdi"
    halflife: int | None = None  # scenario half-life in trading days (None: equal weights)
    max_weight: float = 0.20  # per asset; the CDI is exempt
    min_weight: float = 0.01  # smaller positions are removed
    class_bounds: dict = field(default_factory=dict)  # {asset class: (min, max)}
    # The tangency portfolio maximizes a ratio of noisy estimates: in the backtests it had the
    # lowest Sharpe ratio of the multi-asset portfolios, while the CVaR budgets held
    objective: str = "risk_budget"
    risk_budget: float | None = 0.05  # h-day CVaR (fraction) for objective="risk_budget"
    # Projections: blend history with forward-looking information (Black-Litterman)
    projections: bool = False
    use_analysts: bool = True  # Stocks/BDRs: analysts' target price + dividend yield
    use_fii_yield: bool = True  # FIIs: dividend yield (carry)
    use_bond_carry: bool = True  # Tesouro bonds: their yield; IPCA+ also the Focus expected inflation
    use_focus: bool = True  # CDI: Focus survey's expected Selic path
    use_fundamentals: bool = False  # stocks: tilt by the CVM factor score, alpha = IC x volatility x score
    # The notebook measured a rank IC of 0.085 a month and 0.16 a year (2016-2026) on today's
    # listed stocks, which survivorship inflates: the default takes a third of the annual one
    fundamental_ic: float = 0.05
    relative_projections: bool = True  # only the part above/below peers counts
    analyst_std: float = 1.0  # uncertainty of an analyst projection (annual); see analysts.projections
    fii_std: float = 0.10  # uncertainty of the FII carry projection (annual)
    min_analysts: int = 3


def build_problem(prices, assets, settings, as_of=None):
    """CVaR problem for the assets in `assets` (universe rows indexed by symbol).

    `prices` must contain the CDI column, which is the risk-free rate; the CDI is investable
    only if it is also in `assets`. `as_of` dates the Focus projection (default: last price).
    Returns (problem, details) with mu and rf (daily) and the projections used.
    """
    R_all = opt.simple_returns(prices)
    rf_hist = R_all[CDI]
    investable = [c for c in R_all.columns if c in assets.index]
    R = R_all[investable]
    p = opt.scenario_probabilities(len(R), settings.halflife)
    rf_now, focus = rf_hist.iloc[-1], None
    if settings.projections and settings.use_focus:
        focus = market_data.expected_cdi(as_of if as_of is not None else prices.index[-1])
        if focus is not None:
            rf_now = market_data.annual_to_daily(focus)
    mu = opt.expected_returns(R, p, settings.return_method, rf=rf_hist, rf_now=rf_now)
    prior, views, blocks = mu.copy(), None, []
    if settings.projections and (settings.use_analysts or settings.use_fii_yield):
        blocks.append(analysts.projections(
            assets.loc[investable], mu * opt.TRADING_DAYS, relative=settings.relative_projections,
            analyst_std=settings.analyst_std, fii_std=settings.fii_std,
            min_analysts=settings.min_analysts, use_analysts=settings.use_analysts,
            use_fii_yield=settings.use_fii_yield,
        ))
    bonds = [c for c in investable if c.startswith(TESOURO_PREFIX)]
    if settings.projections and settings.use_bond_carry and bonds:
        carry = market_data.bond_carry(bonds, as_of if as_of is not None else prices.index[-1])
        if len(carry):
            # Uncertainty of a year's return: the bond's own volatility (yields do move)
            P = pd.DataFrame(np.eye(len(carry)), index=carry.index, columns=carry.index)
            blocks.append((P, carry, R[carry.index].std() * np.sqrt(opt.TRADING_DAYS)))
    if blocks:
        P = pd.concat([b[0] for b in blocks]).reindex(columns=investable).fillna(0.0)
        q, sd = pd.concat([b[1] for b in blocks]), pd.concat([b[2] for b in blocks])
        mu = opt.black_litterman(mu, R, P, q, sd, p)
        views = {"P": P, "q": q, "sd": sd}
    alpha = None
    if settings.projections and settings.use_fundamentals and "fundamental_score" in assets:
        # Grinold's rule: a signal with rank IC `ic` moves the expected return by about
        # ic x volatility x the signal's z-score
        z = pd.to_numeric(assets.loc[investable, "fundamental_score"], errors="coerce")
        alpha = (settings.fundamental_ic * R.std() * np.sqrt(opt.TRADING_DAYS) * z).fillna(0.0)
        mu = mu + alpha / opt.TRADING_DAYS
    if CDI in mu:
        mu[CDI] = rf_now
    if settings.risk_drift not in RISK_DRIFTS:
        raise ValueError(f"Unknown risk drift: {settings.risk_drift}")
    problem = opt.CVaRProblem(
        R, mu, p, beta=settings.confidence, h=settings.horizon, max_weight=settings.max_weight,
        min_weight=settings.min_weight, groups=assets.loc[investable, "asset_class"],
        group_bounds=settings.class_bounds, uncapped=(CDI,),
        drift=rf_now if settings.risk_drift == "cdi" else None,
    )
    return problem, {"mu": mu, "prior": prior, "rf": rf_now, "views": views, "focus_cdi": focus,
                     "fundamental_alpha": alpha}


def solve(problem, settings, rf):
    """Weights for the settings' objective. The tangency portfolio excludes the CDI (it is
    the risk-free asset); if nothing beats the risk-free rate, the minimum-CVaR one is used,
    as it is when no portfolio fits the CVaR budget (e.g. stocks only, in a crisis)."""
    if settings.objective == "min_risk":
        return problem.min_risk()
    if settings.objective == "risk_budget":
        try:
            return problem.max_return(settings.risk_budget)
        except opt.InfeasibleError:
            return problem.min_risk()
    for exclude in ((CDI,), ()):
        try:
            return problem.max_ratio(rf, exclude=exclude)
        except opt.InfeasibleError:
            continue
    return problem.min_risk()
