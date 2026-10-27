"""CVaR portfolio optimization written as linear programs (Rockafellar & Uryasev, 2000).

Inputs are daily simple returns, one row per historical scenario. Risk is reported for a
holding period of `h` trading days: the tail dispersion of daily returns is scaled by
sqrt(h) and the expected drift by h. A CDI-only portfolio therefore has a negative CVaR
(even its bad tail is a gain). The drift in the risk measure can be the expected returns or
a single rate such as the CDI's, which keeps a CVaR budget from relying on return forecasts.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

TRADING_DAYS = 252

INFEASIBLE_MSG = (
    "The constraints cannot be satisfied together: relax the max weight per asset, "
    "the class limits or the risk/return target."
)


class InfeasibleError(ValueError):
    """The constraints cannot all be satisfied."""


def simple_returns(prices):
    """Daily simple returns. A portfolio return is the weighted sum of its assets' simple
    returns (log returns do not add up across assets), which is what the LP assumes."""
    return prices.pct_change().iloc[1:]


def scenario_probabilities(n_scenarios, halflife=None):
    """Weight of each historical scenario, oldest first. With a half-life, a day that is
    `halflife` trading days old counts half as much as the most recent one."""
    if not halflife:
        return np.full(n_scenarios, 1 / n_scenarios)
    age = np.arange(n_scenarios)[::-1]
    p = 0.5 ** (age / halflife)
    return p / p.sum()


def effective_sample_size(p):
    """Number of equally weighted scenarios that carry the same information as `p`."""
    return 1 / np.sum(np.square(p))


def weighted_cov(X, p):
    D = X - p @ X
    return (D * p[:, None]).T @ D


def tail_risk(losses, p, beta):
    """Probability-weighted VaR and CVaR (mean loss beyond the VaR) of a loss sample."""
    order = np.argsort(losses)[::-1]
    L, P = losses[order], p[order]
    tail = 1 - beta
    cum = np.cumsum(P)
    k = min(int(np.searchsorted(cum, tail - 1e-12)), len(L) - 1)
    above = cum[k - 1] if k > 0 else 0.0
    cvar = (np.dot(L[:k], P[:k]) + (tail - above) * L[k]) / tail
    return L[k], cvar


def expected_returns(R, p=None, method="shrinkage", rf=None, rf_now=None):
    """Expected daily return of each asset, modelled as the risk-free rate plus a historical
    spread over it: mu = rf_now + E[R - rf]. Fixed income that tracks the CDI is then priced
    at today's (or the projected) rate instead of the average rate of the window.

    - "historical": probability-weighted mean spread.
    - "shrinkage": James-Stein shrinkage of the spreads toward a common Sharpe ratio (the
      target of an asset is SR * its volatility). Sample means are noisy and an optimizer
      amplifies their errors (Michaud, 1989); shrinking them is the classic fix (Jorion,
      1986). In this project's walk-forward tests it did worse than the plain means, whose
      one- or two-year momentum it removes, while its target favors the riskiest assets.

    rf: daily risk-free returns aligned with R (e.g. the CDI); rf_now: the daily rate to use
    going forward (default: the last value of rf).
    """
    X = R.to_numpy(dtype=float)
    p = scenario_probabilities(len(X)) if p is None else np.asarray(p, dtype=float)
    rf_hist = np.zeros(len(X)) if rf is None else rf.reindex(R.index).fillna(0.0).to_numpy()
    rf_now = rf_hist[-1] if rf_now is None else rf_now
    E = X - rf_hist[:, None]
    spread = p @ E
    sigma = np.sqrt(p @ np.square(E - spread))
    if method == "shrinkage":
        risky = sigma > 1e-5
        k = int(risky.sum())
        if k > 3:
            sharpe = max(float(np.mean(spread[risky] / sigma[risky])), 0.0)
            target = sharpe * sigma
            se = sigma / np.sqrt(effective_sample_size(p))
            z2 = np.sum(np.square((spread[risky] - target[risky]) / se[risky]))
            keep = max(0.0, 1 - (k - 3) / z2)
            spread[risky] = target[risky] + keep * (spread[risky] - target[risky])
    elif method != "historical":
        raise ValueError(f"Unknown expected return method: {method}")
    return pd.Series(rf_now + spread, index=R.columns)


def black_litterman(mu_prior, R, P, q, sd, p=None, correlated_views=True):
    """Blends expected returns with views, e.g. analysts' 12-month projections.

    Each view says that portfolio P[k] (a row of weights over the assets; a single asset
    for an absolute view) returns q[k] a year, with uncertainty sd[k] (annual standard
    deviation). The prior's uncertainty is the sampling error of the historical mean
    (tau * Sigma, tau = 1 / effective sample size), so a view moves an asset more when its
    history is short or volatile and when the view is precise. Through Sigma a view also
    moves correlated assets (He & Litterman, 1999).

    With correlated_views, the errors of the views are as correlated as the view portfolios
    themselves: five bullish views on five semiconductor stocks then count about as much as
    one, instead of adding up as if they were independent evidence.
    """
    X = R.to_numpy(dtype=float)
    p = scenario_probabilities(len(X)) if p is None else np.asarray(p, dtype=float)
    cols = list(R.columns)
    prior = mu_prior.reindex(cols).to_numpy(dtype=float)
    if P is None or len(P) == 0:
        return pd.Series(prior, index=cols)
    P = P.reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)
    sigma = weighted_cov(X, p)
    tau_sigma = sigma / effective_sample_size(p)
    d = np.asarray(sd, dtype=float) / TRADING_DAYS
    if correlated_views:
        V = P @ sigma @ P.T
        scale = np.sqrt(np.clip(np.diag(V), 1e-18, None))
        omega = np.outer(d, d) * (V / np.outer(scale, scale))
    else:
        omega = np.diag(np.square(d))
    gap = np.asarray(q, dtype=float) / TRADING_DAYS - P @ prior
    posterior = prior + tau_sigma @ P.T @ np.linalg.solve(P @ tau_sigma @ P.T + omega, gap)
    return pd.Series(posterior, index=cols)


class CVaRProblem:
    """Mean-CVaR portfolio problems on historical scenarios.

    R: DataFrame (scenarios x assets) of daily simple returns; mu: Series of expected daily
    returns; p: scenario probabilities; beta: CVaR confidence level; h: holding period in
    trading days. Constraints: long-only, at most `max_weight` per asset (assets listed in
    `uncapped` are exempt, e.g. the CDI cash position), total weight of each group within
    group_bounds {group: (min, max)}, and positions below `min_weight` are removed by
    solving again without them. drift: the expected daily returns in the risk measure
    (default: mu), a Series or one rate for every asset.
    """

    def __init__(self, R, mu, p=None, beta=0.95, h=21, max_weight=1.0, min_weight=0.0,
                 groups=None, group_bounds=None, uncapped=(), drift=None):
        self.assets = list(R.columns)
        X = R.to_numpy(dtype=float)
        self.S, self.N = X.shape
        self.p = scenario_probabilities(self.S) if p is None else np.asarray(p, dtype=float)
        self.mu = mu.reindex(self.assets).to_numpy(dtype=float)
        if drift is None:
            self.drift = self.mu
        elif isinstance(drift, pd.Series):
            self.drift = drift.reindex(self.assets).to_numpy(dtype=float)
        else:
            self.drift = np.full(self.N, float(drift))
        # Demeaned scenarios carry the tail dispersion; the drift comes from mu
        self.X_dev = X - self.p @ X
        self.beta, self.h, self.min_weight = beta, h, min_weight
        self.ub = np.array([1.0 if a in uncapped else max_weight for a in self.assets])
        self.groups = pd.Series(groups).reindex(self.assets) if groups is not None else None
        self.group_rows = []
        for g, (lo, hi) in (group_bounds or {}).items():
            members = (self.groups == g).to_numpy() if self.groups is not None else np.zeros(self.N, bool)
            if members.any():
                self.group_rows.append((members.astype(float), lo, hi))
            elif lo > 0:
                raise InfeasibleError(f"Minimum of {lo:.0%} in '{g}', but no asset of that class is available.")
        if self.ub.sum() < 1 - 1e-9:
            raise InfeasibleError(
                f"With {self.N} assets and at most {max_weight:.0%} in each, the weights cannot add up to 100%."
            )
        if sum(lo for _, lo, _ in self.group_rows) > 1 + 1e-9:
            raise InfeasibleError("The class minimums add up to more than 100%.")
        # Scenario rows shared by every problem: u_s >= -X_dev[s].w - alpha
        self._scenarios = sparse.hstack(
            [sparse.csr_matrix(-self.X_dev), -np.ones((self.S, 1)), -sparse.identity(self.S)], format="csr"
        )

    # ---- linear program pieces; variables are [w (N), alpha, u (S)] ----
    def _risk_coefs(self):
        """h-day CVaR = sqrt(h) * (alpha + sum(p * u) / (1 - beta)) - h * drift.w"""
        root = np.sqrt(self.h)
        return np.concatenate([-self.h * self.drift, [root], root * self.p / (1 - self.beta)])

    def _return_coefs(self):
        return np.concatenate([self.mu, np.zeros(1 + self.S)])

    def _solve(self, c, ub, A_extra=(), b_extra=()):
        pad = np.zeros(1 + self.S)
        rows, rhs = [self._scenarios], [np.zeros(self.S)]
        for members, lo, hi in self.group_rows:
            rows += [sparse.csr_matrix(np.concatenate([members, pad])), sparse.csr_matrix(np.concatenate([-members, pad]))]
            rhs += [[hi], [-lo]]
        for a, b in zip(A_extra, b_extra):
            rows.append(sparse.csr_matrix(a))
            rhs.append([b])
        bounds = [(0, u) for u in ub] + [(None, None)] + [(0, None)] * self.S
        res = linprog(
            c, A_ub=sparse.vstack(rows, format="csr"), b_ub=np.concatenate(rhs),
            A_eq=np.concatenate([np.ones(self.N), pad])[None, :], b_eq=[1.0],
            bounds=bounds, method="highs",
        )
        if res.status != 0:
            raise InfeasibleError(INFEASIBLE_MSG)
        return res.x[: self.N]

    def _clean(self, solve, ub=None):
        """Solves, then drops positions below min_weight and solves again without them."""
        ub = self.ub.copy() if ub is None else ub.copy()
        w = solve(ub)
        for _ in range(10):
            small = (w > 1e-6) & (w < self.min_weight)
            if not small.any():
                break
            ub[small] = 0
            try:
                w = solve(ub)
            except InfeasibleError:
                break
        w = np.where(w > 1e-6, w, 0.0)
        return pd.Series(w / w.sum(), index=self.assets)

    # ---- portfolios ----
    def min_risk(self, min_return=None):
        """Lowest h-day CVaR, optionally with an expected daily return of at least min_return."""
        c = self._risk_coefs()
        if min_return is None:
            return self._clean(lambda ub: self._solve(c, ub))
        return self._clean(lambda ub: self._solve(c, ub, [-self._return_coefs()], [-min_return]))

    def max_return(self, max_risk=None):
        """Highest expected return with an h-day CVaR of at most max_risk (a fraction)."""
        c = -self._return_coefs()
        if max_risk is None:
            return self._clean(lambda ub: self._solve(c, ub))
        try:
            return self._clean(lambda ub: self._solve(c, ub, [self._risk_coefs()], [max_risk]))
        except InfeasibleError:
            lowest = self.stats(self.min_risk())["cvar"]
            if max_risk < lowest:
                raise InfeasibleError(f"No portfolio has a CVaR this low; the minimum is {lowest:.2%}.") from None
            raise

    def max_ratio(self, rf, exclude=()):
        """Tangency portfolio: the highest (expected return - rf) per unit of tail risk.
        The Charnes-Cooper change of variables y = t * w makes the ratio a linear program."""
        S, N = self.S, self.N
        excess = self.mu - rf
        ub = self.ub.copy()
        ub[[self.assets.index(a) for a in exclude if a in self.assets]] = 0

        def solve(ub):
            if not (excess[ub > 0] > 0).any():
                raise InfeasibleError("No asset is expected to beat the risk-free rate.")
            # variables are [y (N), alpha, u (S), t]
            c = np.concatenate([np.zeros(N), [1.0], self.p / (1 - self.beta), [0.0]])
            rows = [
                sparse.hstack([self._scenarios, sparse.csr_matrix((S, 1))]),
                sparse.hstack([sparse.identity(N), sparse.csr_matrix((N, 1 + S)), sparse.csr_matrix(-ub[:, None])]),
            ]
            rhs = [np.zeros(S), np.zeros(N)]
            for members, lo, hi in self.group_rows:
                pad = np.zeros(1 + S)
                rows += [sparse.csr_matrix(np.concatenate([members, pad, [-hi]])),
                         sparse.csr_matrix(np.concatenate([-members, pad, [lo]]))]
                rhs += [[0.0], [0.0]]
            A_eq = np.vstack([
                np.concatenate([excess, np.zeros(1 + S), [0.0]]),
                np.concatenate([np.ones(N), np.zeros(1 + S), [-1.0]]),
            ])
            bounds = [(0, None)] * N + [(None, None)] + [(0, None)] * S + [(0, None)]
            res = linprog(c, A_ub=sparse.vstack(rows, format="csr"), b_ub=np.concatenate(rhs),
                          A_eq=A_eq, b_eq=[1.0, 0.0], bounds=bounds, method="highs")
            if res.status != 0 or res.x[-1] <= 1e-12:
                raise InfeasibleError(INFEASIBLE_MSG)
            return res.x[:N] / res.x[-1]

        return self._clean(solve, ub)

    def frontier(self, n_points=30, progress=iter):
        """Efficient frontier from the minimum-CVaR to the maximum-return portfolio.
        Returns (stats, weights): one row per portfolio."""
        low, high = self.min_risk(), self.max_return()
        weights = [low]
        for target in progress(np.linspace(self.mu @ low, self.mu @ high, n_points)[1:]):
            try:
                weights.append(self.min_risk(min_return=target))
            except InfeasibleError:
                continue
        W = pd.DataFrame(weights).reset_index(drop=True)
        return pd.DataFrame([self.stats(w) for _, w in W.iterrows()]), W

    def stats(self, w):
        """Expected return, VaR and CVaR over h days (fractions), plus annualized figures."""
        w = w.reindex(self.assets).fillna(0.0).to_numpy() if isinstance(w, pd.Series) else np.asarray(w)
        var_d, cvar_d = tail_risk(-self.X_dev @ w, self.p, self.beta)
        ret, drift = float(self.mu @ w), float(self.drift @ w)
        root = np.sqrt(self.h)
        return {
            "exp_return": self.h * ret,
            "var": root * var_d - self.h * drift,
            "cvar": root * cvar_d - self.h * drift,
            "annual_return": TRADING_DAYS * ret,
            "annual_vol": float(np.sqrt(w @ weighted_cov(self.X_dev, self.p) @ w * TRADING_DAYS)),
            "n_assets": int((w > 1e-6).sum()),
        }

    def risk_contributions(self, w):
        """Share of the portfolio's tail risk due to each asset (Euler allocation of the
        dispersion CVaR): its weight times its average loss in the portfolio's worst
        scenarios. The shares add up to 1; a negative share means the asset hedges the tail."""
        w = w.reindex(self.assets).fillna(0.0).to_numpy() if isinstance(w, pd.Series) else np.asarray(w)
        losses = -self.X_dev @ w
        order = np.argsort(losses)[::-1]
        P = self.p[order]
        tail = 1 - self.beta
        in_tail = np.clip(tail - (np.cumsum(P) - P), 0.0, P)  # probability of each scenario inside the tail
        contrib = w * ((in_tail / tail) @ -self.X_dev[order])
        return pd.Series(contrib / contrib.sum(), index=self.assets)

    def asset_stats(self):
        """stats() of each asset held alone."""
        eye = np.eye(self.N)
        return pd.DataFrame([self.stats(eye[i]) for i in range(self.N)], index=self.assets)
