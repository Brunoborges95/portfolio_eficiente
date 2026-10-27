"""Walk-forward backtests.

Every `rebalance` trading days the portfolio is rebuilt with data available at that moment
only: the last `lookback` days of prices, the latest universe snapshot published on or
before that day (liquidity, analyst consensus) and that day's Focus survey. The portfolio
is then held, drifting with prices, until the next rebalance, and each trade pays `cost`
per unit of turnover. Rebalances older than the first snapshot have no analyst data from
their time; they use the earliest snapshot and are reported as look-ahead.
"""
import functools
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import linprog

import analysts
import market_data
import optimizer as opt
import portfolio
import universe
from market_data import CDI

TRADING_DAYS = 252


@dataclass
class Strategy:
    """A backtest line. kind: "model" (portfolio.ModelSettings), "equal" (1/N of the
    selected risky assets) or "legacy" (the original model at CVaR level legacy_risk, %).
    snapshot pins the universe to one snapshot date for every rebalance (point in time only
    for rebalances after it); by default each rebalance uses the latest snapshot before it."""
    name: str
    rules: analysts.FilterRules = field(default_factory=analysts.FilterRules)
    settings: portfolio.ModelSettings = field(default_factory=portfolio.ModelSettings)
    kind: str = "model"
    legacy_risk: float = 10.0
    snapshot: str | None = None
    pit_liquidity: bool = True  # liquidity filters measured at each rebalance date


def walk_forward(prices, weight_fn, start, end=None, lookback=252, rebalance=21, cost=0.001):
    """Simulates weight_fn(past_prices, date) -> weights, called at the close of each
    rebalance day with the previous `lookback` days of prices.

    Returns a dict with the daily portfolio value (1 on the first formation day), the target
    weights and turnover of each rebalance, and the rebalances that could not be solved
    (the portfolio is then kept as it was).
    """
    prices = prices.loc[:end] if end is not None else prices
    returns = prices.ffill().pct_change(fill_method=None).fillna(0.0)  # not trading (yet or anymore): no price move
    dates = prices.index
    first = max(int(dates.searchsorted(pd.Timestamp(start))), lookback)
    if first >= len(dates):
        raise ValueError("No trading days left after the lookback window; move the start date.")
    value = 1.0
    held = pd.Series(0.0, index=prices.columns)
    curve = {dates[first - 1]: 1.0}
    targets, turnover, errors = {}, {}, {}
    for i0 in range(first, len(dates), rebalance):
        when = dates[i0 - 1]
        try:
            target = weight_fn(prices.iloc[i0 - lookback: i0], when).reindex(prices.columns).fillna(0.0)
        except (opt.InfeasibleError, np.linalg.LinAlgError) as e:
            errors[when] = str(e)
            if held.sum() == 0:
                if CDI not in held:
                    raise
                held[CDI] = 1.0
            target = held
        trade = float((target - held).abs().sum())
        value *= 1 - cost * trade
        growth = (1 + returns.iloc[i0: i0 + rebalance]).cumprod()
        path = value * (growth @ target)
        curve.update(path.to_dict())
        drifted = growth.iloc[-1] * target
        held = drifted / drifted.sum()
        value = float(path.iloc[-1])
        targets[when], turnover[when] = target, trade
    weights = pd.DataFrame(targets).T
    return {
        "value": pd.Series(curve).sort_index(),
        "weights": weights.loc[:, (weights > 0).any()],
        "turnover": pd.Series(turnover),
        "errors": errors,
    }


def performance(value, rf=None, beta=0.95, h=21):
    """Summary statistics of a daily value series; rf: daily risk-free returns (CDI).
    The realized CVaR uses the model's own h-day scaling, to compare with the ex-ante one."""
    r = value.pct_change().dropna()
    rf_r = rf.reindex(r.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=r.index)
    total = value.iloc[-1] / value.iloc[0] - 1
    years = len(r) / TRADING_DAYS
    uniform = np.full(len(r), 1 / len(r))
    _, cvar = opt.tail_risk(-r.to_numpy(), uniform, beta)
    _, cvar_dev = opt.tail_risk(-(r - r.mean()).to_numpy(), uniform, beta)
    monthly = (1 + r).resample("ME").prod() - 1
    monthly_rf = (1 + rf_r).resample("ME").prod() - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {
        "Total return": total,
        "Annual return": (1 + total) ** (1 / years) - 1,
        "Annual volatility": vol,
        "Sharpe (vs CDI)": (r - rf_r).mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
        "Max drawdown": (value / value.cummax() - 1).min(),
        f"Daily CVaR {beta:.0%}": cvar,
        "Realized CVaR (month)": np.sqrt(h) * cvar_dev - h * r.mean(),
        "Months beating CDI": (monthly > monthly_rf + 1e-12).mean(),
    }


@functools.lru_cache(maxsize=64)
def _snapshot(day):
    return universe.read_universe(day)


@functools.lru_cache(maxsize=1)
def _us_history():
    return analysts.load_us_history()


@functools.lru_cache(maxsize=2048)
def selected_assets(day, rules, as_of=None):
    """Universe rows selected by `rules` in the snapshot in force on `day` (indexed by
    symbol), and the date of that snapshot. With `as_of`, the BDRs' analyst data is the
    consensus known on that date (analysts.apply_us_history), not the snapshot's."""
    df, snapshot = _snapshot(day)
    history = _us_history()
    if as_of is not None and history is not None:
        df = analysts.apply_us_history(df, as_of, history)
    df = analysts.apply_filters(df, rules)
    return df[df["status"] == "selected"].set_index("symbol"), snapshot


def candidate_rules(rules):
    """The same classes without liquidity, ranking or analyst cuts: every asset a strategy
    could hold on some date (for the price download)."""
    return replace(rules, min_adtv=0.0, max_per_class=None, ratings=tuple(analysts.RATINGS), include_uncovered=True,
                   min_upside=None, max_downgrade=None, max_dispersion=None)


def legacy_weights(prices, risk_value, a=0.95, h=21, n_points=50):
    """The original model of this project, kept for comparison: log returns + 1, CVaR LP
    with the VaR variable bounded at zero, no position limits, a 50-point frontier at fixed
    returns, and the frontier point whose CVaR (% over h days, sqrt-of-time) is closest to
    risk_value."""
    data = np.where(prices.to_numpy().T <= 0, 1e-07, prices.to_numpy().T)
    Returns = (np.log(data[:, 1:] / data[:, :-1]) + 1).T
    ExpR = Returns.mean(axis=0, keepdims=True)
    N, nS = ExpR.shape[1], Returns.shape[0]
    meanR = np.linspace(ExpR.min(), ExpR.max(), n_points)
    f = np.vstack((np.zeros((N, 1)), 1, 1 / ((1 - a) * nS) * np.ones((nS, 1)))).ravel()
    A = np.hstack([np.ones((nS, N)) - Returns, -np.ones((nS, 1)), -np.eye(nS)])
    Aeq = np.vstack([np.hstack([ExpR, [[0]], np.zeros((1, nS))]), np.hstack([np.ones((1, N)), [[0]], np.zeros((1, nS))])])
    linMap = np.zeros((nS + N + 1, n_points))
    for i, target in enumerate(meanR):
        res = linprog(f, A, np.zeros(nS), Aeq, [target, 1], [(0, None)] * (nS + N + 1), method="highs")
        if res.status == 0:
            linMap[:, i] = res.x
    cvar = 100 * (linMap[N + 1:].sum(axis=0) / ((1 - a) * nS) + linMap[N]) * np.sqrt(h)
    start = int(np.argmin(cvar))
    i = start + int(np.argmin(abs(cvar[start:] - risk_value)))
    w = pd.Series(linMap[:N, i], index=prices.columns)
    return w / w.sum()


def liquid_assets(candidates, traded, when, rules, window=63):
    """Applies the liquidity floor and the per-class ranking cut of `rules` with the average
    traded value of the `window` days up to `when`, instead of the snapshot's: an asset that
    is liquid today because it rallied was often illiquid before (point-in-time liquidity)."""
    adtv = traded.loc[:when].tail(window).mean().reindex(candidates.index)
    exempt = analysts.off_exchange(candidates.index).to_numpy()
    listed = candidates[~exempt].assign(adtv=adtv[~exempt])
    listed = listed[listed["adtv"] >= rules.min_adtv]
    if rules.max_per_class:
        keys = [k for k in analysts.RANK_KEYS[rules.rank_by] if k in listed]
        listed = (listed.sort_values(keys, ascending=False, na_position="last")
                  .groupby("asset_class", group_keys=False).head(rules.max_per_class))
    return pd.concat([listed, candidates[exempt]])


class PointInTimeFundamentals:
    """Fundamental factor scores of stocks on each rebalance date, from the CVM filings
    published by then and prices up to that day (fundamentals.indicators)."""

    def __init__(self, stock_rows, start, end):
        import fundamentals
        self.fundamentals = fundamentals
        self.companies, self.tickers = fundamentals.companies_for(stock_rows[["symbol", "sector", "adtv"]])
        self.panel = fundamentals.filings()
        self.history = fundamentals.annual_history(self.panel)
        self.quotes = market_data.download_quotes(fundamentals.quote_symbols(self.companies), start, end)
        self._cache = {}

    def scores(self, when):
        when = pd.Timestamp(when)
        if when not in self._cache:
            f = self.fundamentals
            ind = f.indicators(self.panel, self.companies, self.quotes, when, self.history)
            scores = f.factor_scores(ind, self.companies["sector"])["fundamental_score"]
            self._cache[when] = self.tickers.set_index("symbol")["cnpj"].map(scores)
        return self._cache[when]


def _uses_fundamentals(strategy):
    return strategy.rules.rank_by in ("auto", "fundamentals") or strategy.settings.use_fundamentals


def _weight_fn(strategy, point_in_time, exante, traded=None, fundamentals=None):
    # With traded values, liquidity is measured at each rebalance (see liquid_assets)
    rules = replace(strategy.rules, min_adtv=0.0, max_per_class=None) if traded is not None else strategy.rules

    def weight_fn(past, when):
        day = when.strftime("%Y-%m-%d")
        assets, snapshot = selected_assets(strategy.snapshot or day, rules, as_of=day)
        point_in_time[when] = snapshot <= day
        if fundamentals is not None and _uses_fundamentals(strategy):
            # The snapshot's score is today's: replace it with the one of the rebalance date
            assets = assets.assign(fundamental_score=fundamentals.scores(when).reindex(assets.index))
        if traded is not None:
            assets = liquid_assets(assets, traded, when, strategy.rules)
        # Assets with a full window of prices; the CDI is always kept as the risk-free rate
        complete = [c for c in past.columns if c in assets.index and past[c].notna().all()]
        risky = [c for c in complete if c != CDI]
        if not risky:  # nothing investable on that date (e.g. BDRs before they were liquid): cash
            return pd.Series({CDI: 1.0})
        if strategy.kind == "equal":
            return pd.Series(1 / len(risky), index=risky)
        if strategy.kind == "legacy":
            return legacy_weights(past[risky], strategy.legacy_risk)
        prices = past[list(dict.fromkeys(complete + [CDI]))]
        problem, details = portfolio.build_problem(prices, assets.loc[complete], strategy.settings, as_of=when)
        w = portfolio.solve(problem, strategy.settings, details["rf"])
        exante[when] = problem.stats(w)["cvar"]
        return w
    return weight_fn


MULTI_ASSET = ("Stocks", "BDRs", "FIIs", "Fixed income", "Global ETFs", "Gold & FX")


def suite(name, snapshot=None):
    """Predefined comparisons.

    "model": the original model vs the new one, and stocks only vs several asset classes,
    over a long period. The universe is pinned to one snapshot and ranked by liquidity with
    no analyst filter, so no analyst opinion leaks from the future (prices of today's
    listed assets still carry survivorship bias).
    "analysts": the analyst filters with point-in-time data only, from the first snapshot on.
    "analysts_bdr": the analyst filters on BDRs with the consensus of each month end, rebuilt
    from the brokers' rating changes (analysts.build_us_history), from 2020 (BDRs became
    liquid when they opened to all investors in late 2020; before that the strategies hold cash).
    "fundamentals": stocks ranked by liquidity vs by the CVM factor score of each rebalance
    date, and the fundamental tilt of expected returns, from 2016.
    "estimators": historical vs shrunk expected returns, each with the CDI's or the expected
    returns as the drift of the CVaR budget, on the "model" universes.
    """
    pinned = snapshot or universe.list_snapshots()[-1]
    neutral = dict(ratings=tuple(analysts.RATINGS), include_uncovered=True, rank_by="liquidity")
    stocks = analysts.FilterRules(classes=("Stocks",), **neutral)
    multi = analysts.FilterRules(classes=MULTI_ASSET, **neutral)
    model = portfolio.ModelSettings
    if name == "model":
        return [
            Strategy("Original model: stocks, CVaR 10%", stocks, kind="legacy", legacy_risk=10.0, snapshot=pinned),
            Strategy("Equal weight: stocks", stocks, kind="equal", snapshot=pinned),
            Strategy("New model: stocks, CVaR <= 10%", stocks, model(objective="risk_budget", risk_budget=0.10), snapshot=pinned),
            Strategy("New model: multi-asset, CVaR <= 10%", multi, model(objective="risk_budget", risk_budget=0.10), snapshot=pinned),
            Strategy("New model: multi-asset, CVaR <= 5%", multi, model(objective="risk_budget", risk_budget=0.05), snapshot=pinned),
            Strategy("New model: multi-asset, tangency", multi, model(objective="max_ratio"), snapshot=pinned),
            Strategy("Equal weight: multi-asset", multi, kind="equal", snapshot=pinned),
        ]
    if name == "estimators":
        out = []
        for label, rules, budget in (("stocks, CVaR <= 10%", stocks, 0.10), ("multi-asset, CVaR <= 5%", multi, 0.05)):
            for method in portfolio.RETURN_METHODS:
                for drift in portfolio.RISK_DRIFTS:
                    settings = model(objective="risk_budget", risk_budget=budget, return_method=method, risk_drift=drift)
                    out.append(Strategy(f"{label}: {method}, drift {drift}", rules, settings, snapshot=pinned))
            out.append(Strategy(f"{label}: equal weight", rules, kind="equal", snapshot=pinned))
        return out
    if name == "analysts":
        variants = {
            "strong_buy only (original default)": analysts.FilterRules(classes=("Stocks",), ratings=("strong_buy",),
                                                                       max_per_class=None, rank_by="score"),
            "excluding underperform/sell": analysts.FilterRules(classes=("Stocks",), rank_by="score"),
            "no analyst filter": stocks,
        }
        out = []
        for label, rules in variants.items():
            out.append(Strategy(f"Equal weight: {label}", rules, kind="equal"))
            out.append(Strategy(f"New model: {label}", rules, model(objective="risk_budget", risk_budget=0.10)))
        return out
    if name == "analysts_bdr":
        variants = {
            "no analyst filter": analysts.FilterRules(classes=("BDRs",), **neutral),
            "excluding underperform/sell": analysts.FilterRules(classes=("BDRs",), rank_by="score"),
            "strong_buy only": analysts.FilterRules(classes=("BDRs",), ratings=("strong_buy",), rank_by="score"),
        }
        return [Strategy(f"Equal weight: {label}", rules, kind="equal", snapshot=pinned)
                for label, rules in variants.items()]
    if name == "fundamentals":
        by_liquidity = stocks
        by_score = replace(stocks, rank_by="fundamentals")
        budget = dict(objective="risk_budget", risk_budget=0.10)
        tilt = model(**budget, projections=True, use_analysts=False, use_fii_yield=False, use_bond_carry=False,
                     use_focus=False, use_fundamentals=True)
        return [
            Strategy("Equal weight: 25 most liquid stocks", by_liquidity, kind="equal", snapshot=pinned),
            Strategy("Equal weight: 25 best fundamental scores", by_score, kind="equal", snapshot=pinned),
            Strategy("New model: most liquid, CVaR <= 10%", by_liquidity, model(**budget), snapshot=pinned),
            Strategy("New model: best scores, CVaR <= 10%", by_score, model(**budget), snapshot=pinned),
            Strategy("New model: most liquid + fundamental tilt", by_liquidity, tilt, snapshot=pinned),
        ]
    raise ValueError(f"Unknown suite: {name}")


def run_suite(strategies, start, end, lookback=252, rebalance=21, cost=0.001, log=print):
    """Runs each strategy plus the CDI and Ibovespa benchmarks over [start, end].

    Returns a dict with the value series (`values`), the metrics table (`metrics`), each
    strategy's walk_forward result (`results`) and the share of rebalances that used a
    universe snapshot from before the rebalance date (`point_in_time`).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    data_start = start - pd.Timedelta(days=int(lookback * 1.6) + 15)
    symbols = {CDI}
    for s in strategies:
        for day in [s.snapshot] if s.snapshot else universe.list_snapshots():
            symbols |= set(selected_assets(day, candidate_rules(s.rules))[0].index)
    log(f"Downloading prices of {len(symbols)} assets...")
    prices, _, traded = market_data.collect_prices(sorted(symbols), data_start, end, require_full_history=False,
                                                   with_traded_value=True)
    prices = prices.dropna(subset=[CDI])
    traded = traded.reindex(prices.index)
    pit_fundamentals = None
    if any(_uses_fundamentals(s) for s in strategies):
        log("Computing point-in-time fundamentals (CVM filings)...")
        latest, _ = _snapshot(universe.list_snapshots()[-1])
        stocks = latest[(latest["asset_class"] == "Stocks") & latest["symbol"].isin(symbols)]
        pit_fundamentals = PointInTimeFundamentals(stocks, data_start, end + pd.Timedelta(days=1))
    values, results, pit, exante = {}, {}, {}, {}
    for s in strategies:
        log(f"Backtesting: {s.name}")
        flags, risks = {}, {}
        weight_fn = _weight_fn(s, flags, risks, traded if s.pit_liquidity else None, pit_fundamentals)
        results[s.name] = walk_forward(prices, weight_fn, start, end, lookback, rebalance, cost)
        values[s.name] = results[s.name]["value"]
        pit[s.name] = float(np.mean(list(flags.values()))) if flags else np.nan
        exante[s.name] = float(np.mean(list(risks.values()))) if risks else np.nan
    first_day = min(v.index[0] for v in values.values())
    cdi = prices.loc[first_day:, CDI]
    values["CDI"] = cdi / cdi.iloc[0]
    ibov = yf.download("^BVSP", start=first_day, end=end + pd.Timedelta(days=1), auto_adjust=True, progress=False)["Close"]
    ibov = ibov.squeeze("columns").dropna()
    if len(ibov):
        values["Ibovespa"] = ibov / ibov.iloc[0]
    rf = prices[CDI].pct_change()
    metrics = pd.DataFrame({name: performance(v, rf) for name, v in values.items()}).T
    metrics["Ex-ante CVaR (month)"] = pd.Series(exante)
    metrics["Avg. turnover"] = pd.Series({n: r["turnover"].iloc[1:].mean() for n, r in results.items()})
    metrics["Point in time"] = pd.Series(pit)
    return {"values": values, "metrics": metrics, "results": results, "point_in_time": pit, "prices": prices}
