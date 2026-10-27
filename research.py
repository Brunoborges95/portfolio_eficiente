"""Research helpers for the notebook: does analyst information predict returns?

- event_study(): returns after a saved universe snapshot, grouped by consensus rating.
- point_in_time_signals(): rebuilds, month by month, the consensus on US companies (the
  BDRs' underlyings) from Yahoo's history of rating changes and price targets, using only
  the actions published up to each month.
- information_coefficients(): cross-sectional rank correlation between each signal and the
  next period's return (the "IC"), date by date.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

import market_data
import universe
from analysts import broker_actions, consensus_at

SIGNALS = ["rating", "upside", "revision", "target_revision", "dispersion", "momentum"]


def event_study(snapshot_day, end, ratings=("strong_buy", "buy", "hold", "underperform", "sell")):
    """Total return from a snapshot date to `end` of every rated asset in that snapshot,
    with the CDI and the Ibovespa over the same period."""
    df, snap = universe.read_universe(snapshot_day)
    rated = df[df["rec_key"].isin(ratings) & (df["asset_class"] == "Stocks")].copy()
    prices, dropped = market_data.collect_prices(list(rated["symbol"]), snap, end, require_full_history=False)
    first = prices.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
    last = prices.apply(lambda c: c.dropna().iloc[-1] if c.notna().any() else np.nan)
    rated["return"] = rated["symbol"].map(last / first - 1)
    cdi = market_data.cdi_index(snap, end)
    ibov = yf.download("^BVSP", start=snap, end=end, auto_adjust=True, progress=False)["Close"].squeeze("columns").dropna()
    benchmarks = {"CDI": cdi.iloc[-1] / cdi.iloc[0] - 1, "Ibovespa": ibov.iloc[-1] / ibov.iloc[0] - 1}
    return rated, benchmarks, dropped


def point_in_time_signals(upgrades, prices, dates, window_days=365, revision_days=91):
    """Consensus signals of each ticker on each date, using only past broker actions.

    rating: mean broker rating (1 = strong buy); revision: rating 3 months earlier minus now
    (positive = upgrades); upside: mean price target / price - 1; target_revision: change of
    the mean target over 3 months; dispersion: (highest - lowest target) / price;
    momentum: 12-month return skipping the last month (a market signal, for comparison).
    """
    rows = []
    for ticker, ud in upgrades.items():
        if ticker not in prices:
            continue
        actions = broker_actions(ud)
        px = prices[ticker].dropna()
        for d in dates:
            if px.empty or d < px.index[0] + pd.Timedelta(days=365):
                continue
            rating, n, targets = consensus_at(actions, d, window_days)
            before, _, old_targets = consensus_at(actions, d - pd.Timedelta(days=revision_days), window_days)
            price = px.asof(d)
            rows.append({
                "date": d, "ticker": ticker, "n": n, "rating": rating,
                "revision": before - rating,
                "upside": targets.mean() / price - 1 if len(targets) else np.nan,
                "target_revision": targets.mean() / old_targets.mean() - 1 if len(targets) and len(old_targets) else np.nan,
                "dispersion": (targets.max() - targets.min()) / price if len(targets) > 1 else np.nan,
                "momentum": px.asof(d - pd.Timedelta(days=30)) / px.asof(d - pd.Timedelta(days=365)) - 1,
            })
    return pd.DataFrame(rows)


def dividend_yield_signals(tickers, dates, start, end, workers=8):
    """Trailing 12-month dividend yield of each B3 fund on each date, from its dividend
    history and unadjusted prices (point in time). Returns (signals, total-return prices)."""
    yahoo = [t + ".SA" for t in tickers]

    def dividends(t):
        try:
            d = yf.Ticker(t).dividends
        except Exception:
            return t, None
        if d is None or d.empty:
            return t, None
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return t, d

    with ThreadPoolExecutor(workers) as ex:
        divs = {t.removesuffix(".SA"): d for t, d in ex.map(dividends, yahoo) if d is not None}
    raw = yf.download(yahoo, start=start, end=end, auto_adjust=False, progress=False)
    close = raw["Close"].rename(columns=lambda c: c.removesuffix(".SA"))
    total = raw["Adj Close"].rename(columns=lambda c: c.removesuffix(".SA"))
    rows = []
    for t, d in divs.items():
        if t not in close:
            continue
        px = close[t].dropna()
        for day in dates:
            if px.empty or day < px.index[0] + pd.Timedelta(days=365):
                continue
            paid = d[(d.index > day - pd.Timedelta(days=365)) & (d.index <= day)].sum()
            rows.append({"date": day, "ticker": t, "dividend_yield": paid / px.asof(day)})
    return pd.DataFrame(rows), total


def fundamental_signals(companies, panel, quotes, dates):
    """Fundamental indicators and factor scores of each company (cnpj) on each date, from
    the filings published by then (fundamentals.indicators / factor_scores)."""
    import fundamentals
    history = fundamentals.annual_history(panel)
    frames = []
    for d in dates:
        ind = fundamentals.indicators(panel, companies, quotes, d, history)
        scores = fundamentals.factor_scores(ind, companies["sector"])
        frames.append(ind.join(scores).assign(date=d))
    out = pd.concat(frames)
    out.index.name = "ticker"
    return out.reset_index()


def forward_returns(prices, dates, months=1):
    """Return of each ticker from each date to `months` months later."""
    # Row lookups on forward-filled prices (DataFrame.asof would skip rows with any NaN)
    filled = prices.ffill()
    out = {}
    for d in dates:
        later = d + pd.DateOffset(months=months)
        if later > prices.index[-1]:
            continue
        now = filled.iloc[filled.index.searchsorted(d, side="right") - 1]
        then = filled.iloc[filled.index.searchsorted(later, side="right") - 1]
        out[d] = then / now - 1
    return pd.DataFrame(out).T


def information_coefficients(signals, fwd, min_names=20, columns=SIGNALS):
    """Spearman correlation, date by date, between each signal and the forward return.
    Signals are oriented so that a positive IC means "the signal works" (a low rating and
    a low dispersion are good)."""
    sign = {"rating": -1, "dispersion": -1}
    rows = {}
    for d, grp in signals.groupby("date"):
        if d not in fwd.index:
            continue
        g = grp.set_index("ticker").join(fwd.loc[d].rename("fwd"), how="inner").dropna(subset=["fwd"])
        ics = {}
        for s in columns:
            x = g[[s, "fwd"]].dropna()
            if len(x) >= min_names:
                ics[s] = sign.get(s, 1) * x[s].rank().corr(x["fwd"].rank())
        rows[d] = ics
    return pd.DataFrame(rows).T


def summarize_ic(ic):
    """Mean IC, its t-statistic and the share of dates with a positive IC."""
    n = ic.notna().sum()
    return pd.DataFrame({
        "mean IC": ic.mean(),
        "t-stat": ic.mean() / ic.std() * np.sqrt(n),
        "% dates IC > 0": (ic > 0).sum() / n,
        "dates": n,
    })


def quintile_spread(signals, fwd, signal):
    """Average forward return of each quintile of `signal` (Q5 = best by the IC orientation)."""
    sign = {"rating": -1, "dispersion": -1}.get(signal, 1)
    out = []
    for d, grp in signals.groupby("date"):
        if d not in fwd.index:
            continue
        g = grp.set_index("ticker").join(fwd.loc[d].rename("fwd"), how="inner").dropna(subset=[signal, "fwd"])
        if len(g) < 25:
            continue
        q = pd.qcut((sign * g[signal]).rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
        out.append(g.groupby(q, observed=True)["fwd"].mean().rename(d))
    return pd.DataFrame(out)
