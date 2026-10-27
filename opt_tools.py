# Streamlit layer of the app: cached data loaders, sidebar controls and plots.
# The model itself lives in optimizer.py / portfolio.py, the data in universe.py,
# analysts.py and market_data.py, and the backtests in backtesting.py.
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from stqdm import stqdm

import analysts
import market_data
import optimizer as opt
import portfolio
import universe

CLASS_COLORS = dict(zip(universe.ASSET_CLASSES, px.colors.qualitative.Safe))


# ---- data ----
@st.cache_data(ttl=6 * 3600, show_spinner="Loading the universe snapshot...")
def load_universe(day=None):
    """Snapshot in force on `day` (default: today) and its date."""
    return universe.read_universe(day)


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices...")
def load_prices(symbols, start, end, require_full_history=True):
    return market_data.collect_prices(list(symbols), start, end, require_full_history=require_full_history)


@st.cache_data(ttl=24 * 3600, show_spinner="Reading the CVM filings...")
def load_cvm_history():
    """Fiscal-year history of every company (None before fundamentals.build_table runs)."""
    import fundamentals
    if not fundamentals.TABLE_PATH.exists():
        return None
    return fundamentals.annual_history(fundamentals.filings())


def warn_dropped(dropped):
    """Shows which assets were left out of the price history and why."""
    if dropped:
        detail = ", ".join(f"{s} ({why})" for s, why in sorted(dropped.items()))
        st.warning(f"Left out for missing data: {detail}")


def snapshot_notice(snapshot):
    age = (pd.Timestamp(date.today()) - pd.Timestamp(snapshot)).days
    st.sidebar.caption(f"Universe snapshot: **{snapshot}** ({age} days old)")
    if age > 30:
        st.sidebar.warning("Analyst data is over a month old: run `python build_dataset.py`.")


# ---- sidebar controls ----
def universe_controls(df, key=""):
    """Widgets for the asset universe and the analyst filters. Returns FilterRules."""
    st.sidebar.header("Asset universe")
    available = [c for c in universe.ASSET_CLASSES if c in set(df["asset_class"])]
    classes = st.sidebar.multiselect(
        "Asset classes", available, [c for c in ("Stocks", "BDRs", "FIIs", "Fixed income") if c in available],
        help="BDRs give access to foreign companies in BRL; Fixed income includes the CDI "
             "(Tesouro Selic / CDB proxy), Tesouro IPCA+ and Prefixado bonds, fixed income ETFs "
             "and FI-Infra funds.", key=f"classes{key}",
    )
    min_adtv = st.sidebar.number_input(
        "Minimum average daily traded value (R$ thousand)", 0, 100_000, 500, 100, key=f"adtv{key}",
        help="Illiquid assets have stale prices, which hide their risk, and are hard to trade.",
    ) * 1000
    max_per_class = st.sidebar.slider("Maximum assets per class", 5, 100, 25, 5, key=f"top{key}",
                                      help="The best ranked by analyst score (Stocks, BDRs) or liquidity.")

    st.sidebar.header("Analyst filter (Stocks and BDRs)")
    ratings = st.sidebar.multiselect(
        "Consensus rating", analysts.RATINGS, ["strong_buy", "buy", "hold"], key=f"ratings{key}",
        help="Excluding Underperform/Sell is what the evidence supports; keeping only "
             "strong_buy did worse out of sample (see the notebook).",
    )
    min_analysts = st.sidebar.slider("Minimum number of analysts", 1, 15, 3, key=f"nan{key}")
    include_uncovered = st.sidebar.checkbox("Keep stocks/BDRs without analyst coverage", False, key=f"unc{key}")
    with st.sidebar.expander("More analyst filters"):
        use_upside = st.checkbox("Minimum upside to the target price", False, key=f"upc{key}")
        min_upside = st.slider("Upside (%)", -20, 50, 0, key=f"up{key}") / 100 if use_upside else None
        use_downgrade = st.checkbox("Exclude recent downgrades", False, key=f"dgc{key}",
                                    help="Consensus worse by more than this over the last 3 months (1-5 scale).")
        max_downgrade = st.slider("Max downgrade", 0.05, 1.0, 0.25, 0.05, key=f"dg{key}") if use_downgrade else None
        use_disp = st.checkbox("Maximum disagreement among analysts", False, key=f"dsc{key}")
        max_disp = st.slider("(High - low target) / price (%)", 20, 300, 100, 10, key=f"ds{key}") / 100 if use_disp else None
        rank_by = st.radio("Rank within each class by", ["auto", "fundamentals", "score", "liquidity"], key=f"rank{key}",
                           format_func={"auto": "Fundamentals for stocks, liquidity for the rest",
                                        "fundamentals": "Fundamental score (CVM) only", "score": "Analyst score",
                                        "liquidity": "Liquidity"}.get,
                           help="The fundamental score ranks stocks by value, quality, momentum and low risk within "
                                "their sector, from the CVM filings; it predicted returns far better than the analyst "
                                "consensus in the notebook, and ranking BDRs by analysts did worse than by liquidity. "
                                "Classes without a score are ranked by liquidity.")
    with st.sidebar.expander("FII filters"):
        use_dy = st.checkbox("Minimum dividend yield", False, key=f"dyc{key}")
        fii_min_dy = st.slider("Dividend yield (%)", 0.0, 20.0, 8.0, 0.5, key=f"dy{key}") / 100 if use_dy else None
        use_pvp = st.checkbox("Maximum P/VP (when available)", False, key=f"pvc{key}")
        fii_max_pvp = st.slider("P/VP", 0.5, 1.5, 1.05, 0.05, key=f"pv{key}") if use_pvp else None
    return analysts.FilterRules(
        classes=tuple(classes), min_adtv=float(min_adtv), ratings=tuple(ratings), min_analysts=min_analysts,
        min_upside=min_upside, max_downgrade=max_downgrade, max_dispersion=max_disp,
        include_uncovered=include_uncovered, fii_min_dy=fii_min_dy, fii_max_pvp=fii_max_pvp,
        max_per_class=max_per_class, rank_by=rank_by,
    )


def model_controls(classes, key="", objective=True):
    """Widgets for the optimization model. Returns ModelSettings."""
    st.sidebar.header("Optimization model")
    settings = {}
    if objective:
        settings["objective"] = st.sidebar.selectbox(
            "Objective", list(portfolio.OBJECTIVES), format_func=portfolio.OBJECTIVES.get, key=f"obj{key}")
        if settings["objective"] == "risk_budget":
            settings["risk_budget"] = st.sidebar.slider(
                "CVaR budget (% per month)", 1.0, 30.0, 5.0, 0.5, key=f"rb{key}",
                help="Average loss you accept in the worst 5% of months. The Ibovespa's is about 10-11%.") / 100
    settings["return_method"] = st.sidebar.selectbox(
        "Expected returns", list(portfolio.RETURN_METHODS), format_func=portfolio.RETURN_METHODS.get, key=f"ret{key}",
        help="The mean of the lookback window carries momentum, and it did better in the notebook's "
             "walk-forward tests; shrinkage pulls every asset toward the same return per unit of risk.")
    settings["risk_drift"] = st.sidebar.selectbox(
        "Expected return in the CVaR", list(portfolio.RISK_DRIFTS), format_func=portfolio.RISK_DRIFTS.get,
        key=f"drift{key}",
        help="With the CDI's, the risk budget does not depend on the expected returns. Crediting each "
             "asset's own let recent winners loosen it: the realized CVaR reached about twice the budget.")
    settings["confidence"] = st.sidebar.select_slider("CVaR confidence level", [0.90, 0.95, 0.975, 0.99], 0.95, key=f"conf{key}")
    halflife = st.sidebar.slider("Scenario half-life (trading days, 0 = equal weights)", 0, 504, 0, 21, key=f"hl{key}",
                                 help="Gives more weight to recent days in the risk and return estimates.")
    settings["halflife"] = halflife or None
    settings["max_weight"] = st.sidebar.slider("Maximum weight per asset (%)", 5, 100, 20, 5, key=f"mw{key}",
                                               help="The CDI is exempt (it is the cash position).") / 100
    settings["min_weight"] = st.sidebar.slider("Minimum position (%)", 0.0, 5.0, 1.0, 0.5, key=f"minw{key}") / 100
    bounds = {}
    with st.sidebar.expander("Limits per asset class (%)"):
        for c in classes:
            lo, hi = st.slider(c, 0, 100, (0, 100), 5, key=f"bound{c}{key}")
            if (lo, hi) != (0, 100):
                bounds[c] = (lo / 100, hi / 100)
    settings["class_bounds"] = bounds

    st.sidebar.header("Projections (forward-looking)")
    settings["projections"] = st.sidebar.toggle(
        "Blend history with projections", False, key=f"proj{key}",
        help="Black-Litterman: each projection moves expected returns according to its confidence.")
    if settings["projections"]:
        settings["use_analysts"] = st.sidebar.checkbox("Analysts' target prices + dividends (Stocks, BDRs)", True, key=f"pa{key}")
        settings["use_fii_yield"] = st.sidebar.checkbox("FII dividend yield as the class's return", True, key=f"pf{key}")
        settings["use_bond_carry"] = st.sidebar.checkbox(
            "Tesouro bonds: their yield (IPCA+ with the Focus inflation)", True, key=f"pb{key}")
        settings["use_focus"] = st.sidebar.checkbox("Focus survey: expected CDI for the next 12 months", True, key=f"pfo{key}")
        settings["relative_projections"] = st.sidebar.checkbox(
            "Only relative to peers (removes analysts' optimism)", True, key=f"prel{key}")
        settings["analyst_std"] = st.sidebar.slider(
            "Uncertainty of analyst projections (% a year)", 10, 150, 100, 10, key=f"pas{key}",
            help="100% matches how little the consensus predicted in the notebook's 10-year "
                 "point-in-time test; lower values trust analysts more.") / 100
        settings["fii_std"] = st.sidebar.slider("Uncertainty of the FII carry (% a year)", 2, 30, 10, 1, key=f"pfs{key}") / 100
        settings["use_fundamentals"] = st.sidebar.checkbox(
            "Fundamental score tilt (stocks, CVM filings)", False, key=f"pfu{key}",
            help="Expected return + IC x volatility x score (Grinold's rule).")
        if settings["use_fundamentals"]:
            settings["fundamental_ic"] = st.sidebar.slider(
                "Predictive power of the score (rank IC)", 0.0, 0.10, portfolio.ModelSettings().fundamental_ic, 0.005,
                key=f"pic{key}", format="%.3f", help="The notebook measures it out of sample; 0 disables the tilt.")
    return portfolio.ModelSettings(**settings)


def funnel(filtered):
    """How many assets each filter removed."""
    counts = filtered["status"].value_counts()
    cols = st.columns(min(len(counts), 6))
    for col, (status, n) in zip(cols, counts.items()):
        col.metric("Selected" if status == "selected" else f"Out: {status}", int(n))


UNIVERSE_COLUMNS = {
    "symbol": st.column_config.TextColumn("Ticker"),
    "asset_class": st.column_config.TextColumn("Class"),
    "sub_class": st.column_config.TextColumn("Type"),
    "name": st.column_config.TextColumn("Name", width="medium"),
    "underlying": st.column_config.TextColumn("Underlying"),
    "rec_key": st.column_config.TextColumn("Consensus"),
    "n_analysts": st.column_config.NumberColumn("Analysts", format="%d"),
    "upside": st.column_config.NumberColumn("Upside", format="percent"),
    "revision": st.column_config.NumberColumn("Revision 3m", format="%.2f", help="Positive: upgrades"),
    "analyst_score": st.column_config.NumberColumn("Analyst score", format="%.2f"),
    "pe": st.column_config.NumberColumn("P/L", format="%.1f"),
    "pb": st.column_config.NumberColumn("P/VP", format="%.2f"),
    "roe": st.column_config.NumberColumn("ROE", format="percent"),
    "fundamental_score": st.column_config.NumberColumn("Fund. score", format="%.2f",
                                                       help="Value, quality, momentum and low risk vs. the sector (z-score)"),
    "dividend_yield": st.column_config.NumberColumn("Div. yield", format="percent"),
    "adtv": st.column_config.NumberColumn("Daily volume (R$)", format="compact"),
    "status": st.column_config.TextColumn("Status"),
}


def universe_table(df):
    cols = [c for c in UNIVERSE_COLUMNS if c in df]
    st.dataframe(df[cols], column_config=UNIVERSE_COLUMNS, hide_index=True, use_container_width=True)


# ---- model ----
class Portfolio_optimization:
    """CVaR model of the app: builds the problem with portfolio.build_problem and plots it."""

    def __init__(self, historic_stocks, assets, settings, as_of=None):
        self.historic_stocks = historic_stocks
        self.assets = assets  # universe rows indexed by symbol
        self.settings = settings
        self.problem, self.details = portfolio.build_problem(historic_stocks, assets, settings, as_of)
        self.stocks = self.problem.assets
        self.classes = assets.reindex(self.stocks)["asset_class"]

    def plot_historic(self):
        """Prices rebased to 100, so that assets of very different prices can be compared."""
        rebased = 100 * self.historic_stocks / self.historic_stocks.iloc[0]
        fig = go.Figure()
        for stock in rebased.columns:
            fig.add_trace(go.Scatter(x=rebased.index, y=rebased[stock], mode="lines", name=stock, line=dict(width=1)))
        fig.update_layout(title="Price history (rebased to 100, dividends reinvested)",
                          xaxis_title="Time period", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    def optimize(self, n_points=30):
        """Efficient frontier plus the minimum-CVaR and tangency portfolios."""
        self.frontier, self.frontier_weights = self.problem.frontier(n_points, progress=stqdm)
        self.min_risk = self.frontier_weights.iloc[0]
        self.tangency = portfolio.solve(self.problem, portfolio.ModelSettings(objective="max_ratio"), self.details["rf"])
        self.asset_stats = self.problem.asset_stats()
        return {"frontier": self.frontier, "weights": self.frontier_weights}

    def key_portfolios(self):
        rows = {"Minimum CVaR": self.min_risk, "Tangency (best return/risk)": self.tangency}
        if self.settings.objective == "risk_budget":
            rows["Your CVaR budget"] = self.problem.max_return(self.settings.risk_budget)
        table = {}
        for name, w in rows.items():
            s = self.problem.stats(w)
            table[name] = {
                "Expected return (month)": s["exp_return"], "CVaR (month)": s["cvar"],
                "Expected return (year)": s["annual_return"], "Volatility (year)": s["annual_vol"],
                "Assets": s["n_assets"], **{f"% {c}": w.groupby(self.classes).sum().get(c, 0.0) for c in sorted(set(self.classes))},
            }
        return pd.DataFrame(table).T, rows

    def plot_efficient_frontiers(self):
        h = self.problem.h
        fig = go.Figure()
        for c, grp in self.asset_stats.groupby(self.classes):
            fig.add_trace(go.Scatter(
                x=100 * grp["cvar"], y=100 * grp["exp_return"], mode="markers", name=c, text=grp.index,
                marker=dict(size=7, opacity=0.6, color=CLASS_COLORS.get(c)),
                hovertemplate="%{text}<br>CVaR %{x:.1f}%<br>Return %{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=100 * self.frontier["cvar"], y=100 * self.frontier["exp_return"],
                                 mode="lines", name="Efficient frontier", line=dict(width=3, color="black")))
        t = self.problem.stats(self.tangency)
        rf_h = h * self.details["rf"]
        fig.add_trace(go.Scatter(x=[-100 * rf_h, 100 * t["cvar"]], y=[100 * rf_h, 100 * t["exp_return"]], mode="lines",
                                 name="CDI + tangency mix", line=dict(dash="dash", color="gray")))
        for name, w, symbol in [("Minimum CVaR", self.min_risk, "diamond"), ("Tangency", self.tangency, "star")]:
            s = self.problem.stats(w)
            fig.add_trace(go.Scatter(x=[100 * s["cvar"]], y=[100 * s["exp_return"]], mode="markers", name=name,
                                     marker=dict(size=14, symbol=symbol, color="crimson")))
        conf = f"{self.problem.beta:.0%}"
        fig.update_layout(title=f"CVaR {conf} efficient frontier ({h} trading days)",
                          xaxis_title=f"CVaR {conf} over {h} days (%)", yaxis_title=f"Expected return over {h} days (%)")
        st.markdown(
            "Each dot is an asset held alone; the black line is the best expected return for each "
            "level of tail risk. CVaR is the average loss in the worst "
            f"{1 - self.problem.beta:.0%} of months; a negative value means even the bad tail is a "
            "gain (the CDI). The dashed line mixes the CDI with the tangency portfolio.")
        st.plotly_chart(fig, use_container_width=True)

    def plot_stocks_distribution(self, plot=True):
        """Composition of the frontier portfolios as risk grows."""
        x = 100 * self.frontier["cvar"]
        y = 100 * self.frontier_weights
        fig = go.Figure()
        for stock in y.columns[(y > 0).any()]:
            fig.add_trace(go.Scatter(x=x, y=y[stock], mode="lines", line=dict(width=0.5), name=stock, stackgroup="one"))
        fig.update_layout(title="Asset distribution along the frontier", xaxis_title="CVaR (%)", yaxis_title="Distribution (%)")
        if plot:
            st.markdown("The area chart shows the proportion of each asset for each level of risk.")
            st.plotly_chart(fig, use_container_width=True)
        return x, y

    def proportion_risk(self, risk_value, plot=True):
        """Highest expected return portfolio with a CVaR of at most risk_value (% over h days)."""
        w = self.problem.max_return(max_risk=risk_value / 100)
        if plot:
            self.plot_allocation(w, f"Portfolio with CVaR <= {risk_value:.1f}%")
        return w

    def plot_allocation(self, w, title):
        w = 100 * w[w > 0].sort_values()
        frame = pd.DataFrame({"weight": w, "class": self.classes.reindex(w.index)})
        left, right = st.columns([3, 2])
        fig = px.bar(frame, x="weight", y=frame.index, color="class", orientation="h",
                     color_discrete_map=CLASS_COLORS, title=title)
        fig.update_layout(xaxis_title="Weight (%)", yaxis_title="", height=max(350, 22 * len(frame)))
        left.plotly_chart(fig, use_container_width=True)
        pie = px.pie(frame, values="weight", names="class", color="class", color_discrete_map=CLASS_COLORS,
                     title="By asset class", hole=0.45)
        right.plotly_chart(pie, use_container_width=True)


def projections_table(po):
    """Expected returns before and after the projections (annual)."""
    d = po.details
    table = pd.DataFrame({
        "Class": po.classes,
        "History (shrunk)" if po.settings.return_method == "shrinkage" else "History": d["prior"] * opt.TRADING_DAYS,
        "With projections": d["mu"] * opt.TRADING_DAYS,
    })
    if d["views"] is not None:
        table["Projection"] = d["views"]["q"].reindex(table.index)
    table["Change"] = table["With projections"] - table.iloc[:, 1]
    return table.sort_values("Change")


# ---- legacy price transformations (experimental) ----
# Both smooth prices before returns are computed, so the measured risk (and the CVaR budget
# you choose) is far below the real one. Kept for comparison in the backtests.
def df_moving_avg(df, sma=25):
    return df.rolling(sma).mean().dropna()


class Particle:
    def __init__(self, points):
        self.position = np.random.rand(points)
        self.velocity = np.random.rand(points)
        self.best_position = np.copy(self.position)
        self.best_score = float("inf")


class PSO_optimal_points:
    def __init__(self, series, num_points):
        self.series = series
        self.num_points = num_points
        self.score = []

    def _interpolate_nan(self, array_like):
        array = array_like.copy()
        nans = np.isnan(array)

        def get_x(a):
            return a.nonzero()[0]

        array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])
        return array

    def _get_values_at_indices(self, indices, array):
        result = np.where(np.isin(np.arange(len(array)), indices), array, np.nan)
        return result.tolist()

    def _top_n_indices(self, lst, n):
        if n > len(lst):
            raise ValueError("N should be less than or equal to the length of the list.")
        indices = np.argsort(lst)[-n:]
        return indices.tolist()

    def objective_function(self, points, original_series):
        if len(points) == 0:
            return float("inf")
        # Objective function: difference between the original series and the chosen points
        subset = self._get_values_at_indices(points, original_series)
        interpolated_series = self._interpolate_nan(np.array(subset))
        return np.sum(np.abs(original_series - interpolated_series))

    def pso(self, num_particles, num_iterations):
        n_series = len(self.series)
        particles = [Particle(n_series) for _ in range(num_particles)]
        global_best_position = None
        global_best_score = float("inf")
        for _ in range(num_iterations):
            for particle in particles:
                points = self._top_n_indices(particle.position, self.num_points)
                score = self.objective_function(points, self.series)
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = np.copy(particle.position)
            for particle in particles:
                inertia, personal_weight, global_weight = 0.5, 2.0, 2.0
                r1, r2 = np.random.rand(n_series), np.random.rand(n_series)
                particle.velocity = (
                    inertia * particle.velocity
                    + personal_weight * r1 * (particle.best_position - particle.position)
                    + global_weight * r2 * (global_best_position - particle.position)
                )
                particle.position = np.clip(particle.position + particle.velocity, 0, 1)
            self.score.append(score)
        return self._top_n_indices(global_best_position, self.num_points)


def df_optimal_pso_points(df, num_points_to_choose=30, num_particles=15, num_iterations=40):
    stocks_series_list = []
    for stock in list(df.columns):
        stock_series = df[stock]
        pop = PSO_optimal_points(stock_series, num_points_to_choose)
        chosen_indices = sorted(pop.pso(num_particles, num_iterations))
        stocks_series_list.append(stock_series.iloc[chosen_indices])
    placeholder_df = pd.DataFrame(index=df.index)
    return pd.concat(stocks_series_list + [placeholder_df], axis=1).sort_index().interpolate(limit_direction="both")
