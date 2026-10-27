# Import necessary libraries
from datetime import date, timedelta

import streamlit as st

import analysts
import backtesting
import opt_tools
import optimizer
import universe
from market_data import CDI

# Page settings
st.set_page_config(page_title="Optimization", page_icon="👋", layout="wide")

# Main title
st.title("Efficient Portfolio Management")

# Introduction Section
st.markdown(
    """
    Build a portfolio across Brazilian stocks, **BDRs** (foreign companies traded on B3),
    real estate funds (FIIs, Fiagros), **fixed income** (CDI, Tesouro Direto bonds, fixed income
    ETFs, FI-Infra), global, gold and crypto ETFs, filtered by the **analysts' consensus** and
    optimized for tail risk.

    ## How it works

    - **Universe:** choose the asset classes and filters in the sidebar. BDRs use the consensus
      on the underlying company (AAPL34 uses Apple's analysts), which covers far more analysts
      than Brazilian stocks have.
    - **Analyst filter:** excludes Underperform/Sell ratings, requires a minimum coverage and
      ranks each class by a score of consensus, upside relative to peers and revisions.
    - **Optimization:** minimizes the CVaR (the average loss in the worst months) for each level
      of expected return, with position limits per asset and per class. Expected returns are the
      means of the lookback window, which carry momentum; the CVaR assumes no asset earns more than
      the CDI, so the risk budget does not depend on those noisy estimates.
    - **Projections (optional):** blends history with forward-looking information, namely the
      analysts' target prices, the FII dividend yields, the Tesouro bonds' yields and the Focus
      survey's expected CDI and inflation, each according to its confidence (Black-Litterman).

    After setting your criteria, click **Generate Optimization**. The **Results** page shows
    the portfolio for each risk level and the **Backtest** page tests the strategy in the past.
    """
)

df_stocks_info, snapshot = opt_tools.load_universe()
opt_tools.snapshot_notice(snapshot)

# Sidebar for date range selection
st.sidebar.header("Choose Date Range")
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=730),
                                   help="Two years or more: the CVaR is estimated from the worst days of the period.")
end_date = st.sidebar.date_input("End Date", date.today())

rules = opt_tools.universe_controls(df_stocks_info)
settings = opt_tools.model_controls(rules.classes)

st.sidebar.header("Experimental transformations")
st.sidebar.caption("Both smooth prices before the risk is measured, so the real risk is far higher than the one shown.")
sma_enabled = st.sidebar.checkbox("Moving Average")
if sma_enabled:
    sma_period = st.sidebar.slider("Select Moving Average Period", min_value=5, max_value=100, value=25, step=5)
pso_opt_enabled = st.sidebar.checkbox("Optimal Points")
if pso_opt_enabled:
    pso_opt = st.sidebar.slider("Select Number of Points", min_value=5, max_value=100, value=10, step=5)

with st.sidebar.expander("Update the universe snapshot"):
    st.caption("Downloads today's B3 listing, liquidity and analyst consensus (about 3 minutes). "
               "Each snapshot is kept, so backtests can use the data known at each date.")
    if st.button("Build today's snapshot"):
        with st.status("Building the snapshot...", expanded=True) as status:
            universe.build_universe(log=st.write)
            st.cache_data.clear()
            backtesting.selected_assets.cache_clear()
            backtesting._snapshot.cache_clear()
            status.update(label="Snapshot saved", state="complete")
        st.rerun()

# Section: Asset universe
st.markdown("## Asset universe and analyst filter")
filtered = analysts.apply_filters(df_stocks_info, rules)
opt_tools.funnel(filtered)
selected = filtered[filtered["status"] == "selected"]
opt_tools.universe_table(selected)
with st.expander("Excluded assets and the filter that excluded them"):
    opt_tools.universe_table(filtered[filtered["status"] != "selected"])

# Section: Optimization Button
if st.button("Generate Optimization", type="primary", disabled=selected.empty):
    # The CDI is always downloaded: it is the risk-free rate even when it is not investable
    symbols = tuple(dict.fromkeys(list(selected["symbol"]) + [CDI]))
    stock_history, dropped = opt_tools.load_prices(symbols, start_date, end_date)
    opt_tools.warn_dropped(dropped)
    if sma_enabled:
        stock_history = opt_tools.df_moving_avg(stock_history, sma=sma_period)
    if pso_opt_enabled:
        stock_history = opt_tools.df_optimal_pso_points(stock_history, pso_opt)
    try:
        po = opt_tools.Portfolio_optimization(stock_history, selected.set_index("symbol"), settings)
        po.optimize()
    except optimizer.InfeasibleError as e:
        st.error(str(e))
        st.stop()
    st.session_state["po"] = po

po = st.session_state.get("po")
if po is not None:
    st.caption("Results of the last click on Generate Optimization (change the settings and click again to update).")

    # Section: Historical Graph
    st.markdown("## Historical Chart")
    po.plot_historic()

    # Section: Efficient Frontier
    st.markdown("## Efficient Frontier")
    po.plot_efficient_frontiers()

    # Section: Key portfolios
    st.markdown("## Key portfolios")
    table, portfolios = po.key_portfolios()
    st.dataframe(table, use_container_width=True, column_config={
        c: st.column_config.NumberColumn(c, format="%d" if c == "Assets" else "percent") for c in table.columns})
    chosen = {"min_risk": "Minimum CVaR", "risk_budget": "Your CVaR budget"}.get(
        po.settings.objective, "Tangency (best return/risk)")
    po.plot_allocation(portfolios[chosen], f"{chosen}: allocation")

    if po.settings.projections:
        st.markdown("## Projections")
        focus = po.details["focus_cdi"]
        if focus is not None:
            st.markdown(f"Focus survey: the market expects an average CDI of **{focus:.2%}** a year over the next 12 months.")
        st.dataframe(opt_tools.projections_table(po), use_container_width=True, column_config={
            c: st.column_config.NumberColumn(c, format="percent")
            for c in ["History (shrunk)", "History", "With projections", "Projection", "Change"]})

    st.markdown("#### In the Results tab, you can find the ideal asset portfolio for each risk level.")
