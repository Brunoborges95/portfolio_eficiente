# Importing necessary libraries
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtesting
import opt_tools
import universe

st.set_page_config(page_title="Backtest", page_icon="👋", layout="wide")

# App title
st.title("Efficient Portfolio - Backtest System")

st.markdown(
    """Evaluate how the strategy in the sidebar would have done in the past, against the CDI,
the Ibovespa and the equal-weight (1/N) portfolio of the same assets.

## How the backtest works (walk-forward)

- On each rebalance date the portfolio is rebuilt with the data available **on that date
  only**: the previous *lookback* days of prices, the latest universe snapshot published
  before it (liquidity, analyst consensus) and that day's Focus survey.
- Liquidity filters use the traded value of the 3 months before each rebalance: an asset that
  is liquid today because it rallied was often illiquid before.
- Between rebalances the weights drift with prices; every trade pays the transaction cost.

**Caveats.** Rebalances older than the first snapshot have no analyst data from their time:
they use the earliest snapshot, which is look-ahead (reported in the *Point in time* column).
And the candidate assets are today's listed ones, so delisted losers are missing
(survivorship bias). Run `python build_dataset.py` regularly to accumulate snapshots.
"""
)

df_stocks_info, snapshot = opt_tools.load_universe()
opt_tools.snapshot_notice(snapshot)

st.sidebar.header("Backtest period")
start_date = st.sidebar.date_input("First rebalance", date.today() - timedelta(days=730))
end_date = st.sidebar.date_input("End date", date.today())
lookback = st.sidebar.slider("Lookback (trading days of history per decision)", 126, 756, 504, 21,
                             help="More history gives more scenarios per asset: with one year, the optimizer "
                                  "overfits the few worst days when there are many assets.")
rebalance = st.sidebar.selectbox("Rebalance every", [21, 63, 126], format_func=lambda d: {21: "month", 63: "quarter", 126: "semester"}[d])
cost = st.sidebar.slider("Transaction cost (% of the traded value)", 0.0, 1.0, 0.1, 0.05) / 100
pin = st.sidebar.checkbox(
    "Use today's snapshot on every date", False,
    help="Extends the analyst data to the whole period, at the cost of look-ahead: today's ratings "
         "were not known in the past. Prefer it only with the analyst filter disabled.")

rules = opt_tools.universe_controls(df_stocks_info, key="bt")
settings = opt_tools.model_controls(rules.classes, key="bt")

st.sidebar.header("Compare with")
compare = st.sidebar.multiselect("Benchmarks on the same universe", ["Equal weight (1/N)", "Original model (CVaR 10%)"],
                                 ["Equal weight (1/N)"])
invested_value = st.number_input("Invested value (R$)", min_value=1000, max_value=10_000_000, value=5000, step=1000)

if st.button("Backtest", type="primary"):
    pinned = universe.list_snapshots()[-1] if pin else None
    strategies = [backtesting.Strategy("Your strategy", rules, settings, snapshot=pinned)]
    if "Equal weight (1/N)" in compare:
        strategies.append(backtesting.Strategy("Equal weight (1/N)", rules, kind="equal", snapshot=pinned))
    if "Original model (CVaR 10%)" in compare:
        strategies.append(backtesting.Strategy("Original model (CVaR 10%)", rules, kind="legacy", snapshot=pinned))
    with st.status("Running the backtest...", expanded=True) as status:
        try:
            st.session_state["backtest"] = backtesting.run_suite(
                strategies, start_date, end_date, lookback, rebalance, cost, log=st.write)
        except (ValueError, KeyError) as e:
            status.update(label="Backtest failed", state="error")
            st.error(str(e))
            st.stop()
        status.update(label="Backtest done", state="complete", expanded=False)

out = st.session_state.get("backtest")
if out is not None:
    values = pd.DataFrame(out["values"]).ffill()
    final = values.iloc[-1] * invested_value
    st.markdown(
        f"<span style='color:green; font-size:larger; font-weight:bold'> Initial investment: R$ {invested_value:,.0f}</span>",
        unsafe_allow_html=True)
    for name, v in final.items():
        color = "green" if v >= invested_value else "red"
        st.markdown(f"{name}: final value <span style='color:{color}; font-size:larger; font-weight:bold'>"
                    f"R$ {v:,.2f}</span>", unsafe_allow_html=True)

    fig = go.Figure([go.Scatter(x=values.index, y=values[c] * invested_value, mode="lines", name=c) for c in values])
    fig.update_layout(title="Portfolio value evolution", xaxis_title="Time period", yaxis_title="Portfolio value (R$)")
    st.plotly_chart(fig, use_container_width=True)

    drawdown = values / values.cummax() - 1
    fig_dd = go.Figure([go.Scatter(x=drawdown.index, y=100 * drawdown[c], mode="lines", name=c) for c in drawdown])
    fig_dd.update_layout(title="Drawdown (fall from the previous peak)", xaxis_title="Time period", yaxis_title="%")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("## Metrics")
    metrics = out["metrics"]
    pct = [c for c in metrics.columns if c not in ("Sharpe (vs CDI)",)]
    st.dataframe(metrics, use_container_width=True, column_config={
        **{c: st.column_config.NumberColumn(c, format="percent") for c in pct},
        "Sharpe (vs CDI)": st.column_config.NumberColumn("Sharpe (vs CDI)", format="%.2f"),
    })
    st.caption("Ex-ante CVaR is what the model expected at each rebalance; Realized CVaR is the same "
               "measure on the strategy's actual daily returns. Realized well above ex-ante means "
               "the model underestimated the risk.")
    pit = out["point_in_time"].get("Your strategy", 1.0)
    if pit < 1:
        st.warning(f"{1 - pit:.0%} of the rebalances used a universe snapshot from after their date (look-ahead).")

    result = out["results"]["Your strategy"]
    if result["errors"]:
        st.warning(f"{len(result['errors'])} rebalances could not be solved and kept the previous portfolio.")
    weights = 100 * result["weights"]
    classes = df_stocks_info.set_index("symbol")["asset_class"]
    by_class = weights.T.groupby(classes.reindex(weights.columns).fillna("Other")).sum().T
    fig_w = go.Figure([go.Scatter(x=by_class.index, y=by_class[c], mode="lines", name=c, stackgroup="one",
                                  line=dict(width=0.5, color=opt_tools.CLASS_COLORS.get(c))) for c in by_class])
    fig_w.update_layout(title="Your strategy: allocation by asset class at each rebalance",
                        xaxis_title="Rebalance date", yaxis_title="Weight (%)")
    st.plotly_chart(fig_w, use_container_width=True)
    with st.expander("Weights at each rebalance"):
        st.dataframe(weights.round(2), use_container_width=True)
