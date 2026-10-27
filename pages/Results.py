import streamlit as st

import explainer
import opt_tools
import optimizer

st.set_page_config(page_title="Results", page_icon="👋", layout="wide")

# Main title
st.title("Portfolio analysis: investment distribution")

st.markdown(
    """Welcome to the interactive portfolio analysis! After running the optimization on the
    main page, here you can explore how the investments are distributed and adjust the level
    of risk to your preferences.

## Risk adjustment

Use the slider to choose the maximum CVaR: the average loss in the worst months that you
accept. For that level, the optimizer finds the portfolio with the highest expected return.
A negative CVaR means that even the bad months are gains (mostly CDI)."""
)

po = st.session_state.get("po")
if po is None:
    st.info("Run **Generate Optimization** on the main page first.")
    st.stop()

# Section: Stock Distribution in Portfolio
po.plot_stocks_distribution()

lo, hi = 100 * po.frontier["cvar"].min(), 100 * po.frontier["cvar"].max()
if po.settings.objective == "risk_budget":
    default = 100 * po.settings.risk_budget
else:
    default = 100 * po.problem.stats(po.tangency)["cvar"]
default = min(max(default, lo), hi)
risk_value = st.slider(f"Maximum CVaR {po.problem.beta:.0%} over {po.problem.h} trading days (%)",
                       min_value=round(lo, 1), max_value=round(hi, 1), value=round(default, 1), step=0.1)
try:
    w = po.proportion_risk(risk_value, plot=False)
except optimizer.InfeasibleError as e:
    st.error(str(e))
    st.stop()

stats = po.problem.stats(w)
cols = st.columns(5)
cols[0].metric("Expected return (month)", f"{stats['exp_return']:.2%}")
cols[1].metric("Expected return (year)", f"{stats['annual_return']:.2%}")
cols[2].metric("CVaR (month)", f"{stats['cvar']:.2%}")
cols[3].metric("Volatility (year)", f"{stats['annual_vol']:.2%}")
cols[4].metric("Assets", stats["n_assets"])

po.plot_allocation(w, f"Portfolio with CVaR <= {risk_value:.1f}%")

held = w[w > 0].sort_values(ascending=False).rename("weight").to_frame()
info = po.assets.reindex(held.index)
held = held.join(info[[c for c in ["asset_class", "name", "rec_key", "n_analysts", "upside"] if c in info]])
held["expected_return_year"] = po.details["mu"].reindex(held.index) * optimizer.TRADING_DAYS
st.dataframe(held, use_container_width=True, column_config={
    "weight": st.column_config.NumberColumn("Weight", format="percent"),
    "asset_class": "Class", "name": "Name", "rec_key": "Consensus",
    "n_analysts": st.column_config.NumberColumn("Analysts", format="%d"),
    "upside": st.column_config.NumberColumn("Upside", format="percent"),
    "expected_return_year": st.column_config.NumberColumn("Expected return (year)", format="percent"),
})
st.download_button("Download the portfolio (CSV)", held.to_csv().encode("utf-8"), "portfolio.csv", "text/csv")

# Section: AI explanation
st.markdown("## Why these assets? (AI explanation)")
st.caption(
    "Claude writes the text from the facts below, which the app computes (CVM filings, analyst consensus, "
    "the optimizer), and every number it cites is checked against them. It explains the model's choices; "
    "it is not investment advice. Needs ANTHROPIC_API_KEY (environment or .env).")
universe_df, _ = opt_tools.load_universe()
n_held = int((w > 0).sum())
top = st.slider("Holdings to explain (largest first)", 1, max(2, min(20, n_held)), min(10, n_held))
facts_port, facts_hold = explainer.build_facts(
    po.problem, w, po.assets, po.details["mu"], po.settings, po.details, po.historic_stocks,
    universe_df, opt_tools.load_cvm_history(), top)
with st.expander("Facts sent to the model"):
    st.json({"carteira": facts_port, "ativos": facts_hold})
if st.button("Explain with Claude", type="primary"):
    with st.spinner("Claude is writing the explanation..."):
        try:
            st.session_state["explanation"] = (risk_value, top, *explainer.explain(facts_port, facts_hold))
        except explainer.ExplanationError as e:
            st.error(str(e))
explanation = st.session_state.get("explanation")
if explanation is not None and explanation[:2] == (risk_value, top):
    _, _, result, checks, _ = explanation
    st.markdown(result.overview)
    for h in result.holdings:
        st.markdown(f"### {h.symbol}: {h.headline}")
        st.markdown(h.justification)
        st.markdown(f"**Pontos contra.** {h.counterpoints}")
        unverified = checks.get(h.symbol, [])
        if unverified:
            st.warning(f"Numbers not found in the facts sent: {', '.join(unverified)}")
        else:
            st.caption("Every number above was checked against the facts sent.")
elif explanation is not None:
    st.info("The explanation was written for another CVaR level or number of holdings; click the button again.")
