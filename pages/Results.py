import streamlit as st
import utils
# Section: Stock Distribution in Portfolio
## Display the stock distribution graph in the portfolio
st.markdown("## Stock Distribution in Portfolio")
historico_stocks = st.session_state.historico_stocks
w = st.session_state.w
risk_measure = st.session_state.risk_measure
po = utils.Portfolio_optimization(historico_stocks)
x, y = po.plot_stocks_distribution(risk_measure, w, metric="CVaR")

risk_value = st.slider("Select a value", min_value=0, max_value=100, value=20, step=5)
po.proportion_risk(risk_value, x, y)