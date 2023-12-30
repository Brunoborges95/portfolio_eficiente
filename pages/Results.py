import streamlit as st
import utils

# Section: Stock Distribution in Portfolio
## Display the stock distribution graph in the portfolio


# Main title
st.title("Portfolio Analysis: Investment Distribution")

st.markdown(
    """Welcome to our interactive portfolio analysis platform! Here, you can explore the distribution of your investments across different stocks and fine-tune the level of risk according to your preferences.

## Current Portfolio Distribution

In the chart below, we present the current percentage distribution of your investments in various stocks. We use portfolio optimization metrics to provide a clear view of how your assets are allocated.

## Risk Adjustment

We want you to have control over the risk level in your portfolio. Use the slider below to adjust the risk according to your preferences. Observe how the portfolio distribution dynamically adapts to your choices.

**Risk Slider:**
- Minimum: 0
- Maximum: 100
- Increment: 5

Feel free to experiment and find the investment distribution that aligns best with your financial goals. Happy analyzing!"""
)


historico_stocks = st.session_state.historico_stocks
w = st.session_state.w
risk_measure = st.session_state.risk_measure
po = utils.Portfolio_optimization(historico_stocks)
x, y = po.plot_stocks_distribution(risk_measure, w, metric="CVaR")

risk_value = st.slider("Select a value", min_value=0, max_value=100, value=20, step=5)
po.proportion_risk(risk_value, x, y)
