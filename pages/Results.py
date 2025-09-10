import streamlit as st
import opt_tools

# Section: Stock Distribution in Portfolio
## Display the stock distribution graph in the portfolio

# Main title
st.title("Portfolio Analysis: Investment Distribution")

st.markdown(
    """Welcome to our interactive portfolio analysis platform! After running the portfolio optimization step, here you can explore the distribution of your investments across different stocks and adjust the risk level according to your preferences.

Current Portfolio Distribution

In the chart below, we present the current percentage distribution of your investments in various stocks. We use portfolio optimization metrics to provide a clear view of how your assets are allocated.

## Risk Adjustment

We want you to have control over the risk level in your portfolio. Use the slider below to adjust the risk according to your preferences. Notice how the portfolio distribution dynamically adapts to your choices.

**Risk Slider:**
- Minimum: 0
- Maximum: 100
- Step: 5

Feel free to experiment and find the investment distribution that best aligns with your financial goals."""
)

historic_stocks = st.session_state.stock_history
w = st.session_state.w
risk_measure = st.session_state.risk_measure
po = opt_tools.Portfolio_optimization(historic_stocks)
x, y = po.plot_stocks_distribution(risk_measure, w, metric="CVaR")

risk_value = st.slider(
    "Select a value", min_value=0, max_value=100, value=20, step=5
)
po.proportion_risk(risk_value, x, y)
