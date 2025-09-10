# Importing necessary libraries
import streamlit as st
from datetime import datetime, timedelta
import opt_tools

st.set_page_config(
    page_title="Backtest",
    page_icon="👋",
)

# App title
st.title("Efficient Portfolio - Backtest System")

st.markdown(
    """# Efficient Portfolio Backtest System

Welcome to the efficient portfolio backtest system! This app empowers you to evaluate the historical performance of an optimized investment portfolio based on your selected criteria. Explore the impact of different risk levels, investment periods, and portfolio compositions on your investment strategy.

## Stock selection and configuration

- Choose a start and end date to analyze historical stock data.
- Select stock type, sector, and analyst recommendations to tailor your portfolio.
- Adjust parameters such as the number of days before the current date, invested value, and risk values.

## Running the backtest

Click the "Backtest" button to simulate an investment and visualize your portfolio value evolution over time. The test considers different risk scenarios and provides insights into how your portfolio would have performed in the past.

Start your exploration now and make data-driven decisions for future investments!
"""
)

days_before = st.slider(
    "Days before current date", min_value=30, max_value=180, value=60, step=10
)
end_date_backtest = datetime.now() - timedelta(days=days_before)
df_stocks_info = opt_tools.read_stocks_info(end_date_backtest)

# Adding a table with the data
st.dataframe(df_stocks_info)

st.sidebar.header("Choose date range")
start_date = st.sidebar.date_input("Start date", datetime(2025, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.now())

stock_type = st.multiselect(
    "Select stock type", list(df_stocks_info["Type"].unique())
)

stock_analyst = st.multiselect(
    "Select analyst recommendation",
    df_stocks_info["Recommendation"].unique(),
    ["strong_buy"],
)
df_filter = df_stocks_info.query(
    "Type in @stock_type and Recommendation in @stock_analyst"
)
stocks_codes = [i + ".SA" for i in df_filter.symbol.unique()]

invested_value = st.slider(
    "Invested value", min_value=1000, max_value=180000, value=5000, step=1000
)
risk_values = st.multiselect("Select risk values", range(10, 100, 10), 20)

st.sidebar.header("Advanced transformations")
sma_true = st.sidebar.checkbox("Moving average")
if sma_true:
    sma = st.sidebar.slider(
        "Select periods", min_value=0, max_value=100, value=25, step=5
    )
else:
    sma = None

weight_true = st.sidebar.checkbox("Time weighted")
if weight_true:
    weight = st.sidebar.slider(
        "Select a weight", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )
else:
    weight = None

pso_opt_true = st.sidebar.checkbox("Optimal points")
if pso_opt_true:
    pso_opt = st.sidebar.slider(
        "Select number of points", min_value=5, max_value=100, value=10, step=5
    )
else:
    pso_opt = None

if st.button("Backtest"):
    opt_tools.backtest(
        invested_value,
        stocks_codes,
        risk_values,
        start_date,
        end_date_backtest,
        end_date,
        sma,
        weight,
        pso_opt,
    )
