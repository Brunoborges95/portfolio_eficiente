# Import necessary libraries
import streamlit as st
from datetime import datetime
import opt_tools

# Page settings
st.set_page_config(
    page_title="Optimization",
    page_icon="👋",
)

# Main title
st.title("Efficient Portfolio Management")

# Introduction Section
st.markdown(
    """
    Welcome to the efficient portfolio management app! This tool empowers you to build an ideal investment portfolio by carefully selecting stocks based on specific criteria and using advanced optimization algorithms.

    ## Stock Selection Criteria

    - **Date Range:** Choose a start and end date to analyze stock data.
    - **Stock Type:** Select the type of stock (e.g., common, preferred).
    - **Sector:** Filter stocks by sector or explore all sectors.
    - **Analyst Recommendation:** Tailor your portfolio based on analyst recommendations.

    After setting your criteria, click the 'Generate Optimization' button to proceed.

    ## Optimization Algorithm

    The algorithm uses historical stock data to optimize the portfolio composition. It employs a risk metric, Value at Risk (VaR) or Conditional Value at Risk (CVaR), to determine the ideal proportion of assets in the portfolio.

    ## Efficient Frontier Chart

    Explore the efficient frontier chart, which illustrates the optimal trade-off between expected return and risk for different portfolio compositions. The algorithm calculates the stock proportions at each risk level, providing valuable insights for strategic decision-making.

    Dive into the world of efficient portfolio management and make data-driven investment decisions. Click 'Generate Optimization' to get started!
    """
)

df_stocks_info = opt_tools.read_stocks_info(datetime.now())
# Section: Stock Data Table
## Display a table with stock data
st.markdown("## Stock Data Table")
st.dataframe(df_stocks_info)

# Sidebar for date range selection
st.sidebar.header("Choose Date Range")

# Section: Date Range
## Allow selecting the date range
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

st.sidebar.header("Advanced Transformations")
sma_enabled = st.sidebar.checkbox("Moving Average")
if sma_enabled:
    sma_period = st.sidebar.slider(
        "Select Moving Average Period", min_value=0, max_value=100, value=25, step=5
    )

weight_enabled = st.sidebar.checkbox("Time Weighted")
if weight_enabled:
    weight = st.sidebar.slider(
        "Select Weight", min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )

pso_opt_enabled = st.sidebar.checkbox("Optimal Points")
if pso_opt_enabled:
    pso_opt = st.sidebar.slider(
        "Select Number of Points", min_value=5, max_value=100, value=10, step=5
    )

# Section: Stock Selection
## Allow selecting stock type, sector, and analyst recommendation
st.markdown("## Stock Selection")
stock_type = st.multiselect(
    "Select Stock Type", list(df_stocks_info["Type"].unique())
)

stock_recommendation = st.multiselect(
    "Select Analyst Recommendation",
    df_stocks_info["Recommendation"].unique(),
    ["strong_buy"],
)
df_filter = df_stocks_info.query(
    "Type in @stock_type and Recommendation in @stock_recommendation"
)
stock_codes = [i + ".SA" for i in df_filter.symbol.unique()]

# Section: Optimization Button
## Generate optimization on button press

if st.button("Generate Optimization"):
    stock_history = opt_tools.collect_historic_stocks(
        stock_codes, start_date, end_date
    )
    if sma_enabled:
        stock_history = opt_tools.df_moving_avg(stock_history, sma=sma_period)
    if weight_enabled:
        stock_history = opt_tools.df_weighted(stock_history, recent_weight=weight)
    if pso_opt_enabled:
        stock_history = opt_tools.df_optimal_pso_points(stock_history, pso_opt)
    stocks = list(stock_history.columns)
    st.dataframe(stock_history)
    po = opt_tools.Portfolio_optimization(stock_history)

    # Section: Historical Graph
    ## Display the historical graph of stocks
    st.markdown("## Historical Chart")
    po.plot_historic()

    r_dict = po.returns()
    Returns = r_dict["Returns"]
    ExpR = r_dict["Expected Returns"]

    # Section: Portfolio Optimization
    ## Display optimization results
    st.markdown("## Portfolio Optimization")
    opt_dict = po.optimize(Returns, ExpR)
    meanR = opt_dict["meanR"]
    risk_measure = opt_dict["risk_measure"]
    w = opt_dict["w"]

    # Section: Efficient Frontier
    ## Display the efficient frontier based on risk metric
    st.markdown("## Efficient Frontier")
    po.plot_efficient_frontiers(risk_measure, meanR, metric="CVaR")

    st.session_state["stock_history"] = stock_history
    st.session_state["w"] = w
    st.session_state["risk_measure"] = risk_measure

    st.markdown("#### In the Results tab, you can find the ideal asset portfolio for each risk level.")
