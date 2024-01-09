# Importando as bibliotecas necessárias
import streamlit as st
from datetime import datetime
import yfinance as yfin
import utils


yfin.pdr_override()

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
    Welcome to the Efficient Portfolio Management app! This tool empowers you to construct an optimal investment portfolio by carefully selecting stocks based on specific criteria and utilizing advanced optimization algorithms.

    ## Stock Selection Criteria

    - **Date Range:** Choose a start and end date to analyze stock data.
    - **Stock Type:** Select the type of stocks (e.g., common, preferred).
    - **Sector:** Filter stocks by sector or explore across all sectors.
    - **Analyst Recommendation:** Tailor your portfolio based on analyst recommendations.

    After setting your criteria, click the 'Generate Optimization' button to proceed.

    ## Optimization Algorithm

    The algorithm uses historical stock data to optimize the portfolio composition. It employs a risk metric, either Value at Risk (VaR) or Conditional Value at Risk (CVaR), to determine the ideal proportion of assets in the portfolio.

    ## Efficient Frontier Graph

    Explore the efficient frontier graph, which illustrates the optimal trade-off between expected return and risk for different portfolio compositions. The algorithm calculates the proportion of stocks at each risk level, providing valuable insights for strategic decision-making.

    Dive into the world of efficient portfolio management and make data-driven investment decisions. Click 'Generate Optimization' to get started!
    """
)

df_stocks_info = utils.read_stocks_info()
# Section: Stock Data Table
## Display a table with stock data
st.markdown("## Stock Data Table")
st.dataframe(df_stocks_info)

# Sidebar for date range selection
st.sidebar.header("Choose Date Range")

# Section: Date Range
## Allow selecting the date range
data_inicio = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
data_fim = st.sidebar.date_input("End Date", datetime.now())

st.sidebar.header("Advanced Transformations")
sma_true = st.sidebar.checkbox("maving average")
if sma_true:
    sma = st.sidebar.slider(
        "Select a number of periods", min_value=0, max_value=100, value=25, step=5
    )

weight_true = st.sidebar.checkbox("Time Weighted")
if weight_true:
    weight = st.sidebar.slider(
        "Select a weight", min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )

pso_opt_true = st.sidebar.checkbox("Optimal points")
if pso_opt_true:
    pso_opt = st.sidebar.slider(
        "Select the number of points", min_value=5, max_value=100, value=10, step=5
    )

# Section: Stock Selection
## Allow selecting stock type, sector, and analyst recommendation
st.markdown("## Stock Selection")
stock_type = st.multiselect(
    "Select stock type", df_stocks_info["Tipo"].unique(), ["ON"]
)
stock_sector = st.multiselect(
    "Select sector", list(df_stocks_info["Setor"].unique()) + ["All"], ["All"]
)
if "All" in stock_sector:
    stock_sector = df_stocks_info["Setor"].unique()
stock_technical = st.multiselect(
    "Select analyst recommendation",
    df_stocks_info["Mensal"].unique(),
    ["Compra Forte"],
)
df_filter = df_stocks_info.query(
    "Tipo in @stock_type and Setor in @stock_sector and Mensal in @stock_technical"
)
stocks_codes = [i + ".SA" for i in df_filter.Códigos.unique()]


# Section: Optimization Button
## Generate optimization on button press

if st.button("Generate Optimization"):
    historico_stocks = utils.collect_historico_stocks(
        stocks_codes, data_inicio, data_fim
    )
    if sma_true:
        historico_stocks = utils.df_moving_avg(historico_stocks, sma=sma)
    if weight_true:
        historico_stocks = utils.df_weighted(historico_stocks, recent_weight=weight)
    if pso_opt_true:
        historico_stocks = utils.df_optimal_pso_points(historico_stocks, pso_opt)
    stocks = list(historico_stocks.columns)
    st.dataframe(historico_stocks)
    po = utils.Portfolio_optimization(historico_stocks)

    # Section: Historical Graph
    ## Display the historical graph of stocks
    st.markdown("## Historical Graph")
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

    st.session_state["historico_stocks"] = historico_stocks
    st.session_state["w"] = w
    st.session_state["risk_measure"] = risk_measure
