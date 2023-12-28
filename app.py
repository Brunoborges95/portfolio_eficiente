# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yfin
import plotly.express as px
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

# Data file path
path_df = "C:\\Users\\bruno\\OneDrive\\Documentos\\Machine_learning_projects\\portfolio_eficiente_app\\portfolio_eficiente\\df_stocks_info.parquet"
df_stocks_info = pd.read_parquet(path_df)

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

# Section: Stock Selection
## Allow selecting stock type, sector, and analyst recommendation
st.sidebar.markdown("## Stock Selection")
stock_type = st.multiselect(
    "Select stock type", df_stocks_info["Tipo"].unique(), ["ON"]
)
stock_sector = st.multiselect(
    "Select sector", list(df_stocks_info["Setor"].unique()) + ["All"], ["All"]
)
if "All" in stock_sector:
    stock_sector = df_stocks_info["Setor"].unique()
stock_technical = st.multiselect("Select analyst recommendation",
    df_stocks_info["Mensal"].unique(),
    ["Compra Forte"],
)
df_filter = df_stocks_info.query(
    "Tipo in @stock_type and Setor in @stock_sector and Mensal in @stock_technical"
)
stocks_codes = [i + ".SA" for i in df_filter.Códigos.unique()]

# Section: Optimization Button
## Generate optimization on button press

if st.button('Generate Optimization'):
    historico_stocks = utils.collect_historico_stocks(
        stocks_codes, data_inicio, data_fim
    )
    stocks = list(historico_stocks.columns)
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

    st.session_state['historico_stocks'] = historico_stocks
    st.session_state['w'] = w
    st.session_state['risk_measure'] = risk_measure
