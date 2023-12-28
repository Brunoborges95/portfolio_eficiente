# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px  # imagens interativas
import utils

st.set_page_config(
    page_title="Backtest",
    page_icon="👋",
)

# Título do aplicativo
st.title("Portfólio Eficiente - Backtest System")

st.markdown(
    """# Efficient Portfolio Backtest System

Welcome to the Efficient Portfolio Backtest System! This application empowers you to evaluate the historical performance of an optimized investment portfolio based on your selected criteria. Explore the impact of different risk levels, investment periods, and portfolio compositions on your investment strategy.

## Stock Selection and Configuration

- Choose a start and end date to analyze historical stock data.
- Select the type of stocks, sector, and analyst recommendations to tailor your portfolio.
- Adjust parameters such as the number of days before the current date, invested value, and risk values.

## Performing the Backtest

Click the "Backtest" button to simulate an investment and visualize the evolution of your portfolio's value over time. The backtest considers different risk scenarios and provides insights into how your portfolio would have performed in the past.

Start your exploration now and make data-driven decisions for future investments!"""
)

path_df = "C:\\Users\\bruno\\OneDrive\\Documentos\\Machine_learning_projects\\portfolio_eficiente_app\\portfolio_eficiente\\df_stocks_info.parquet"
df_stocks_info = pd.read_parquet(path_df)

# Adicionando uma tabela com os dados
st.dataframe(df_stocks_info)

st.sidebar.header("Choose Date Range")
data_inicio = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
data_fim = st.sidebar.date_input("End Date", datetime.now())

stock_type = st.multiselect(
    "Select stock type", df_stocks_info["Tipo"].unique(), ["ON"]
)
stock_sector = st.multiselect(
    "Select sector", list(df_stocks_info["Setor"].unique()) + ["All"], ["All"]
)
if "All" in stock_sector:
    stock_sector = df_stocks_info["Setor"].unique()
stock_tecnichal = st.multiselect(
    "Select analyst recommendation",
    df_stocks_info["Mensal"].unique(),
    ["Compra Forte"],
)
df_filter = df_stocks_info.query(
    "Tipo in @stock_type and Setor in @stock_sector and Mensal in @stock_tecnichal"
)
stocks_codes = [i + ".SA" for i in df_filter.Códigos.unique()]

days_before = st.slider(
    "Days before the current date", min_value=30, max_value=180, value=60, step=10
)
end_date_backtest = datetime.now() - timedelta(days=days_before)
invested_value = st.slider(
    "Invested value", min_value=1000, max_value=180000, value=5000, step=1000
)
risk_values = st.multiselect("Select the risk values", range(10, 100, 10), 20)

if st.button("Backtest"):
    utils.backtest(
        invested_value,
        stocks_codes,
        risk_values,
        data_inicio,
        end_date_backtest,
        data_fim,
    )
