# Importando as bibliotecas necessárias
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yfin
import utils


yfin.pdr_override()

st.set_page_config(
    page_title="Backtest",
    page_icon="👋",
)

# Título do aplicativo
st.title("Portfólio Eficiente - Sistema de Backtest")

st.markdown(
    """# Sistema de backtest do portfólio eficiente

Bem -vindo ao sistema eficiente do sistema de backtest! Este aplicativo capacita você a avaliar o desempenho histórico de um portfólio de investimentos otimizado com base em seus critérios selecionados.Explore o impacto de diferentes níveis de risco, períodos de investimento e composições de portfólio em sua estratégia de investimento.

## Seleção e configuração de estoque

- Escolha uma data de início e término para analisar dados históricos de ações.
- Selecione o tipo de ações, setor e recomendações de analistas para adaptar seu portfólio.
- Ajuste os parâmetros como o número de dias antes da data atual, valor investido e valores de risco.

## Executando o backtest

Clique no botão "Backtest" para simular um investimento e visualizar a evolução do valor do seu portfólio ao longo do tempo.O teste considera cenários de risco diferentes e fornece informações sobre como seu portfólio teria se apresentado no passado.

Comece sua exploração agora e tome decisões orientadas a dados para futuros investimentos!"""
)


days_before = st.slider(
    "Dias antes da data atual", min_value=30, max_value=180, value=60, step=10
)
end_date_backtest = datetime.now() - timedelta(days=days_before)
df_stocks_info = utils.read_stocks_info(end_date_backtest)

# Adicionando uma tabela com os dados
st.dataframe(df_stocks_info)

st.sidebar.header("Escolha o intervalo de data")
data_inicio = st.sidebar.date_input("Data de início", datetime(2023, 1, 1))
data_fim = st.sidebar.date_input("Data final", datetime.now())

stock_type = st.multiselect(
    "Selecione Tipo de estoque", list(df_stocks_info["Tipo"].unique())+['Cryptocurrency'], ["Cryptocurrency"]
)
if stock_type=='Cryptocurrency':
    stocks_codes = utils.get_cryptocurrency_codes(num_currencies=500)
else:
    stock_sector = st.multiselect(
        "Selecionar setor", list(df_stocks_info["Setor"].unique()) + ["All"], ["All"]
    )
    if "All" in stock_sector:
        stock_sector = df_stocks_info["Setor"].unique()
    stock_tecnichal = st.multiselect(
        "Selecione a recomendação do analista",
        df_stocks_info["Mensal"].unique(),
        ["Compra Forte"],
    )
    df_filter = df_stocks_info.query(
        "Tipo in @stock_type and Setor in @stock_sector and Mensal in @stock_tecnichal"
    )
    stocks_codes = [i + ".SA" for i in df_filter.Códigos.unique()]

invested_value = st.slider(
    "Valor investido", min_value=1000, max_value=180000, value=5000, step=1000
)
risk_values = st.multiselect("Selecione os valores de risco", range(10, 100, 10), 20)

st.sidebar.header("Transformações avançadas")
sma_true = st.sidebar.checkbox("Média móvel")
if sma_true:
    sma = st.sidebar.slider(
        "Selecione vários períodos", min_value=0, max_value=100, value=25, step=5
    )
else:
    sma = None

weight_true = st.sidebar.checkbox("Tempo ponderado")
if weight_true:
    weight = st.sidebar.slider(
        "Selecione um peso", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )
else:
    weight = None

pso_opt_true = st.sidebar.checkbox("Pontos ideais")
if pso_opt_true:
    pso_opt = st.sidebar.slider(
        "Selecione o número de pontos", min_value=5, max_value=100, value=10, step=5
    )
else:
    pso_opt = None

if st.button("backtest"):
    utils.backtest(
        invested_value,
        stocks_codes,
        risk_values,
        data_inicio,
        end_date_backtest,
        data_fim,
        sma,
        weight,
        pso_opt,
    )
