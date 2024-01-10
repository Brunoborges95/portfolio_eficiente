# Importando as bibliotecas necessárias
import streamlit as st
from datetime import datetime
import yfinance as yfin
import utils


yfin.pdr_override()

# Page settings
st.set_page_config(
    page_title="Otimização",
    page_icon="👋",
)

# Main title
st.title("Gestão Eficiente de Portfólio")

# Introduction Section
st.markdown(
    """
    Bem -vindo ao aplicativo de gerenciamento de portfólio eficiente! Essa ferramenta capacita você a construir um portfólio ideal de investimento, selecionando cuidadosamente as ações com base em critérios específicos e utilizando algoritmos de otimização avançada.

    ## Critérios de seleção de estoque

    - **DATA DE DATA:** Escolha uma data de início e término para analisar dados de ações.
    - **Tipo de estoque:** Selecione o tipo de estoque (por exemplo, comum, preferido).
    - **Setor:** Filtrar estoques por setor ou explorar em todos os setores.
    - **Recomendação do analista:** adapta seu portfólio com base nas recomendações dos analistas.

    Depois de definir seus critérios, clique no botão 'Gerar otimização' para prosseguir.

    ## Algoritmo de otimização

    O algoritmo usa dados históricos de ações para otimizar a composição do portfólio. Emprega uma métrica de risco, valor em risco (VAR) ou valor condicional em risco (CVAR), para determinar a proporção ideal de ativos no portfólio.

    ## Gráfico de fronteira eficiente

    Explore o gráfico de fronteira eficiente, que ilustra o trade-off ideal entre o retorno esperado e o risco de diferentes composições de portfólio. O algoritmo calcula a proporção de ações em cada nível de risco, fornecendo informações valiosas para a tomada de decisões estratégicas.

    Mergulhe no mundo da gestão eficiente do portfólio e tome decisões de investimento orientadas a dados. Clique em 'Gere a otimização' para começar!
    """
)

df_stocks_info = utils.read_stocks_info()
# Section: Stock Data Table
## Display a table with stock data
st.markdown("## Tabela de dados de estoque")
st.dataframe(df_stocks_info)

# Sidebar for date range selection
st.sidebar.header("Escolha o intervalo de data")

# Section: Date Range
## Allow selecting the date range
data_inicio = st.sidebar.date_input("Data de início", datetime(2023, 1, 1))
data_fim = st.sidebar.date_input("Data final", datetime.now())

st.sidebar.header("Transformações avançadas")
sma_true = st.sidebar.checkbox("Média móvel")
if sma_true:
    sma = st.sidebar.slider(
        "Selecione vários períodos", min_value=0, max_value=100, value=25, step=5
    )

weight_true = st.sidebar.checkbox("Tempo ponderado")
if weight_true:
    weight = st.sidebar.slider(
        "Selecione um peso", min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )

pso_opt_true = st.sidebar.checkbox("Pontos ideais")
if pso_opt_true:
    pso_opt = st.sidebar.slider(
        "Selecione o número de pontos ", min_value=5, max_value=100, value=10, step=5
    )

# Section: Stock Selection
## Allow selecting stock type, sector, and analyst recommendation
st.markdown("## Seleção de estoque")
stock_type = st.multiselect(
    "Selecione Tipo de estoque", df_stocks_info["Tipo"].unique(), ["ON"]
)
stock_sector = st.multiselect(
    "Selecionar setor", list(df_stocks_info["Setor"].unique()) + ["All"], ["All"]
)
if "All" in stock_sector:
    stock_sector = df_stocks_info["Setor"].unique()
stock_technical = st.multiselect(
    "Selecione a recomendação do analista",
    df_stocks_info["Mensal"].unique(),
    ["Compra Forte"],
)
df_filter = df_stocks_info.query(
    "Tipo in @stock_type and Setor in @stock_sector and Mensal in @stock_technical"
)
stocks_codes = [i + ".SA" for i in df_filter.Códigos.unique()]


# Section: Optimization Button
## Generate optimization on button press

if st.button("Gerar otimização"):
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
    st.markdown("## Gráfico histórico")
    po.plot_historic()

    r_dict = po.returns()
    Returns = r_dict["Returns"]
    ExpR = r_dict["Expected Returns"]

    # Section: Portfolio Optimization
    ## Display optimization results
    st.markdown("## Otimização do portfólio")
    opt_dict = po.optimize(Returns, ExpR)
    meanR = opt_dict["meanR"]
    risk_measure = opt_dict["risk_measure"]
    w = opt_dict["w"]

    # Section: Efficient Frontier
    ## Display the efficient frontier based on risk metric
    st.markdown("## Fronteira eficiente")
    po.plot_efficient_frontiers(risk_measure, meanR, metric="CVaR")

    st.session_state["historico_stocks"] = historico_stocks
    st.session_state["w"] = w
    st.session_state["risk_measure"] = risk_measure

    st.markdown("#### Na aba Resultados, você pode encontrar a carteira de ativos ideal para cada nível de risco.")
