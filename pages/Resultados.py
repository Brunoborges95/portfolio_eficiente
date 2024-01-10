import streamlit as st
import utils

# Section: Stock Distribution in Portfolio
## Display the stock distribution graph in the portfolio


# Main title
st.title("Análise de portfólio: distribuição de investimentos")

st.markdown(
    """Bem -vindo à nossa plataforma interativa de análise de portfólio! Após executar a etapa de otimização do portfólio,  aqui você pode explorar a distribuição de seus investimentos em diferentes ações e ajustar o nível de risco de acordo com suas preferências.

Distribuição atual do portfólio

No gráfico abaixo, apresentamos a distribuição percentual atual de seus investimentos em várias ações. Utilizamos métricas de otimização de portfólio para fornecer uma visão clara de como seus ativos são alocados.

## Ajuste de risco

Queremos que você tenha controle sobre o nível de risco em seu portfólio. Use o controle deslizante abaixo para ajustar o risco de acordo com suas preferências.Observe como a distribuição do portfólio se adapta dinamicamente às suas escolhas.

**Slider de risco:**
- Mínimo: 0
- Máximo: 100
- Incremento: 5

Sinta -se à vontade para experimentar e encontrar a distribuição de investimentos que se alinha melhor com seus objetivos financeiros."""
)


historico_stocks = st.session_state.historico_stocks
w = st.session_state.w
risk_measure = st.session_state.risk_measure
po = utils.Portfolio_optimization(historico_stocks)
x, y = po.plot_stocks_distribution(risk_measure, w, metric="CVaR")

risk_value = st.slider(
    "Selecione um valor", min_value=0, max_value=100, value=20, step=5
)
po.proportion_risk(risk_value, x, y)
