# Importando as bibliotecas necessárias
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime
import pandas_datareader.data as web #importar dados históricos de ativos
import yfinance as yfin
import plotly.express as px #imagens interativas
import plotly.graph_objects as go
import numpy as np
import numpy.matlib
from scipy.optimize import linprog #algoritmo para otimização linear
from time import sleep 
from tqdm.notebook import tqdm #avaliar passar em um loop

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

yfin.pdr_override()

# Título do aplicativo
st.title('Portfólio Eficiente')
         
# Criando um DataFrame fictício para o exemplo
df_stocks_info = pd.read_parquet('df_stocks_info.parquet')

# Adicionando uma tabela com os dados
st.dataframe(df_stocks_info)

st.sidebar.header('Escolha o Intervalo de Datas')
data_inicio = st.sidebar.date_input('Data Inicial', datetime(2020, 1, 1))
data_fim = st.sidebar.date_input('Data Final', datetime.now())

stock_type = st.multiselect('Selecione o tipo de ação', df_stocks_info['Tipo'].unique())
stock_sector = st.multiselect('Selecione o setor', df_stocks_info['Setor'].unique())
stock_tecnichal = st.multiselect('Selecione a recomendação dos analistas', df_stocks_info['Mensal'].unique())

st.button('Gerar base histórica', on_click=click_button)
if st.session_state.clicked:
    df_filter = df_stocks_info.query('Tipo in @stock_type and Setor in @stock_sector and Mensal in @stock_tecnichal')
    stocks = [i+'.SA' for i in df_filter.Códigos.unique()]
    historico_stocks = web.DataReader(stocks, data_inicio, data_fim)['Adj Close']

    #colunas que contém nan
    colunas_nan = list(historico_stocks.iloc[:,list(historico_stocks.isna().any())].columns)
    if colunas_nan!=[]:
        print(f'Os ativos {colunas_nan} contém dados faltantes, e portante não devem ser utilizadas no portfólio')
        historico_stocks = historico_stocks.drop(columns=colunas_nan)
        stocks = list(historico_stocks.columns)
    else:
        stocks = list(historico_stocks.columns)

    fig = go.Figure()
    for stock in stocks:
        fig.add_trace(go.Scatter(x=historico_stocks.index, y=historico_stocks[stock],
                                        mode='lines',
                                        name=f'{stock}'))
    fig.update_layout(title='Histórico de fechamento das ações ', 
                        xaxis_title='Período',
                        yaxis_title='Preço')
    st.plotly_chart(fig, use_container_width=True)

    # Adicionando uma seção de texto
    st.header('Aqui está um exemplo de seção de texto')

    if st.button('Gerar otimização'):
        data = historico_stocks.values.transpose()
        Returns = np.zeros((data.shape[1]-1,data.shape[0]))
        for i in range(Returns.shape[1]):
            Returns[:,i] = np.log(data[i,1::]/data[i,0:-1])+1
        Returns = Returns[:,:-1]

        wE = 221 ; #Janela de estimativa
        R = Returns[::-1,:] #(a:k:b) a = primeiro indice, k = step size, b = último
        R = R[0:wE+1,:]

        ExpR = np.mean(R, axis=0).reshape(-1,1).T #retorno esperado
        MeanR = numpy.matlib.repmat(ExpR,wE+1,1)

        N = ExpR.shape[1] #número de ativos

        nS = Returns.shape[0] #número de cenários
        V0 = 1 #Initial wealth
        a = 0.95 #Confidence level
        h = 21 #Número de day trades por mês
        meanR = np.linspace(np.min(ExpR),np.max(ExpR),50)
        #Allocation for linprog output
        linMap = np.zeros((nS+N+1,len(meanR)))
        f = np.vstack((np.zeros((N,1)),1,1/((1-a)*nS)*np.ones((nS,1))))
        #Coeficientes da restrição de desigualdade
        w = np.ones((nS,N)) - Returns
        v = -np.ones((nS,1))
        y = -np.eye(nS)
        A = np.hstack([w,v,y])
        b = np.zeros((nS,1))
        #Restrições do coeficiente de igualdade
        Aeq = np.vstack([np.hstack([ExpR,np.array([0]).reshape(-1,1),np.zeros((1,nS))]),np.hstack([np.ones((1,N)),np.array([0]).reshape(-1,1),np.zeros((1,nS))])])
        #Restrições de fronteira
        lb = np.zeros((1,nS+N+1))
        ub = np.full((1,nS+N+1), np.inf)
        bounds = np.vstack([lb,ub]).transpose()
            
        for i in tqdm(range(len(meanR))):
            beq = np.vstack([meanR[i]*V0,V0])
            linMap[:,i] = linprog(f,A,b,Aeq,beq,bounds, method='revised simplex')['x']          
        VaR = linMap[N,:]
        CVaR = sum(linMap[N+1::,:])/((1-a)*nS) + linMap[N,:]
        w = linMap[0:ExpR.shape[1],np.argmin(CVaR)::]
        meanR = 100*(meanR-1)*h
        VaR = 100*VaR*np.sqrt(h)
        CVaR = 100*CVaR*np.sqrt(h)

        fig_1 = go.Figure(data=go.Scatter(x=VaR[meanR>0], y=meanR[meanR>0]))
        fig_1.update_layout(title='Value at Risk - Fronteira Eficiente', 
            xaxis_title='Value at Risk (%)',
            yaxis_title='Retorno Esperado (%)')
        st.plotly_chart(fig_1, use_container_width=True)

        fig_2 = go.Figure(data=go.Scatter(x=CVaR[meanR>0], y=meanR[meanR>0]))
        fig_2.update_layout(title='Conditional Value at Risk - Fronteira Eficiente', 
                            xaxis_title='Conditional Value at Risk (%)',
                            yaxis_title='Retorno Esperado (%)')
        st.plotly_chart(fig_2, use_container_width=True)


        x=CVaR[np.argmin(CVaR)::]
        y = 100*w

        fig_3 = go.Figure()
        for i in range(len(y)):
            fig_3.add_trace(go.Scatter(
                            x=x, y=y[i],
                            mode='lines',
                            line=dict(width=0.5),
                            name=stocks[i],
                            stackgroup='one'))
        fig_3.update_layout(title='Distribuição de Ações na carteira - CVAR', 
                            xaxis_title='Conditional Value at Risk (%)',
                            yaxis_title='Distribuição (%)')
        st.plotly_chart(fig_3, use_container_width=True)


        def proportion_CVAR(CVAR_value, CVaR_portfolio, y, stocks):
            i = np.argmin(abs(CVaR_portfolio-CVAR_value))
            CVAR = {}
            for j, stock in zip(range(len(stocks)-1), stocks):
                CVAR[stock] = round(y[j][i],2)
            CVAR = pd.Series(CVAR)
            CVAR = CVAR[CVAR!=0].sort_values()
            fig_4 = px.bar(CVAR, y=CVAR.index,x=CVAR, orientation='h')
            fig_4.update_layout(
            title='Proporção de ativos na carteira',
            xaxis_title='Proporção',
            yaxis_title= 'Ações')
            st.plotly_chart(fig_4, use_container_width=True)
            return CVAR, fig

        proportion_CVAR(20, x, y, stocks)[1]