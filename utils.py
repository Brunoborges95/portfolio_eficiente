# Importando as bibliotecas necessárias
import streamlit as st
import plotly.express as px
import pandas as pd
import pandas_datareader.data as web  # importar dados históricos de ativos
import plotly.express as px  # imagens interativas
import plotly.graph_objects as go
import numpy as np
import numpy.matlib
from scipy.optimize import linprog  # algoritmo para otimização linear
from stqdm import stqdm  # avaliar passar em um loop
from datetime import datetime, timedelta
import boto3

# Function to drop columns with NaN values
def drop_columns_with_nan(df):
    # Identify columns with NaN values
    columns_nan = list(df.iloc[:, list(df.isna().any())].columns)

    # Check if there are columns with NaN values
    if columns_nan != []:
        st.markdown(
            f"<span style='color:red'> Aviso: </span> Os ativos <span style='color:yellow'>{', '.join(columns_nan)}</span> conter dados ausentes e não devem ser usados no portfólio.",
            unsafe_allow_html=True,
        )

        # Drop columns with NaN values
        df_new = df.drop(columns=columns_nan)
    else:
        # If no NaN values are found, return the original dataframe
        return df

    # Return the dataframe without columns with NaN values
    return df_new


def listar_diretorios_s3(bucket, prefix=''):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    caminhos = [f's3://{bucket}/{content.get("Prefix")}' for content in response.get('CommonPrefixes', [])]
    return caminhos


def find_directory_date(caminhos, data_alvo):
    diretorios = [d for d in caminhos] 
    diretorios.sort(reverse=True)
    data_alvo_str = data_alvo.strftime('%Y-%m-%d')
    
    for diretorio in diretorios:
        if diretorio.split('/')[-2] == data_alvo_str:
            return diretorio

    for diretorio in diretorios:
        if diretorio.split('/')[-2] < data_alvo_str:
            return diretorio
    
    if diretorios:
        return diretorios[-1]

    return None


# Data file path
def read_stocks_info(date):
    #date = datetime(2024, 1, 20)
    bucket = 'bbs-datalake'
    prefix = 'SourceZone/stock_info/'
    path = listar_diretorios_s3(bucket, prefix)
    path_df = find_directory_date(path, date)+'df_stocks_info.csv'
    print(path_df)
    df_stocks_info = pd.read_csv(path_df)

    col_str = [
        "Nome",
        "Códigos",
        "Bolsa",
        "Setor",
        "Indústria",
        "15 minutos",
        "Hora",
        "Diário",
        "Semanal",
        "Mensal",
    ]
    for column in df_stocks_info.columns:
        if column not in col_str:
            if (
                df_stocks_info[column].dtype == "O"
            ):  # Verifica se a coluna é do tipo objeto (string)
                df_stocks_info[column] = pd.to_numeric(
                    df_stocks_info[column], errors="coerce"
                )
    Tipos_depara = {
        "34": "BDR",
        "11": "FII",
        "3": "ON",
        "4": "PN",
        "5": "PNA",
        "6": "PNB",
        "31": "BDR",
        "12": "Subscrição",
        "33": "BDR",
        "32": "BDR",
        "35": "BDR",
        "7": "PNC",
        "8": "PND",
    }
    df_stocks_info["Tipo"] = (
        df_stocks_info["Códigos"]
        .str[4:]
        .replace(to_replace="[^0-9]", value="", regex=True)
        .replace(Tipos_depara)
    )
    return df_stocks_info


@st.cache_data
def collect_historico_stocks(stocks_codes, data_inicio, data_fim):
    historico_stocks = web.DataReader(stocks_codes, data_inicio, data_fim)["Adj Close"]
    historico_stocks = drop_columns_with_nan(historico_stocks)
    return historico_stocks


class Portfolio_optimization:
    def __init__(self, historico_stocks):
        self.historico_stocks = historico_stocks
        self.stocks = list(historico_stocks.columns)

    # Function to plot historical stock prices
    def plot_historic(self):
        fig = go.Figure()

        # Iterate through each stock in the portfolio
        for stock in self.stocks:
            # Add a trace for each stock, representing historical closing prices
            fig.add_trace(
                go.Scatter(
                    x=self.historico_stocks.index,
                    y=self.historico_stocks[stock],
                    mode="lines",
                    name=f"{stock}",
                )
            )

        # Update layout with title and axis labels
        fig.update_layout(
            title="Preços de fechamento histórico das ações",
            xaxis_title="Período de tempo",
            yaxis_title="Preço",
        )

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def returns(self):
        """Esta função calcula retornos logarítmicos para cada estoque com base em dados históricos de ações,
        considera um comprimento especificado da janela (nós) e retorna um dicionário contendo os retornos calculados,
        retornos esperados e retornos médios."""
        # Transpose historical stock data for easier calculation
        data = self.historico_stocks.values.transpose()
        data = np.where(data < 0, 1e-07, data)
        # Initialize an array to store logarithmic returns
        Returns = np.zeros((data.shape[1] - 1, data.shape[0]))

        # Calculate logarithmic returns for each stock
        for i in range(Returns.shape[1]):
            Returns[:, i] = np.log(data[i, 1::] / data[i, 0:-1]) + 1

        # Keep only the necessary data based on the specified window length (wE)
        R = Returns[::-1, :]  # Reverse the array
        # R = R[0 : wE + 1, :]  # Slice to include the specified window length

        # Calculate expected returns and mean return
        ExpR = np.mean(R, axis=0).reshape(-1, 1).T  # Expected return
        MeanR = numpy.matlib.repmat(
            ExpR, R.shape[0], 1
        )  # Repeat expected return to match the shape of Returns

        # Return a dictionary containing Returns, Expected Returns, and Mean Return
        return {"Returns": Returns, "Expected Returns": ExpR, "Mean Return": MeanR}

    def optimize(self, Returns, ExpR, a=0.95, h=21, metric="CVaR"):
        """Esta função otimiza um portfólio baseado em uma métrica de risco especificada (VAR ou CVAR)
        usando programação linear.Calcula os valores médios de retorno e o correspondente
        Valores da medição de risco para uma série de possíveis retornos.Os pesos otimizados do portfólio e
        A medida de risco calculada é então retornada como um dicionário."""
        N = ExpR.shape[1]  # Number of assets
        nS = Returns.shape[0]  # Number of scenarios
        V0 = 1  # Initial wealth
        meanR = np.linspace(
            np.min(ExpR), np.max(ExpR), 50
        )  # Mean return values for optimization
        linMap = np.zeros((nS + N + 1, len(meanR)))  # Allocation for linprog output

        # Objective function coefficients
        f = np.vstack((np.zeros((N, 1)), 1, 1 / ((1 - a) * nS) * np.ones((nS, 1))))

        # Coefficients for inequality constraints
        w = np.ones((nS, N)) - Returns
        v = -np.ones((nS, 1))
        y = -np.eye(nS)
        A = np.hstack([w, v, y])
        b = np.zeros((nS, 1))

        # Coefficients for equality constraints
        Aeq = np.vstack(
            [
                np.hstack([ExpR, np.array([0]).reshape(-1, 1), np.zeros((1, nS))]),
                np.hstack(
                    [np.ones((1, N)), np.array([0]).reshape(-1, 1), np.zeros((1, nS))]
                ),
            ]
        )

        # Constraints for boundary values
        lb = np.zeros((1, nS + N + 1))
        ub = np.full((1, nS + N + 1), np.inf)
        bounds = np.vstack([lb, ub]).transpose()

        # Loop through mean return values and solve linear programming problem
        for i in stqdm(range(len(meanR))):
            beq = np.vstack([meanR[i] * V0, V0])
            linMap[:, i] = linprog(f, A, b, Aeq, beq, bounds, method="highs-ds")["x"]

        # Adjust mean return values for display
        meanR = 100 * (meanR - 1) * h

        # Calculate risk measure based on the chosen metric (VaR or CVaR)
        if metric == "VaR":
            v = linMap[N, :]
            w = linMap[0 : ExpR.shape[1], np.argmin(v) : :]
            v = 100 * v * np.sqrt(h)
        elif metric == "CVaR":
            v = sum(linMap[N + 1 :, :]) / ((1 - a) * nS) + linMap[N, :]
            w = linMap[0 : ExpR.shape[1], np.argmin(v) : :]
            v = 100 * v * np.sqrt(h)

        # Return the results as a dictionary
        return {"meanR": meanR, "risk_measure": v, "w": w}

    # Function to plot efficient frontiers based on risk metric (VaR or CVaR)
    def plot_efficient_frontiers(self, values, meanR, metric="CVaR"):
        # Create a Plotly figure with a scatter plot
        fig_1 = go.Figure(data=go.Scatter(x=values[meanR > 0], y=meanR[meanR > 0]))

        # Update layout with title and axis labels
        fig_1.update_layout(
            title=f"{metric} - Fronteira eficiente",
            xaxis_title=f"{metric} (%)",
            yaxis_title="Retorno esperado (%)",
        )

        # Markdown explanation about the efficient frontiers
        st.markdown(
            """Os gráficos abaixo representam o retorno para cada valor do risco considerado.
            A m[etrica escolhida foi o CVaR, por ser uma medida de risco coerente e tambem pelo fato de que, devido à natureza não convexa da otimização do VAR, sua fronteira eficiente da otimização
            exibe um comportamento mais "caótico" em comparação com a otimização do CVAR, que tem um comportamento mais suave."""
        )

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_1, use_container_width=True)

    # Function to plot the distribution of stocks in the portfolio
    def plot_stocks_distribution(self, values, w, metric="CVaR", plot=True):
        # Extract data for x-axis (risk values)
        x = values[np.argmin(values) : :]

        # Extract portfolio allocation data for y-axis
        y = 100 * w

        # Create a Plotly figure
        fig_3 = go.Figure()

        # Iterate through each stock and add a trace for the stock's distribution
        for i in range(len(y)):
            fig_3.add_trace(
                go.Scatter(
                    x=x,
                    y=y[i],
                    mode="lines",
                    line=dict(width=0.5),
                    name=self.stocks[i],
                    stackgroup="one",
                )
            )

        # Update layout with title and axis labels
        fig_3.update_layout(
            title=f"Distribuição de ações no portfólio - {metric}",
            xaxis_title=f"{metric} (%)",
            yaxis_title="Distribuição (%)",
        )

        # Display a markdown explanation if the plot parameter is True
        if plot:
            st.markdown(
                """O gráfico de área abaixo representa a proporção de ativos no portfólio para cada valor considerado de risco."""
            )
            st.plotly_chart(fig_3, use_container_width=True)

        # Return the x and y data
        return x, y

    # Function to visualize the proportion of assets in the portfolio for a given risk value
    def proportion_risk(self, risk_value, x, y, plot=True):
        # Find the index corresponding to the given risk value
        i = np.argmin(abs(x - risk_value))

        # Create a dictionary to store the proportion of each stock at the specified risk value
        risk = {}
        for j, stock in enumerate(self.stocks):
            risk[stock] = round(y[j][i], 2)

        # Convert the dictionary to a pandas Series for easier manipulation
        risk = pd.Series(risk)

        # Create a bar chart using Plotly Express
        risk_2 = risk[risk != 0].sort_values()
        fig_4 = px.bar(risk_2, y=risk_2.index, x=risk_2, orientation="h")

        # Update layout with title and axis labels
        fig_4.update_layout(
            title="Proporção de ativos no portfólio",
            xaxis_title="Proporção",
            yaxis_title="Ações",
        )

        # Display a markdown explanation if the plot parameter is True
        if plot:
            st.markdown(
                """O resultado final é, para um determinado valor de risco,
                        As ações que eu deveria investir em e em que proporção, para o meu portfólio ideal."""
            )
            st.plotly_chart(fig_4, use_container_width=True)

        # Return the proportion of assets for each stock at the specified risk value
        return risk


class Particle:
    def __init__(self, points):
        self.position = np.random.rand(points)
        self.velocity = np.random.rand(points)
        self.best_position = np.copy(self.position)
        self.best_score = float("inf")


class PSO_optimal_points:
    def __init__(self, series, num_points):
        self.series = series
        self.num_points = num_points
        self.score = []

    def _interpolate_nan(self, array_like):
        array = array_like.copy()
        nans = np.isnan(array)

        def get_x(a):
            return a.nonzero()[0]

        array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])
        return array

    def _get_values_at_indices(self, indices, array):
        result = np.where(np.isin(np.arange(len(array)), indices), array, np.nan)
        return result.tolist()

    def _top_n_indices(self, lst, n):
        if n > len(lst):
            raise ValueError(
                "N should be less than or equal to the length of the list."
            )

        indices = np.argsort(lst)[-n:]
        return indices.tolist()

    def objective_function(self, points, original_series):
        if len(points) == 0:
            return float("inf")
        # Função objetivo: diferença entre a série original e os pontos escolhidos
        subset = self._get_values_at_indices(points, original_series)
        interpolated_series = self._interpolate_nan(np.array(subset))
        # Calcula a diferença entre a série original e os pontos interpolados
        score = np.sum(np.abs(original_series - interpolated_series))
        return score

    # Restante do código permanece o mesmo
    def pso(self, num_particles, num_iterations):
        n_series = len(self.series)
        particles = [Particle(n_series) for _ in range(num_particles)]

        global_best_position = None
        global_best_score = float("inf")

        for _ in range(num_iterations):
            for particle in particles:
                points = self._top_n_indices(particle.position, self.num_points)
                score = self.objective_function(points, self.series)

                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = np.copy(particle.position)

            for particle in particles:
                inertia = 0.5
                personal_weight = 2.0
                global_weight = 2.0

                r1, r2 = np.random.rand(n_series), np.random.rand(n_series)
                particle.velocity = (
                    inertia * particle.velocity
                    + personal_weight
                    * r1
                    * (particle.best_position - particle.position)
                    + global_weight * r2 * (global_best_position - particle.position)
                )
                particle.position = np.clip(particle.position + particle.velocity, 0, 1)
            self.score.append(score)
        return self._top_n_indices(global_best_position, self.num_points)


def df_optimal_pso_points(
    df, num_points_to_choose=30, num_particles=15, num_iterations=40
):
    stocks_series_list = []
    for stock in list(df.columns):
        stock_series = df[stock]
        pop = PSO_optimal_points(stock_series, num_points_to_choose)
        chosen_indices = pop.pso(num_particles, num_iterations)
        sorted_v = sorted(range(len(chosen_indices)), key=lambda k: chosen_indices[k])
        sorted_indices = [chosen_indices[i] for i in sorted_v]
        chosen_dates = [stock_series.index[i] for i in sorted_indices]
        stock_series_opt_points = stock_series[chosen_dates]
        stocks_series_list.append(stock_series_opt_points)
        placeholder_df = pd.DataFrame(index=df.index)
    pd_new = (
        pd.concat(stocks_series_list + [placeholder_df], axis=1)
        .sort_values(by="Date")
        .interpolate(limit_direction="both")
    )
    return pd_new


def df_moving_avg(df, sma=25):
    return df.rolling(sma).mean().dropna()


def df_weighted(df, recent_weight=0.8):
    new_df = df.copy()
    for stock in list(df.columns):
        new_df[stock] = (
            np.linspace(1, 1 - recent_weight, len(df[stock]))[::-1] * df[stock]
        )
    return new_df


def backtest(
    valor_investido,
    stocks_codes,
    risk_values,
    start_date_backtest,
    end_date_backtest,
    current_date,
    sma=None,
    weight=None,
    pso_opt=None,
):
    """Esta função executa um backtest em uma estratégia de otimização de portfólio,
    calcular o valor do portfólio ao longo do tempo para diferentes níveis de risco.
    Os resultados são visualizados usando uma figura plopt."""
    # Collect historical stock data for the backtest period
    historico_stocks = collect_historico_stocks(
        stocks_codes, start_date_backtest, current_date
    )

    # Split the data into training and testing sets
    train = historico_stocks[:end_date_backtest]
    test = historico_stocks[end_date_backtest:]

    if sma is not None:
        train = df_moving_avg(train, sma=sma)
    if weight is not None:
        train = df_weighted(train, recent_weight=weight)
    if pso_opt is not None:
        train = df_optimal_pso_points(train, pso_opt)

    # Initialize a Portfolio_optimization object for training data
    po = Portfolio_optimization(train)

    # Calculate returns and optimize portfolio for training data
    r_dict = po.returns()
    Returns = r_dict["Returns"]
    ExpR = r_dict["Expected Returns"]
    opt_dict = po.optimize(Returns, ExpR, a=0.95, h=21, metric="CVaR")
    risk_measure = opt_dict["risk_measure"]
    w = opt_dict["w"]

    # Plot stocks distribution for training data
    x, y = po.plot_stocks_distribution(risk_measure, w, metric="CVaR")

    # Generate portfolios for different risk values
    portfolios = [
        po.proportion_risk(risk_value, x, y, False) for risk_value in risk_values
    ]

    fig = go.Figure()

    # Iterate through each portfolio and perform the backtest
    st.markdown(
        f"<span style='color:green; font-size:larger; font-weight:bold'> Investimento inicial: {valor_investido}</span>",
        unsafe_allow_html=True,
    )

    for portfolio, risco in zip(portfolios, risk_values):
        inv_inicio = test.iloc[0]
        inv_fim = test.iloc[-1]
        a = portfolio / inv_inicio * inv_fim
        valor_atual = sum(a) / 100 * valor_investido

        # Display information about the backtest results
        if valor_atual < valor_investido:
            cor = "red"
        else:
            cor = "green"

        # Exibe o texto formatado com a cor determinada pela condição
        st.markdown(
            f"Para o risco = {risco}, O valor atual do investimento é <span style='color:{cor}; font-size:larger; font-weight:bold'>{round(valor_atual, 2)}</span>",
            unsafe_allow_html=True,
        )

        # Calculate the portfolio value at each time point in the testing set
        valor = {}
        for i in test.index:
            inv_fim = test.loc[i]
            v = sum(portfolio / inv_inicio * inv_fim) / 100 * valor_investido
            valor[i] = v

        # Create a trace for each portfolio in the Plotly figure
        fig.add_trace(
            go.Scatter(
                x=list(valor.keys()),
                y=list(valor.values()),
                mode="lines",
                name=f"Risco = {risco}",
            )
        )

    # Update layout with title and axis labels for the Plotly figure
    fig.update_layout(
        title="Evolução do valor do portfólio",
        xaxis_title="Período de tempo",
        yaxis_title="Valor do portfólio",
    )

    # Display the Plotly figure using Streamlit
    st.plotly_chart(fig, use_container_width=True)
