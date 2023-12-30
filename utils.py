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


# Function to drop columns with NaN values
def drop_columns_with_nan(df):
    # Identify columns with NaN values
    columns_nan = list(df.iloc[:, list(df.isna().any())].columns)

    # Check if there are columns with NaN values
    if columns_nan != []:
        st.write(
            f"The assets {columns_nan} contain missing data and should not be used in the portfolio."
        )
        
        # Drop columns with NaN values
        df_new = df.drop(columns=columns_nan)
    else:
        # If no NaN values are found, return the original dataframe
        return df
    
    # Return the dataframe without columns with NaN values
    return df_new



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
            title="Historical Closing Prices of Stocks",
            xaxis_title="Time Period",
            yaxis_title="Price",
        )

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def returns(self, wE=221):
        '''This function calculates logarithmic returns for each stock based on historical stock data, 
        considers a specified window length (wE), and returns a dictionary containing the calculated returns, 
        expected returns, and mean returns.'''
        # Transpose historical stock data for easier calculation
        data = self.historico_stocks.values.transpose()

        # Initialize an array to store logarithmic returns
        Returns = np.zeros((data.shape[1] - 1, data.shape[0]))

        # Calculate logarithmic returns for each stock
        for i in range(Returns.shape[1]):
            Returns[:, i] = np.log(data[i, 1::] / data[i, 0:-1]) + 1

        # Keep only the necessary data based on the specified window length (wE)
        R = Returns[::-1, :]  # Reverse the array
        R = R[0 : wE + 1, :]  # Slice to include the specified window length

        # Calculate expected returns and mean return
        ExpR = np.mean(R, axis=0).reshape(-1, 1).T  # Expected return
        MeanR = numpy.matlib.repmat(ExpR, wE + 1, 1)  # Repeat expected return to match the shape of Returns

        # Return a dictionary containing Returns, Expected Returns, and Mean Return
        return {"Returns": Returns, "Expected Returns": ExpR, "Mean Return": MeanR}
    

    def optimize(self, Returns, ExpR, a=0.95, h=21, metric="CVaR"):
        '''This function optimizes a portfolio based on a specified risk metric (VaR or CVaR) 
        using linear programming. It calculates the mean return values and the corresponding 
        risk measure values for a range of possible returns. The optimized portfolio weights and 
        the calculated risk measure are then returned as a dictionary.'''
        N = ExpR.shape[1]  # Number of assets
        nS = Returns.shape[0]  # Number of scenarios
        V0 = 1  # Initial wealth
        meanR = np.linspace(np.min(ExpR), np.max(ExpR), 50)  # Mean return values for optimization
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
                np.hstack([np.ones((1, N)), np.array([0]).reshape(-1, 1), np.zeros((1, nS))]),
            ]
        )

        # Constraints for boundary values
        lb = np.zeros((1, nS + N + 1))
        ub = np.full((1, nS + N + 1), np.inf)
        bounds = np.vstack([lb, ub]).transpose()

        # Loop through mean return values and solve linear programming problem
        for i in stqdm(range(len(meanR))):
            beq = np.vstack([meanR[i] * V0, V0])
            linMap[:, i] = linprog(f, A, b, Aeq, beq, bounds, method="revised simplex")["x"]

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
        return {"meanR": meanR, "risk_measure": v, 'w':w}

    # Function to plot efficient frontiers based on risk metric (VaR or CVaR)
    def plot_efficient_frontiers(self, values, meanR, metric="CVaR"):
        # Create a Plotly figure with a scatter plot
        fig_1 = go.Figure(data=go.Scatter(x=values[meanR > 0], y=meanR[meanR > 0]))

        # Update layout with title and axis labels
        fig_1.update_layout(
            title=f"{metric} - Efficient Frontier",
            xaxis_title=f"{metric} (%)",
            yaxis_title="Expected Return (%)",
        )

        # Markdown explanation about the efficient frontiers
        st.markdown(
            """The charts below represent the return for each value of the considered risk. 
            Due to the non-convex nature of VaR optimization, the efficient frontier of VaR optimization 
            exhibits a more "chaotic" behavior compared to CVaR optimization, which has a smoother behavior."""
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
            title=f"Stocks Distribution in Portfolio - {metric}",
            xaxis_title=f"{metric} (%)",
            yaxis_title="Distribution (%)",
        )

        # Display a markdown explanation if the plot parameter is True
        if plot:
            st.markdown('''The area chart below represents the proportion of assets in 
                        the portfolio for each considered risk value.''')
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
            title="Proportion of Assets in Portfolio",
            xaxis_title="Proportion",
            yaxis_title="Stocks",
        )

        # Display a markdown explanation if the plot parameter is True
        if plot:
            st.markdown('''The final result is, for a given risk value, 
                        the stocks I should invest in and in what proportion, for my optimal portfolio.''')
            st.plotly_chart(fig_4, use_container_width=True)

        # Return the proportion of assets for each stock at the specified risk value
        return risk



def backtest(
    valor_investido,
    stocks_codes,
    risk_values,
    start_date_backtest,
    end_date_backtest,
    current_date,
):
    '''This function performs a backtest on a portfolio optimization strategy, 
    calculating the value of the portfolio over time for different levels of risk. 
    The results are visualized using a Plotly figure.'''
    # Collect historical stock data for the backtest period
    historico_stocks = collect_historico_stocks(
        stocks_codes, start_date_backtest, current_date
    )

    # Split the data into training and testing sets
    train = historico_stocks[:end_date_backtest]
    test = historico_stocks[end_date_backtest:]

    # Extract stock names from the historical data
    stocks = list(historico_stocks.columns)

    # Initialize a Portfolio_optimization object for training data
    po = Portfolio_optimization(train, stocks)

    # Calculate returns and optimize portfolio for training data
    r_dict = po.returns()
    Returns = r_dict["Returns"]
    ExpR = r_dict["Expected Returns"]
    opt_dict = po.optimize(Returns, ExpR, a=0.95, h=21, metric="CVaR")
    risk_measure = opt_dict["risk_measure"]

    # Plot stocks distribution for training data
    x, y = po.plot_stocks_distribution(risk_measure, metric="CVaR")

    # Generate portfolios for different risk values
    portfolios = [
        po.proportion_risk(risk_value, x, y, False) for risk_value in risk_values
    ]

    fig = go.Figure()

    # Iterate through each portfolio and perform the backtest
    for portfolio, risco in zip(portfolios, risk_values):
        inv_inicio = test.iloc[0]
        inv_fim = test.iloc[-1]
        a = portfolio / inv_inicio * inv_fim
        valor_atual = sum(a) / 100 * valor_investido

        # Display information about the backtest results
        st.write(f"Initial Investment: {valor_investido}")
        st.write(
            f"For risk = {risco}, the current value of the investment is {round(valor_atual, 2)}"
        )

        # Calculate the portfolio value at each time point in the testing set
        valor = {}
        for i in test.index:
            inv_fim = test.loc[i]
            v = sum(portfolio / inv_inicio * inv_fim) / 100 * valor_investido
            valor[i] = v

        # Create a trace for each portfolio in the Plotly figure
        fig.add_trace(
            go.Scatter(x=list(valor.keys()), y=list(valor.values()), mode="lines", name=f"Risk = {risco}")
        )

    # Update layout with title and axis labels for the Plotly figure
    fig.update_layout(
        title="Portfolio Value Evolution",
        xaxis_title="Time Period",
        yaxis_title="Portfolio Value",
    )

    # Display the Plotly figure using Streamlit
    st.plotly_chart(fig, use_container_width=True)



class Particle:
    def __init__(self, points):
        self.position = np.random.rand(points)
        self.velocity = np.random.rand(points)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

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
            raise ValueError("N should be less than or equal to the length of the list.")
    
        indices = np.argsort(lst)[-n:]
        return indices.tolist()
    
    
    def objective_function(self, points, original_series):
        if  len(points)==0:
            return float('inf')
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
        global_best_score = float('inf')
    
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
                    inertia * particle.velocity +
                    personal_weight * r1 * (particle.best_position - particle.position) +
                    global_weight * r2 * (global_best_position - particle.position)
                )
                particle.position = np.clip(particle.position + particle.velocity, 0, 1)
            self.score.append(score)
        return self._top_n_indices(global_best_position, self.num_points)
    

def df_optimal_pso_points(df, num_points_to_choose = 30, num_particles=15, num_iterations=40):
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
    pd_new = pd.concat(stocks_series_list+[placeholder_df], axis=1).sort_values(by='Date').interpolate(limit_direction='both')
    return pd_new
    

def df_moving_avg(df, sma=25):
    return df.rolling(sma).mean().dropna()


def df_weighted(df, recent_weight=.8):
    new_df = df.copy()
    for stock in list(df.columns):
        new_df[stock] = np.linspace(1, recent_weight, len(df[stock]))[::-1]*df[stock]
    return new_df



