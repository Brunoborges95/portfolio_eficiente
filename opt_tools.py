# Import necessary libraries
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from stqdm import stqdm
import yfinance as yf
import numpy.matlib

# Function to drop columns with NaN values
def drop_columns_with_nan(df):
    # Identify columns with NaN values
    columns_nan = list(df.iloc[:, list(df.isna().any())].columns)

    # Check if there are columns with NaN values
    if columns_nan != []:
        st.markdown(
            f"<span style='color:red'> Warning: </span> The assets <span style='color:yellow'>{', '.join(columns_nan)}</span> contain missing data and should not be used in the portfolio.",
            unsafe_allow_html=True,
        )

        # Drop columns with NaN values
        df_new = df.drop(columns=columns_nan)
    else:
        # If no NaN values are found, return the original dataframe
        return df

    # Return the dataframe without columns with NaN values
    return df_new

# Data file path
def read_stocks_info(date):
    try:
        df_stocks_info = pd.read_csv(f"datasets/{date.strftime('%Y-%m-%d')}/df_stocks_info.csv")
    except FileNotFoundError:
        return pd.read_csv(f"datasets/2025-09-10/df_stocks_info.csv")
    return df_stocks_info

@st.cache_data
def collect_historic_stocks(stocks_codes, data_inicio, data_fim):
    historic_stocks = yf.download(stocks_codes, start=data_inicio, end=data_fim)["Close"]
    historic_stocks = drop_columns_with_nan(historic_stocks)
    return historic_stocks

class Portfolio_optimization:
    def __init__(self, historic_stocks):
        self.historic_stocks = historic_stocks
        self.stocks = list(historic_stocks.columns)

    # Function to plot historical stock prices
    def plot_historic(self):
        fig = go.Figure()

        # Iterate through each stock in the portfolio
        for stock in self.stocks:
            # Add a trace for each stock, representing historical closing prices
            fig.add_trace(
                go.Scatter(
                    x=self.historic_stocks.index,
                    y=self.historic_stocks[stock],
                    mode="lines",
                    name=f"{stock}",
                )
            )

        # Update layout with title and axis labels
        fig.update_layout(
            title="Historical closing prices of stocks",
            xaxis_title="Time period",
            yaxis_title="Price",
        )

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def returns(self):
        """This function calculates logarithmic returns for each stock based on historical stock data,
        considers a specified window length (nodes), and returns a dictionary containing the calculated returns,
        expected returns, and mean returns."""
        # Transpose historical stock data for easier calculation
        data = self.historic_stocks.values.transpose()
        data = np.where(data <= 0, 1e-07, data)
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
        """This function optimizes a portfolio based on a specified risk metric (VAR or CVAR)
        using linear programming. It calculates the mean return values and the corresponding
        risk measure values for a range of possible returns. The optimized portfolio weights and
        the calculated risk measure are then returned as a dictionary."""
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
            title=f"{metric} - Efficient Frontier",
            xaxis_title=f"{metric} (%)",
            yaxis_title="Expected return (%)",
        )

        # Markdown explanation about the efficient frontiers
        st.markdown(
            """The charts below represent the return for each value of the considered risk.
            The chosen metric was CVaR, as it is a coherent risk measure and also because, due to the non-convex nature of VaR optimization, its efficient frontier
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
            title=f"Stock distribution in the portfolio - {metric}",
            xaxis_title=f"{metric} (%)",
            yaxis_title="Distribution (%)",
        )

        # Display a markdown explanation if the plot parameter is True
        if plot:
            st.markdown(
                """The area chart below represents the proportion of assets in the portfolio for each considered risk value."""
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
            title="Proportion of assets in the portfolio",
            xaxis_title="Proportion",
            yaxis_title="Stocks",
        )

        # Display a markdown explanation if the plot parameter is True
        if plot:
            st.markdown(
                """The final result is, for a given risk value,
                        the stocks I should invest in and in what proportion, for my ideal portfolio."""
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
        # Objective function: difference between the original series and the chosen points
        subset = self._get_values_at_indices(points, original_series)
        interpolated_series = self._interpolate_nan(np.array(subset))
        # Calculate the difference between the original series and the interpolated points
        score = np.sum(np.abs(original_series - interpolated_series))
        return score

    # Rest of the code remains the same
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
    """This function performs a backtest on a portfolio optimization strategy,
    calculating the portfolio value over time for different risk levels.
    The results are visualized using a plotly figure."""
    # Collect historical stock data for the backtest period
    historic_stocks = collect_historic_stocks(
        stocks_codes, start_date_backtest, current_date
    )

    # Split the data into training and testing sets
    train = historic_stocks[:end_date_backtest]
    test = historic_stocks[end_date_backtest:]

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
        f"<span style='color:green; font-size:larger; font-weight:bold'> Initial investment: {valor_investido}</span>",
        unsafe_allow_html=True,
    )

    for portfolio, risk in zip(portfolios, risk_values):
        initial_investment = test.iloc[0]
        final_investment = test.iloc[-1]
        allocation = portfolio / initial_investment * final_investment
        current_value = sum(allocation) / 100 * valor_investido

        # Display information about the backtest results
        color = "green" if current_value >= valor_investido else "red"

        # Display the formatted text with the color determined by the condition
        st.markdown(
            f"For risk = {risk}, the current value of the investment is <span style='color:{color}; font-size:larger; font-weight:bold'>{round(current_value, 2)}</span>",
            unsafe_allow_html=True,
        )

        # Calculate the portfolio value at each time point in the testing set
        value = {}
        for i in test.index:
            final_investment = test.loc[i]
            v = sum(portfolio / initial_investment * final_investment) / 100 * valor_investido
            value[i] = v

        # Create a trace for each portfolio in the Plotly figure
        fig.add_trace(
            go.Scatter(
                x=list(value.keys()),
                y=list(value.values()),
                mode="lines",
                name=f"Risk = {risk}",
            )
        )

    # Update layout with title and axis labels for the Plotly figure
    fig.update_layout(
        title="Portfolio value evolution",
        xaxis_title="Time period",
        yaxis_title="Portfolio value",
    )

    # Display the Plotly figure using Streamlit
    st.plotly_chart(fig, use_container_width=True)
