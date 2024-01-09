# Efficient Portfolio Management App

Welcome to the Efficient Portfolio Management app! This tool is designed to empower you in constructing an optimal investment portfolio by carefully selecting stocks based on specific criteria and utilizing advanced optimization algorithms. The app is divided into three main sections: Main, Backtest, and Results.
## Theory:
The main aspect of portfolio theory is that the individual risk of an asset differs from its risk in the portfolio, making diversification capable of minimizing the non-systematic risk of the assets collectively. Through this minimization, it is possible to select the optimal proportion of each asset in the portfolio, optimizing the risk-return relationship of the bond portfolio. The figure below effectively illustrates this concept: For more than 30 assets, it is possible to practically eliminate all non-systematic risk from the portfolio. The remaining risk is associated with market, credit, liquidity, or operational factors.

Here, the concept revolves around understanding risk as the amount one is willing to accept losing. Quite simple, isn't it? Suppose you are considering a $1000$ reais investment. Your manager informs you that in portfolio $X$, you may have a return of $300$% per year, with a maximum potential loss of $900$ reais and a $5$% chance of occurrence. The Value at Risk (VaR) represents the expected maximum loss (not to be confused with the maximum possible loss), which is $900$ reais, and the VaR alpha is the likelihood of exceeding that loss (in this example, $5$%). Minimizing VaR involves selecting the optimal set of assets that, with the same return, reduces this expected maximum loss.

As mentioned earlier, VaR fails in subadditivity and, worse, in a property that optimization enthusiasts (including myself) love—convexity. Fortunately, there is a measure that, in addition to being convex, is coherent. Conditional Value at Risk (CVaR) examines losses that exceed the VaR limit. In the example of portfolio $X$, this means analyzing losses for 5%, 4%,... chance and taking an average of these. VaR and CVaR are closely related, and minimizing CVaR will also lead to a reduction in the VaR of the portfolio. The figure below depicts a normal curve with the expected losses of VaR and CVaR and their corresponding expected probabilities.
<ol>
<li> VaR attempts to summarize in a single number ($\alpha$) the expected maximum loss within a certain period with a certain degree of statistical confidence: $$VaR_{1-\alpha}(X):=inf\left \{t\in \mathbb{R} : Pr(X\leqslant  t)\geqslant 1-\alpha\right \}, \alpha\in [0,1]$$ </li>
<li> CVaR can be defined as the conditional expectation of losses exceeding the VaR:
 $$CVaR_{1-\alpha}(X):=inf\left (t+\frac{1}{\alpha}E[X-t]_{+} \right ), \alpha\in (0,1]$$ $$CVaR_{1-\alpha}(X)=\frac{1}{\alpha}\int_{0}^{\alpha}VaR_{1-t}(X)dt,$$ onde $[c]_{+}=max(0,c).$

Through a series of mathematical manipulations, Rockafellar and Uryasev (2000) rewrites the CVaR calculation in terms of a function $F_{\beta}$ given by
$$F_{\beta}(x,\alpha)=\alpha+(1-\beta)^{-1}\int_{y\in \mathbb{R}^{m}}[f(x,y)-\alpha]p(y)dy,$$ 
where $p(y)$ is the probability density function of market variables; $\beta$ is the chosen probability level, and $f(x,y)$ is a loss function associated with portfolio $x$ and market variables $y$.</li>
</ol>

For discrete values, the equation above can be rewritten as:

$$F_{\beta}(x,\alpha)=\alpha+\frac{1}{n(1-\beta)}\sum_{k=1}^{n}[f(x,y)-\alpha]_{+}$$

Thus, Rockafellar and Uryasev (2000) utilize the linear function $F_{\beta}$ to define the optimization form of a stock portfolio using CVaR as a risk measure:

$$Min \text{ }\\text{ }\\text{ }\text{ }\alpha+\frac{1}{n(1-\beta)}\sum_{k=1}^{n}[-w_{i}R_{i}-\alpha]_{+}$$

$$S.a \text{ }\\text{ }\\text{ }\\sum_{k=1}^{n}w_{i}=1$$

$$\text{ }\text{ }\text{ }\text{ }\text{ } 0\leqslant w_{i}\leq 1$$

where $n$ is the sample size and $w_{i}$ is the proportion of each asset in the portfolio.

## Main

### Overview
The Main section of the app allows you to set criteria for stock selection and employs advanced optimization algorithms to create an efficient investment portfolio.

#### Stock Selection Criteria

- **Date Range:** Choose a start and end date to analyze stock data.
- **Stock Type:** Select the type of stocks (e.g., common, preferred).
- **Sector:** Filter stocks by sector or explore across all sectors.
- **Analyst Recommendation:** Tailor your portfolio based on analyst recommendations.

After setting your criteria, click the 'Generate Optimization' button to proceed.

#### Optimization Algorithm

The algorithm uses historical stock data to optimize the portfolio composition. It employs a risk metric, either Value at Risk (VaR) or Conditional Value at Risk (CVaR), to determine the ideal proportion of assets in the portfolio.

## Backtest

### Overview
The Backtest section of the app enables you to analyze and backtest the efficiency of your selected criteria and optimization algorithms.

#### Stock Selection Criteria

- **Date Range:** Choose a start and end date to analyze stock data.
- **Stock Type:** Select the type of stocks (e.g., common, preferred).
- **Sector:** Filter stocks by sector or explore across all sectors.
- **Analyst Recommendation:** Tailor your portfolio based on analyst recommendations.

After setting your criteria, click the 'Run Backtest' button to evaluate the historical performance of your portfolio.

#### Optimization Algorithm

The backtest utilizes historical stock data to assess the effectiveness of the chosen optimization algorithm, providing insights into the potential performance of your portfolio.

## Results

### Overview
The Results section of the app provides an interactive platform for exploring the distribution of your investments and fine-tuning the level of risk according to your preferences.

#### Current Portfolio Distribution

The chart displays the current percentage distribution of your investments in various stocks, using portfolio optimization metrics for a clear view of asset allocation.

#### Risk Adjustment

Adjust the risk level in your portfolio using the slider. Observe how the portfolio dynamically adapts to your choices.

**Risk Slider:**
- Minimum: 0
- Maximum: 100
- Increment: 5

Feel free to experiment and find the investment distribution that aligns best with your financial goals.

---

Thank you for using the Efficient Portfolio Management app. For more details, documentation, and updates, refer to the source code and documentation available on this GitHub repository. If you encounter any issues or have suggestions, please open an issue on the repository.