# Efficient Portfolio Management App

Welcome to the Efficient Portfolio Management app! It builds portfolios across Brazilian stocks, **BDRs** (foreign companies traded on B3), real estate funds, **fixed income** (CDI, Tesouro ETFs, FI-Infra) and global, gold and crypto ETFs, filtered by the **analysts' consensus** and optimized for tail risk (CVaR). The app has three pages: Optimization (main), Results and Backtest. The notebook `Portfólio Eficiente.ipynb` explains the theory and holds the analyses behind the defaults.
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

## What changed

| Area | Before | Now |
|---|---|---|
| Assets | B3 stocks and FIIs from investpy's 2021 list | Stocks, **BDRs**, FIIs, Fiagros, **fixed income** (CDI, **Tesouro Direto** IPCA+ and Prefixado bonds, fixed income ETFs, FI-Infra), Brazil and global ETFs, gold, dollar and crypto ETFs, from today's B3 listing |
| Analyst data | Yahoo's rating label of B3 stocks | Rating, number of analysts, price targets and 3-month revisions; BDRs use the consensus on the underlying company (AAPL34 uses Apple's ~40 analysts) |
| Expected returns | Historical means | Historical means (they carry momentum; James-Stein shrinkage is an option, and lost in the backtests), plus optional projections (analysts, FII dividend yield, bond yields, Focus survey) through Black-Litterman |
| Risk | CVaR scaled by sqrt(h) | h-day CVaR with the CDI's drift, so the risk budget does not rest on return forecasts |
| Constraints | None | Max weight per asset and per asset class, minimum position |
| Portfolios | Nearest of 50 fixed-return frontier points | Exact linear programs: minimum CVaR, maximum return for a CVaR budget, tangency |
| Data | Any asset with one missing day dropped; Yahoo's data as is | Frozen-quote and bad-tick filters, own total-return index (Yahoo's adjustment is wrong for some funds), gap filling, B3 calendar |
| Backtest | One train/test split with look-ahead | Walk-forward with costs, point-in-time snapshots, liquidity, fundamentals and BDR analyst history, CDI, Ibovespa and 1/N benchmarks |
| Fundamentals | None | CVM filings since 2011 (point in time): P/L, P/VP, ROE, margins, leverage, growth; a sector-neutral factor score |
| Explanations | None | Claude writes why each holding is in the portfolio from the computed facts, with every cited number checked |

## Model

The optimizer solves the Rockafellar-Uryasev linear program above on daily historical scenarios (simple returns, since a portfolio's log return is not the weighted sum of its assets' log returns), with:

- **Risk over h trading days:** CVaR = sqrt(h) x tail dispersion of daily returns - h x the CDI's daily return. A CDI-only portfolio has a negative CVaR (even its bad months are gains), and no asset gets credit for its own expected return: in the backtests, crediting it let recent winners loosen the budget and the realized CVaR reached about twice the target (the option remains).
- **Expected returns:** the CDI plus each asset's mean spread over it in the lookback window. Sample means are noisy and an optimizer amplifies their errors (Michaud, 1989), but over one or two years they also carry momentum: in the walk-forward tests below they beat James-Stein shrinkage toward a common return per unit of risk (still available), whose target also favors the riskiest assets.
- **Objectives:** maximum return for a CVaR budget (default: 5% a month; the Ibovespa's is about 10-11%), minimum CVaR, or the tangency portfolio (best excess return per unit of CVaR, via the Charnes-Cooper transformation).
- **Constraints:** long only, max weight per asset (the CDI is exempt), min/max per asset class, positions below the minimum removed by solving again.
- **Scenario weights:** an optional half-life gives recent days more weight (it replaces the old "time weighted" price transformation, which added an artificial drift).
- **Projections (optional):** Black-Litterman views from the analysts' 12-month price targets plus dividends (relative to peers), the FII median dividend yield (for the class as a whole), the Tesouro bonds' yields (carry; IPCA+ with the Focus expected inflation) and the Focus survey's expected CDI. View errors are assumed to be as correlated as the assets, so many bullish views on similar stocks do not add up.

## Data

- `python build_dataset.py` saves `datasets/<date>/universe.csv`: the most traded tickers of each kind (brapi.dev), liquidity (Yahoo Finance), the analyst consensus, the stocks' CVM fundamentals and the BDR-to-company mapping (cached in `datasets/reference/bdr_underlying.csv`). It also refreshes the CVM filings and the BDRs' analyst history (below). **Run it regularly**: Yahoo keeps no history of the B3 consensus, so the snapshots are that history, and backtests use the latest one available on each rebalance date. On Windows, to run it every Sunday:

  ```
  schtasks /Create /SC WEEKLY /D SUN /ST 09:00 /TN "Portfolio snapshot" /TR "C:\Users\bruno\anaconda3\python.exe <project folder>\build_dataset.py"
  ```
- Fundamentals: the annual (DFP) and quarterly (ITR) statements of every listed company since 2011 from the CVM's open data, with the date each filing was first received (`fundamentals.py`; raw files cached in `datasets/cvm/`, extracted accounts in `datasets/reference/cvm_fundamentals.csv.gz`).
- Analyst history of the BDRs: the month-end consensus of their US companies since 2012, rebuilt from each broker's rating changes and price targets (`datasets/reference/analyst_history_us.csv.gz`). Backtests use the consensus of each rebalance date.
- Prices: closing prices and dividends from Yahoo Finance turned into a total-return index. Quotes frozen for 10 days or more are blanked (Yahoo kept IRFM11, IMAB11 and B5P211 at the same price from 2022 to January 2026) and isolated bad prints are removed.
- Tesouro Direto: daily prices and yields of the IPCA+ and Prefixado bonds from the National Treasury's open data, cached in `datasets/reference/tesouro_direto.csv.gz` (the site often rate-limits).
- CDI: Central Bank SGS series 12. Selic and IPCA expectations: Focus survey (Olinda API), also for past dates.
- If brapi.dev starts requiring a token, set `BRAPI_TOKEN` in `.env`.

## Analyst recommendations: what the evidence says

The notebook tests each signal with point-in-time data:

| Test | Result |
|---|---|
| B3 consensus of 2025-09-10 vs. the next 13 months | Median return: strong_buy +5%, buy +29%, hold +43.5% (CDI +16%); the 2 underperform stocks lost 29% |
| 10 years of consensus on the US companies behind the BDRs (128 months) | Rank IC at 12 months of about 0.05 for the rating and 0.04 for the upside; none for revisions |
| FII dividend yield, 2022-2026 | IC of about 0.04 at 1-3 months; the highest yields did worst (yield traps) |
| BDRs, each month's consensus (2020-2026 backtest, 25 BDRs, 1/N) | No filter (most liquid): 30.3% a year, Sharpe 0.86; excluding Underperform/Sell and ranking by analysts: 25.1%, 0.67; strong_buy only: 23.3%, 0.54 |

Hence the defaults: exclude Underperform/Sell rather than keeping only strong_buy (the old default, and the worst group out of sample), require 3 or more analysts, do not rank by the consensus, and give the analyst projections a low confidence (100% a year of uncertainty, which matches an IC of about 0.04).

**Point in time.** Yahoo keeps no history of the B3 consensus, so the weekly snapshots are that history. For the BDRs' US companies, `analysts.build_us_history` rebuilds the consensus of every month end since 2012 from each broker's rating changes and price targets; the brokers' labels are mapped onto Yahoo's 1-5 scale with a linear fit (correlation 0.89), and backtests use the consensus of each rebalance date.

## Fundamentals: a quantitative recommendation

`fundamentals.py` computes, for each stock and date, P/L, P/VP, ROE, net and EBIT margins, net debt / equity, growth, the profitable years among the last five, dividend yield, momentum and volatility, from the CVM filings published by that date and the prices up to it. The **fundamental score** averages four sector-neutral z-score groups: value (earnings yield, book-to-market, dividend yield), quality (ROE, margin, low leverage, steady profits), momentum and low risk.

| Signal (rank IC, point in time, 128 months, ~200 stocks) | 1 month | 3 months | 12 months |
|---|---|---|---|
| Fundamental score | 0.085 | 0.122 | 0.162 |
| Value / quality / momentum / low risk | 0.062 / 0.064 / 0.047 / 0.059 | | |
| Analyst consensus (BDRs' companies), for comparison | 0.017 | 0.026 | 0.048 |

In the 2016-2026 backtest (two-year lookback), the 25 stocks with the best score each month returned 24.7% a year (Sharpe 0.74) against 19.5% (0.46) for the 25 most liquid, and the optimized portfolio 25.6% (0.74) against 12.4% (0.22); with a one-year lookback (notebook) the optimized portfolio of the best scores reached 32.6% (0.97). So **stocks are now ranked by the fundamental score by default** (`rank_by="auto"`), and the score can also tilt expected returns (alpha = IC x volatility x score, with a conservative IC of 0.05). Caveats: the candidates are today's listed stocks, which flatters value and quality; share counts start in 2020 in the CVM's files (earlier dates use the first filing that has one, adjusted for splits); about 3% of market values still differ from Yahoo's by more than half (restructurings, unit structures).

## AI explanations (Claude)

`explainer.py` builds a sheet of facts for each holding (fundamentals against the sector median, five years of history, analyst consensus, weight, share of the portfolio's tail risk, correlation with the rest, bond yield...), with every number already formatted in Portuguese. Claude (`claude-opus-5-5`, structured output, with Anthropic's server-side fallback on refusals) writes a headline, a justification and the counterpoints for each holding from those facts only, and every number in the text is checked against them. The model does not pick the assets: the optimizer does, and the text explains its choice. It is not investment advice (in Brazil, recommendations to third parties require a certified analyst, CVM Resolution 20/2021). Set `ANTHROPIC_API_KEY` in the environment or in `.env` and use **Explain with Claude** on the Results page.

## Backtests

```
python run_backtests.py --suite model          # original vs new model, stocks vs several classes, since 2021
python run_backtests.py --suite estimators     # historical vs shrunk means, CDI vs expected drift in the CVaR
python run_backtests.py --suite analysts       # B3 analyst filters with point-in-time snapshots only
python run_backtests.py --suite analysts_bdr   # BDR analyst filters with each month's consensus, since 2020
python run_backtests.py --suite fundamentals   # stocks by liquidity vs by the fundamental score, since 2016
python run_backtests.py --suite model --lookback 252 --rebalance 63 --cost 0.002
```

Each run saves `metrics.csv`, `values.csv` and `chart.html` in `results/`. The Backtest page runs the same engine with the sidebar settings.

Results from 2021-01 to 2026-10 (monthly rebalancing, 0.1% cost per unit of turnover, two-year lookback). Every listed asset is a candidate and the 25 most traded of each class are taken at each rebalance; an earlier version drew the candidates from the assets that are the most traded today, which favors those that rallied and overstated every result (1/N on stocks by 8 points a year).

| Strategy | Annual return | Volatility | Sharpe vs CDI | Max drawdown | CVaR/month expected / realized |
|---|---|---|---|---|---|
| Original model, stocks, CVaR 10% | 13.4% | 18.4% | 0.18 | -21.4% | - / 10.5% |
| Equal weight (1/N), stocks | 9.6% | 20.6% | 0.02 | -31.0% | - / 12.4% |
| New model, stocks, CVaR <= 10% | 13.0% | 17.9% | 0.16 | -24.5% | 12.0% / 10.4% |
| New model, multi-asset, CVaR <= 10% | 26.4% | 18.3% | 0.78 | -25.0% | 10.0% / 10.2% |
| New model, multi-asset, CVaR <= 5% | 15.8% | 10.6% | 0.41 | -13.1% | 5.0% / 5.8% |
| New model, multi-asset, tangency | 11.7% | 14.3% | 0.09 | -33.2% | 7.6% / 9.2% |
| Equal weight (1/N), multi-asset | 13.0% | 10.4% | 0.18 | -19.1% | - / 5.7% |
| CDI | 11.5% | 0.2% | | | |
| Ibovespa | 9.8% | 18.1% | 0.00 | -26.5% | - / 10.9% |

The stock-only portfolio's expected CVaR averages more than its 10% budget because no stock portfolio fits that budget when the window includes March 2020; the backtest then holds the least risky one. With a one-year lookback the new model keeps 10.6% a year on stocks (Sharpe 0.05) while the original falls to 3.5% (-0.24), and the multi-asset portfolios return 31.4% (CVaR <= 10%, Sharpe 0.89) and 17.7% (CVaR <= 5%, 0.49) against 12.1% (0.10) for 1/N. On B3 stocks alone, 2021-2026 was a poor period (the Ibovespa returned less than the CDI); the model adds the most when it combines asset classes. The tangency portfolio, which maximizes a ratio of noisy estimates, had the lowest Sharpe ratio of the multi-asset portfolios, which is why it is not the default.

**Which expected returns.** The first version of the new model shrank the historical means toward a common Sharpe ratio and credited each asset's expected return in the CVaR. `--suite estimators` compares those choices (annual return, Sharpe vs CDI, realized CVaR per month):

| Stocks, CVaR <= 10% | 2021-2026 | 2021-2026, one-year lookback | 2016-2026 |
|---|---|---|---|
| **Historical means, CDI drift (default)** | 13.0%, 0.16, 10.4% | 10.6%, 0.05, 10.7% | 12.4%, 0.22, 12.3% |
| Historical means, expected drift | 15.5%, 0.28, 11.1% | 10.6%, 0.07, 12.9% | 16.9%, 0.38, 14.5% |
| Shrinkage, CDI drift | 5.5%, -0.20, 10.7% | -1.7%, -0.55, 12.0% | 9.3%, 0.08, 12.7% |
| Shrinkage, expected drift (first version) | 4.8%, -0.22, 11.4% | 0.4%, -0.42, 12.5% | 11.1%, 0.17, 13.7% |
| Equal weight (1/N) | 9.6%, 0.02, 12.4% | 9.4%, 0.01, 12.3% | 19.5%, 0.46, 15.3% |

| Multi-asset, CVaR <= 5% | 2021-2026 | 2021-2026, one-year lookback | 2016-2026 |
|---|---|---|---|
| **Historical means, CDI drift (default)** | 15.8%, 0.41, 5.8% | 17.7%, 0.49, 7.0% | 14.8%, 0.45, 6.4% |
| Historical means, expected drift | 23.2%, 0.67, 9.8% | 28.4%, 0.71, 13.4% | 21.0%, 0.62, 10.6% |
| Shrinkage, CDI drift | 11.5%, 0.05, 5.7% | 10.2%, -0.04, 7.0% | 12.7%, 0.28, 6.7% |
| Shrinkage, expected drift (first version) | 12.0%, 0.09, 6.6% | 14.7%, 0.27, 7.8% | 14.4%, 0.37, 7.8% |
| Equal weight (1/N) | 13.0%, 0.18, 5.7% | 12.1%, 0.10, 5.4% | 20.2%, 0.66, 9.0% |

Historical means beat shrinkage in all twelve comparisons: over one or two years a mean carries momentum, which shrinkage toward a common return per unit of risk removes, and that target favors the riskiest assets, while on B3 the low-risk stocks did better. Crediting the expected returns in the CVaR raises returns, but the realized risk ran two to almost three times the multi-asset budget (9.8-13.4% a month for 5%); with the CDI's drift it stayed within 1.0-1.4 times the budget. For about the same realized risk, a 10% budget with the CDI's drift returned 26.4% a year (Sharpe 0.78, realized CVaR 10.2%) against 23.2% (0.67, 9.8%) for 5% with the expected drift. Hence the defaults: historical means and the CDI's drift (both remain options in the sidebar). 1/N remains hard to beat (DeMiguel, Garlappi & Uppal, 2009): over 2016-2026 it had the best Sharpe ratio on both universes. Caveats: the candidates are today's listed assets (survivorship bias, strongest for BDRs), and Yahoo revises its data, so reruns can differ slightly.

## The app

- **Optimization (main page):** choose the asset classes, liquidity, analyst and FII filters, and the model in the sidebar. The page shows the filter funnel, the selected assets with their consensus, the efficient frontier (with each asset and the CDI-tangency line), the key portfolios and, with projections on, how each projection moved the expected returns.
- **Results:** pick the maximum CVaR and get the highest expected return portfolio for it, by asset and by class, with a CSV download, the facts behind each holding and, with an API key, Claude's explanation of why each one is there.
- **Backtest:** walk-forward test of the sidebar strategy against the CDI, the Ibovespa, 1/N and the original model, with drawdowns, metrics (including expected vs realized CVaR) and the allocation by class over time.

## Running

```
pip install -r requirements.txt
streamlit run app.py
python build_dataset.py   # refresh the data (first run ~10 minutes)
pytest                    # 63 offline unit tests
pytest -m network         # 6 integration tests with live data
docker-compose up
```

## References

- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*.
- Michaud, R. (1989). The Markowitz optimization enigma: is "optimized" optimal? *Financial Analysts Journal*.
- Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. *Journal of Financial and Quantitative Analysis*.
- Black, F. & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*; He, G. & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. Goldman Sachs.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus naive diversification. *Review of Financial Studies*.
- Womack, K. (1996). Do brokerage analysts' recommendations have investment value? *Journal of Finance*.
- Jegadeesh, N., Kim, J., Krische, S. & Lee, C. (2004). Analyzing the analysts: when do recommendations add value? *Journal of Finance*.
- Da, Z. & Schaumburg, E. (2011). Relative valuation and analyst target price forecasts. *Journal of Financial Markets*.
- Diether, K., Malloy, C. & Scherbina, A. (2002). Differences of opinion and the cross section of stock returns. *Journal of Finance*.
