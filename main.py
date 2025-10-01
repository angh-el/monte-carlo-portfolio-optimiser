import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from src.portfolio import Portfolio
from src.optimiser import Optimiser
from src.simulator import Simulator
import src.risk_metrics as risk_metrics


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'MMM', 'AMD']

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365) 

portfolio = Portfolio(tickers)

optimiser = Optimiser(portfolio)

# print(optimiser.min_variance())
# print(optimiser.max_sharpe())
# print(optimiser.target_return(0.001))

print("\n\n")

simulator = Simulator(portfolio, optimiser)
results = simulator.random_portfolios()

# top_portfolios = results.nlargest(5, "Sharpe")
# print("Top 5 random portolio - by sharpe")
# print(top_portfolios[["Return", "Volatility", "Sharpe"]])


returns = portfolio.returns.mean(axis=1)
cummulative = (1+returns).cumprod()

var = risk_metrics.value_at_risk(returns)
cvar = risk_metrics.conditional_var(returns)
max_drawdown = risk_metrics.max_drawdown(cummulative)
vol = risk_metrics.volatility(returns)

print("\n")
print("Risk metrics (Monte Carlo Simulation #1)")
print("VaR ", var)
print("CVaR ", cvar)
print("max drawdown ", max_drawdown)
print("volatility ", vol)


mpt_weights = optimiser.max_sharpe()
mpt_portfolio_returns = portfolio.returns @ mpt_weights
mpt_cum_returns = (1 + mpt_portfolio_returns).cumprod()
mpt_var = risk_metrics.value_at_risk(mpt_portfolio_returns)
mpt_cvar = risk_metrics.conditional_var(mpt_portfolio_returns)
mpt_max_dd = risk_metrics.max_drawdown(mpt_cum_returns)
mpt_vol = risk_metrics.volatility(mpt_portfolio_returns)
print("\n")
print("Risk metrics (Modern Portfolio Theory)")
print("VaR ", mpt_var)
print("CVaR ", mpt_cvar)
print("max drawdown ", mpt_max_dd)
print("volatility ", mpt_vol)




#############################
weights = np.random.random(len(portfolio.mean_returns))
weights /= np.sum(weights)

# weights = optimiser.min_variance()
# weights = optimiser.max_sharpe()
# weights = optimiser.target_return(0.001)


mc_sims = 100
T = 100

initialPortfolio = 10000

meanM = np.full(shape=(T, len(weights)), fill_value=portfolio.mean_returns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(portfolio.cov_matrix)
    daily_returns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, daily_returns.T)+1)*initialPortfolio

# plt.plot(portfolio_sims)
# plt.ylabel('Portfolio Value £')
# plt.xlabel('Days')
# plt.title('Monte Carlo Simulation of stock portfolio')
# plt.show()


## working out the metrics

final_portfolio_values = portfolio_sims[-1, :]
portfolio_returns = np.diff(portfolio_sims, axis=0) / portfolio_sims[:-1]

confidence_level = 0.99
VaR = np.percentile(portfolio_returns[-1], (1-confidence_level)*100)
CVaR = portfolio_returns[-1][portfolio_returns[-1] <= VaR].mean()

def max_drawdown(path):
    cumulative_max = np.maximum.accumulate(path)
    drawdowns = (path - cumulative_max) / cumulative_max
    return np.min(drawdowns)

max_drawdowns = [max_drawdown(portfolio_sims[:, i]) for i in range(mc_sims)]
max_dd_avg = np.mean(max_drawdowns)
daily_vol = np.std(portfolio_returns, axis=0)
volatily_avg = np.mean(daily_vol) * np.sqrt(252)

print("\n")
print("Risk metrics (Monte Carlo Simulation #2)")
print("VaR ", VaR)
print("CVaR ", CVaR)
print("max drawdown ", max_dd_avg)
print("volatily ", volatily_avg)


####################################################