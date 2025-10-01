import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from src.portfolio import Portfolio
from src.optimiser import Optimiser


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300) 

portfolio = Portfolio(tickers)

optimiser = Optimiser(portfolio)

# print(optimiser.min_variance())
# print(optimiser.max_sharpe())
# print(optimiser.target_return(0.001))






#############################
# weights = np.random.random(len(portfolio.mean_returns))
# weights /= np.sum(weights)

# mc_sims = 100
# T = 100

# initialPortfolio = 10000

# meanM = np.full(shape=(T, len(weights)), fill_value=portfolio.mean_returns)
# meanM = meanM.T

# portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

# for m in range(0, mc_sims):
#     Z = np.random.normal(size=(T, len(weights)))
#     L = np.linalg.cholesky(portfolio.cov_matrix)
#     daily_returns = meanM + np.inner(L, Z)
#     portfolio_sims[:,m] = np.cumprod(np.inner(weights, daily_returns.T)+1)*initialPortfolio

# plt.plot(portfolio_sims)
# plt.ylabel('Portfolio Value £')
# plt.xlabel('Days')
# plt.title('Monte Carlo Simulation of stock portfolio')
# plt.show()

####################################################