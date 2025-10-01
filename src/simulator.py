import numpy as np
import pandas as pd
from .portfolio import Portfolio

class Simulator:

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.mean_returns = portfolio.mean_returns.values
        self.cov_matrix = portfolio.cov_matrix.values
        self.n_assets = len(self.mean_returns)

    
    def random_portfolios(self, n=5000):
        results = []
        for _ in range(n):
            weights = np.random.dirichlet(np.ones(self.n_assets))

            port_return = np.dot(weights, self.mean_returns)
            port_vol = np.sqrt(weights @ self.cov_matrix @ weights.T)
            sharpe = port_return / port_vol if port_vol > 0 else np.nan

            results.append({"Return":port_return,"Volatility":port_vol,"Sharpe": sharpe,"Weights": weights})

        return pd.DataFrame(results)
    
    def simulate_paths(self, n_sims = 1000, horizon = 252):
        L = np.linalg.cholesky(self.cov_matrix)
        mean = self.mean_returns / horizon
        sims = np.zeros((n_sims, horizon, self.n_assets))

        for i in range(n_sims):
            shocks = np.random.normal(size=(horizon, self.n_assets))
            correlated = shocks @ L.T
            sims[i] = mean + correlated

        return sims
    
