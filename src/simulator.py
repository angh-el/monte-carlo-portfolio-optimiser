import numpy as np
import pandas as pd
import cvxpy as cp
from .portfolio import Portfolio


class Simulator:

    def __init__(self, portfolio, optimiser):
        self.portfolio = portfolio
        self.mean_returns = portfolio.mean_returns.values
        self.cov_matrix = portfolio.cov_matrix.values
        self.n_assets = len(self.mean_returns)
        self.optimiser = optimiser

    
    def random_portfolios(self, n=5000):
        results = []
        for _ in range(n):
            weights = np.random.dirichlet(np.ones(self.n_assets))
            # weights = np.random.random(self.n_assets)
            # weights /= np.sum(weights)

            # weights = self.optimiser.min_variance()
            # weights = self.optimiser.max_sharpe()
            # weights = self.optimiser.target_return(0.001)


            port_return = np.dot(weights, self.mean_returns)
            port_vol = np.sqrt(weights @ self.cov_matrix @ weights.T)
            sharpe = port_return / port_vol if port_vol > 0 else np.nan

            results.append({"Return":port_return,"Volatility":port_vol,"Sharpe": sharpe,"Weights": weights})

        return pd.DataFrame(results)
    
    def simulate_paths(self, n_sims = 1000, horizon = 252):
        weights = np.random.random(len(self.portfolio.mean_returns))
        weights /= np.sum(weights)

        mc_sims = 100
        T = 100

        initialPortfolio = 10000

        meanM = np.full(shape=(T, len(weights)), fill_value=self.portfolio.mean_returns)
        meanM = meanM.T

        portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

        for m in range(0, mc_sims):
            Z = np.random.normal(size=(T, len(weights)))
            L = np.linalg.cholesky(self.portfolio.cov_matrix)
            daily_returns = meanM + np.inner(L, Z)
            portfolio_sims[:,m] = np.cumprod(np.inner(weights, daily_returns.T)+1)*initialPortfolio
        
        return portfolio_sims
    
    def highest_sharpe_portoflios(self, results):
        top_10 = results.sort_values(by="Sharpe", ascending=False).head(10)
        top_weights = np.stack(top_10["Weights"].values)
        average_weights = np.mean(top_weights, axis=0)
        mc_best = {
            "weights": average_weights,
            "returns": self.portfolio.returns
        }

        return mc_best

            
