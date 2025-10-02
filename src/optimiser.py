import numpy as np
import cvxpy as cp

from .portfolio import Portfolio
from pypfopt import EfficientFrontier, expected_returns, risk_models

class Optimiser:


    def __init__(self, portfolio, risk_free_rate=0.02):
        self.portfolio = portfolio
        self.prices = portfolio.prices
        self.tickers = portfolio.prices.columns.tolist()
        self.mu = expected_returns.mean_historical_return(self.prices)
        self.S = risk_models.sample_cov(self.prices)
        self.risk_free_rate = risk_free_rate

    def max_sharpe(self, min_weight=0.1):
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=(min_weight, 1))
        ef.add_constraint(lambda w: w >= min_weight)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=True)
        return np.array([cleaned_weights[ticker] for ticker in self.tickers])

    def min_variance(self, min_weight=0.1):
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=(min_weight, 1))
        ef.add_constraint(lambda w: w >= min_weight)
        ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=True)
        return np.array([cleaned_weights[ticker] for ticker in self.tickers])

    def target_return(self, target=0.05, min_weight=0.1):
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=(min_weight, 1))
        ef.add_constraint(lambda w: w >= min_weight)
        ef.efficient_return(target_return=target)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=True)
        return np.array([cleaned_weights[ticker] for ticker in self.tickers])
























    # def __init__(self, portfolio, risk_free_rate=0.02):
    #     self.portfolio = portfolio
    #     self.mean_returns = portfolio.mean_returns.values
    #     self.cov_matrix = portfolio.cov_matrix.values
    #     self.n_assets = len(self.mean_returns)
    #     self.risk_free_rate = risk_free_rate

    # def min_variance(self):
    #     w = cp.Variable(self.n_assets)
    #     risk = cp.quad_form(w, self.cov_matrix)
    #     prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
    #     prob.solve()
    #     return w.value
    
    # def max_sharpe(self, max_weight=0.4):
    #     # w = cp.Variable(self.n_assets)
    #     # ret = self.mean_returns @ w
    #     # risk = cp.quad_form(w, self.cov_matrix)
    #     # sharpe = (ret - self.risk_free_rate) / cp.sqrt(risk)
    #     # prob = cp.Problem(cp.Minimize(sharpe), [cp.sum(w) == 1, w >= 0])    
    #     # prob.solve()
    #     # return w.value

    #     w = cp.Variable(self.n_assets)
    #     ret = self.mean_returns @ w
    #     risk = cp.quad_form(w, self.cov_matrix)
    #     prob = cp.Problem(cp.Maximize(ret - self.risk_free_rate),[cp.sum(w) == 1, w >= 0, risk <= 1])  
    #     prob.solve()
        
    #     if w.value is None:
    #         raise ValueError("Optimisation")
        
    #     return w.value


        


    # def target_return(self, target):
    #     w = cp.Variable(self.n_assets)
    #     risk = cp.quad_form(w, self.cov_matrix)
    #     ret = self.mean_returns @ w
    #     constraints = [cp.sum(w) == 1, w >= 0, ret >= target]
    #     prob = cp.Problem(cp.Minimize(risk), constraints)
    #     prob.solve()
    #     return w.value
    
