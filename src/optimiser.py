import numpy as np
import cvxpy as cp

from .portfolio import Portfolio

class Optimiser:

    def __init__(self, portfolio, risk_free_rate=0.02):
        self.portfolio = portfolio
        self.mean_returns = portfolio.mean_returns.values
        self.cov_matrix = portfolio.cov_matrix.values
        self.n_assets = len(self.mean_returns)
        self.risk_free_rate = risk_free_rate

    def min_variance(self):
        w = cp.Variable(self.n_assets)
        risk = cp.quad_form(w, self.cov_matrix)
        prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
        prob.solve()
        return w.value
    
    def max_sharpe(self):
        # w = cp.Variable(self.n_assets)
        # ret = self.mean_returns @ w
        # risk = cp.quad_form(w, self.cov_matrix)
        # sharpe = (ret - self.risk_free_rate) / cp.sqrt(risk)
        # prob = cp.Problem(cp.Minimize(sharpe), [cp.sum(w) == 1, w >= 0])    
        # prob.solve()
        # return w.value

        w = cp.Variable(self.n_assets)
        ret = self.mean_returns @ w
        risk = cp.quad_form(w, self.cov_matrix)
        prob = cp.Problem(cp.Maximize(ret - self.risk_free_rate),[cp.sum(w) == 1, w >= 0, risk <= 1])  
        prob.solve()
        
        if w.value is None:
            raise ValueError("Optimisation")
        
        return w.value
    
    def target_return(self, target):
        w = cp.Variable(self.n_assets)
        risk = cp.quad_form(w, self.cov_matrix)
        ret = self.mean_returns @ w
        constraints = [cp.sum(w) == 1, w >= 0, ret >= target]
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve()
        return w.value
    