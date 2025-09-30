import numpy as np
import cvxpy as cp

class Optimiser:

    def __init__(self, portfolio, risk_free_rage=0.02):
        self.portfolio = portfolio
        self.mean_returns = portfolio.mean_returns.values
        