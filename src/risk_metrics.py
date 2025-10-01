import numpy as np
import pandas as pd

def value_at_risk(returns, alpha = 0.05):
    # alpha -> confidence level
    # defult at 0.05 for 5% VaR

    return np.percentile(returns, 100*alpha)

def conditional_var(returns, alpha = 0.05):
    var = value_at_risk(returns, alpha)
    return returns[returns <= var].mean()

def max_drawdown(cumulative_returns):
    cum = np.array(cumulative_returns)
    running_max = np.maximum.accumulate(cum)

    drawdowns = (cum - running_max) / running_max
    return drawdowns.min()

def volatility(returns, annualise = True, periods = 252):
    vol = np.std(returns, ddof=1)
    return vol * np.sqrt(periods) if annualise else vol

