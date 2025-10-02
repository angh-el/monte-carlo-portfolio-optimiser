import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from src.portfolio import Portfolio
from src.optimiser import Optimiser
from src.simulator import Simulator
import src.risk_metrics as risk_metrics
from src.visualiser import Visualiser


def main():
    tickers = ['STX', 'WDC', 'PLTR', 'NEM', 'MU', 'APP']

    portfolio = Portfolio(tickers)

    optimiser = Optimiser(portfolio)


    simulator = Simulator(portfolio, optimiser)
    
    results = simulator.random_portfolios()
    sims = simulator.simulate_paths()


    visualiser = Visualiser(portfolio)
    visualiser.plot_monte_carlo_cloud(sims)
    visualiser.plot_efficient_frontier(results)

    mc_best = simulator.highest_sharpe_portoflios(results)
    mpt_top = optimiser.max_sharpe()
    visualiser.plot_weights_and_metrics(mc_best, mpt_top, tickers)



if __name__ == "__main__":
    main()
