import pandas as pd
import numpy as np
import datetime as dt
from data_fetcher import DataFetcher


class Portfolio:
    
    def __init__(self, tickers, start=dt.datetime.now() - dt.timedelta(days=300), end=dt.datetime.now()):
        self.tickers = tickers
        self.start = start
        self.end = end

        self.prices: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.mean_returns: pd.Series | None = None
        self.cov_matrix: pd.DataFrame | None = None

        self.fetch_data()
        self.compute_statistics()

    
    def fetch_data(self):
        fetcher = DataFetcher(self.tickers, self.start, self.end)
        self.prices = fetcher.get_price_data()

    
    def compute_statistics(self):
        # log returns
        # self.returns = np.log(self.prices / self.prices.shift(1)).dropna()    

        # percentage change
        self.returns = self.prices.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov