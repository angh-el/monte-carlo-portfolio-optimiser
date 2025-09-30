import yfinance as yf
import pandas as pd
import datetime


class DataFetcher:

    def __init__(self, tickers, start, end):
        self.ticker = tickers
        self.start = start
        self.end = end

    def get_price_data(self):
        data = yf.download(self.ticker, start=self.start, end=self.end)['Close']
    