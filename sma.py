import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

stocks = ["AAPL", "MSFT", "GOOG"]
start = "2015-01-05"
end = "2020-01-05"

def download_data(stocks, start, end):
    ticker = yf.download(stocks, start, end, auto_adjust = False, progress = False)
    return ticker['Adj Close']

data = download_data(stocks, start, end)

