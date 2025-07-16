import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

stocks = ["AAPL", "MSFT"]
start = "2015-01-05"
end = "2020-01-05"

def download_data(stocks, start, end):
    ticker = yf.download(stocks, start, end, auto_adjust = False, progress = False)
    return ticker['Adj Close']

raw = download_data(stocks, start, end)

data = pd.DataFrame(raw['AAPL'])
data.rename(columns = {'AAPL' : "price"}, inplace = True)

data['SMA1'] = data['price'].rolling(42).mean()
data['SMA2'] = data['price'].rolling(252).mean()

plt.figure(figsize = (10,6))
plt.plot(data['price'], label = "price", color = "black")
plt.plot(data['SMA1'], label = "SMA1", color = "blue")
plt.plot(data['SMA2'], label = "SMA2", color = "red")

plt.title("AAPL 42 & 252 MA")
plt.grid(True)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Adjusted closing price")
plt.show()

data['returns'] = np.log(data['price'] / data['price'].shift(1))

"""
for extremely risk prudent strategy, 
we go long-only given that we are trading on cash account, 
no margin-call
"""

data['signal'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
data.dropna(inplace = True)

data['strategy'] = data['signal'].shift(1) * data['returns']
cumulative = data[['returns', 'strategy']].cumsum().apply(np.exp)
cumulative.columns = ['Buy & Hold', 'Strategy']

plt.figure(figsize=(10,6))
plt.plot(cumulative['Buy & Hold'], label="Buy & Hold")
plt.plot(cumulative['Strategy'], label="Strategy")
plt.legend()
plt.title("Cumulative Gross Performance")
plt.grid(True)
plt.show()

