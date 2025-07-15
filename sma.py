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



