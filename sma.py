"""
Long-only strat given this is code execution, not in full control
sma + mean-reversion when volatility <= threshold (aka choppy sideways)
sma + momentum when volatility > threshold (sloping / trendy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

stocks = ["AAPL", "MSFT"]
start = "2010-01-05"
end = "2025-01-05"

SMA1 = 10
SMA2 = 30
BB_WINDOW = 20
MOMENTUM_WINDOW = 10
MOMENTUM_THRESHOLD = 0.005
VOL_WINDOW = 10
VOL_THRESHOLD = 0.015 

# retrieve data
def download_data(stocks, start, end):
    ticker = yf.download(stocks, start, end, auto_adjust = False, progress = False)
    return ticker['Adj Close']

raw = download_data(stocks, start, end)

data = pd.DataFrame(raw['AAPL'])
data.rename(columns = {'AAPL' : "price"}, inplace = True)

# SMA signal
data['SMA1'] = data['price'].rolling(SMA1).mean()   
data['SMA2'] = data['price'].rolling(SMA2).mean()

# mean-reversion signal
data['bb_mean'] = data['price'].rolling(BB_WINDOW).mean()
data['bb_std'] = data['price'].rolling(BB_WINDOW).std()
data['lower_bb'] = data['bb_mean'] - 1.5 * data['bb_std']

# momentum signal
data['momentum'] = data['price'].pct_change(MOMENTUM_WINDOW)

# signal implementation
data['returns'] = np.log(data['price'] / data['price'].shift(1))
data['volatility'] = data['returns'].rolling(VOL_WINDOW).std()

def get_signal(row):
    if row['SMA1'] > row['SMA2']:
         # Low volatility → mean-reversion strategy
        if row['volatility'] <= VOL_THRESHOLD:
            if row['price'] < row['lower_bb']:
                return 'MR'
            
        # High volatility → momentum strategy
        else:
            if row['momentum'] > MOMENTUM_THRESHOLD:
                return 'MOM'
    return "NONE"

data['regime'] = data.apply(get_signal, axis=1)
data['signal'] = np.where(data['regime'].isin(['MR', 'MOM']), 1, 0)
data.dropna(inplace = True)

# backtesting implementation
data['strategy'] = data['signal'].shift(1) * data['returns']
cumulative = data[['returns', 'strategy']].cumsum().apply(np.exp)  # to get gross returns rather than log returns
cumulative.columns = ['Benchmark', 'Strategy']
plt.figure(figsize=(10,6))
plt.plot(cumulative['Benchmark'], label="Buy & Hold")
plt.plot(cumulative['Strategy'], label="Strategy")
plt.legend()
plt.title("Cumulative Gross Performance")
plt.grid(True)
plt.show()

print("Total trades taken:", data['signal'].sum())

daily_sharpe = (data['strategy'].mean() / data['strategy'].std()) * np.sqrt(252)
print("Sharpe ratio:", daily_sharpe)