import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# === PARAMETERS ===
stocks = ["AAPL", "MSFT", "EURUSD=X", "SPY", "QQQ"]
start = "2010-01-05"
end = "2025-01-05"
STOCK_TO_BACKTEST = "SPY"
RSI_PERIOD = 14
RSI_ENTRY = 30
RSI_EXIT = 50

# Retrieve Data
def download_data(stocks, start, end):
    ticker = yf.download(stocks, start, end, auto_adjust=False, progress=False)
    return ticker['Adj Close']

raw = download_data(stocks, start, end)
data = pd.DataFrame(raw[STOCK_TO_BACKTEST])
data.rename(columns={STOCK_TO_BACKTEST: "price"}, inplace=True)

# Compute RSI
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data['rsi'] = compute_rsi(data['price'], RSI_PERIOD)
data['returns'] = np.log(data['price'] / data['price'].shift(1))

# Signal Logic
position = 0
signals = []

for r in data.itertuples():
    if r.rsi < RSI_ENTRY and position == 0:
        signals.append(1)  # enter long
        position = 1
    elif r.rsi > RSI_EXIT and position == 1:
        signals.append(0)  # exit to cash
        position = 0
    else:
        signals.append(position)  # hold previous position

data['signal'] = signals

# Backtesting
data['strategy'] = data['signal'].shift(1) * data['returns']
data.dropna(inplace=True)

cumulative = data[['returns', 'strategy']].cumsum().apply(np.exp)
cumulative.columns = ['Benchmark', 'Strategy']

plt.figure(figsize=(10, 6))
plt.plot(cumulative['Benchmark'], label="Buy & Hold")
plt.plot(cumulative['Strategy'], label="RSI Strategy")
plt.legend()
plt.title(f"Cumulative Gross Performance: {STOCK_TO_BACKTEST}")
plt.grid(True)
plt.show()

print("Total trades taken:", data['signal'].diff().abs().sum() / 2)
daily_sharpe = (data['strategy'].mean() / data['strategy'].std()) * np.sqrt(252)
print("Sharpe ratio:", daily_sharpe)
