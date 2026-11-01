"""Download_index_data.py - downloads market context to Data/index/.

Outputs (daily OHLCV, 2018-01-01 .. 2026-06-01):
    Data/index/sp500_stock_data.csv   ^GSPC (S&P 500)
    Data/index/vix_stock_data.csv     ^VIX  (implied volatility)

The 2018 start matches the backfilled stock history so the pre-COVID regime
is preserved when these series are merged into the feature pipeline.
"""

import os

import yfinance as yf

DATA_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data"
INDEX_DIR = os.path.join(DATA_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

START = "2018-01-01"
END = "2026-06-01"

print(f"Downloading ^GSPC (S&P 500) {START} .. {END} ...")
data = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)

if data.empty:
    print("WARNING: No data for ^GSPC")
else:
    data.to_csv(os.path.join(INDEX_DIR, "sp500_stock_data.csv"))
    print(f"Saved sp500_stock_data.csv ({len(data)} rows: "
          f"{data.index.min().date()} .. {data.index.max().date()})")

print(f"Downloading ^VIX (market volatility) {START} .. {END} ...")
vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
if vix.empty:
    print("WARNING: No data for ^VIX")
else:
    vix.to_csv(os.path.join(INDEX_DIR, "vix_stock_data.csv"))
    print(f"Saved vix_stock_data.csv ({len(vix)} rows: "
          f"{vix.index.min().date()} .. {vix.index.max().date()})")