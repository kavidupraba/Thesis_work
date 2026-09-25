"""Download_index_data.py - downloads S&P 500 index (^GSPC) to Data/index/."""

import yfinance as yf
import os

DATA_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data"
INDEX_DIR = os.path.join(DATA_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

print("Downloading ^GSPC (S&P 500) ...")
data = yf.download("^GSPC", start="2019-01-01", end="2026-06-01")

if data.empty:
    print("WARNING: No data for ^GSPC")
else:
    data.to_csv(os.path.join(INDEX_DIR, "sp500_stock_data.csv"))
    print(f"Saved sp500_stock_data.csv ({len(data)} rows)")

print("Downloading ^VIX (market volatility) ...")
vix = yf.download("^VIX", start="2019-01-01", end="2026-06-01")
if vix.empty:
    print("WARNING: No data for ^VIX")
else:
    vix.to_csv(os.path.join(INDEX_DIR, "vix_stock_data.csv"))
    print(f"Saved vix_stock_data.csv ({len(vix)} rows)")