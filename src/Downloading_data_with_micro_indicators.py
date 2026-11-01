import yfinance as yf
import pandas as pd
import os

DATA_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data"
os.makedirs(DATA_DIR, exist_ok=True)

tickers = [
    # Technology
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
    # Finance
    "JPM", "GS", "V", "MA", "BAC",
    # Energy
    "XOM", "CVX", "COP",
    # Healthcare
    "JNJ", "PFE", "UNH",
    # Consumer
    "WMT", "PG", "KO", "DIS",
    # Industrials
    "CAT", "GE", "BA"
]

# 1. Download macro indicators ONCE and prepare them
print("Downloading macro indicators (VIX, TNX, Oil, Gold)...")
macro_symbols = ["^VIX", "^TNX", "CL=F", "GC=F"]
macro_data = yf.download(macro_symbols, start="2020-01-01", end="2026-06-01")["Close"]

# Clean column names just in case
macro_data.columns = ["Oil", "Gold", "VIX",
                      "TNX"]  # Adjust order if needed based on yfinance output, or use dictionary lookup below

# CRITICAL: Shift macro data by 1 day to prevent data leakage (yesterday's close for today's prediction)
macro_data = macro_data.shift(1)
macro_data = macro_data.rename(columns={
    "VIX": "VIX_Lag1",
    "TNX": "TNX_Lag1",
    "CL=F": "Oil_Lag1",
    "GC=F": "Gold_Lag1"
})

# 2. Loop through each company, merge, and save
for ticker in tickers:
    print(f"Processing {ticker}...")
    stock_df = yf.download(ticker, start="2020-01-01", end="2026-06-01")

    if stock_df.empty:
        print(f"WARNING: No data for {ticker}, skipping.")
        continue

    # Flatten MultiIndex columns if yfinance returns them (depends on version)
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)

    # Merge stock data with pre-shifted macro data on dates
    combined_df = stock_df.join(macro_data, how="inner")

    # Drop any rows where macro data might be missing
    combined_df = combined_df.dropna()

    # Save to the requested new filename format
    filename = f"{ticker}_stock_data_with_indicators.csv"
    filepath = os.path.join(DATA_DIR, filename)

    combined_df.to_csv(filepath)
    print(f"Saved {filename} ({len(combined_df)} rows)")

print("All datasets successfully updated with indicators!")