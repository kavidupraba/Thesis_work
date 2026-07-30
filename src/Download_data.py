"""
Download_data.py
=================
Purpose:  Downloads historical stock price data from Yahoo Finance for a list of tickers.
Output:   Saves CSV files to ../Data/{TICKER}_stock_data.csv for each ticker.
          Each CSV contains columns: Open, High, Low, Close, Volume with a multi-level header
          (Price level, Ticker level) and Date as the row index.
"""

import yfinance as yf
import os

DATA_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data"

tickers = [
    # Technology
    "AAPL",   # Apple - stable tech
    "NVDA",   # NVIDIA - AI / semiconductor
    "MSFT",   # Microsoft - large cap tech
    "GOOGL",  # Alphabet - tech giant
    "AMZN",   # Amazon - e-commerce / cloud
    "META",   # Meta - social media
    "TSLA",   # Tesla - EV / highly volatile
    "AVGO",   # Broadcom - semiconductor
    "ORCL",   # Oracle - enterprise software
    "CRM",    # Salesforce - CRM software

    # Finance
    "JPM",    # JPMorgan - banking
    "GS",     # Goldman Sachs - investment banking
    "V",      # Visa - payments
    "MA",     # Mastercard - payments
    "BAC",    # Bank of America - banking

    # Energy
    "XOM",    # Exxon Mobil - oil & energy
    "CVX",    # Chevron - oil & energy
    "COP",    # ConocoPhillips - oil & gas

    # Healthcare
    "JNJ",    # Johnson & Johnson - healthcare
    "PFE",    # Pfizer - pharmaceuticals
    "UNH",    # UnitedHealth - health insurance

    # Consumer
    "WMT",    # Walmart - retail
    "PG",     # Procter & Gamble - consumer goods
    "KO",     # Coca-Cola - beverages
    "DIS",    # Disney - entertainment

    # Industrials
    "CAT",    # Caterpillar - heavy equipment
    "GE",     # General Electric - industrial conglomerate
    "BA",     # Boeing - aerospace
]

os.makedirs(DATA_DIR, exist_ok=True)

for ticker in tickers:
    print(f"Downloading {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end="2026-06-01")

    if data.empty:
        print(f"WARNING: No data for {ticker}, skipping.")
        continue

    data.to_csv(os.path.join(DATA_DIR, f"{ticker}_stock_data.csv"))
    print(f"Saved {ticker}_stock_data.csv ({len(data)} rows)")