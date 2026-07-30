"""
Preprocess_data.py
==================
Purpose:  Cleans and featurizes raw yfinance CSV files for model consumption.
Pipeline:
  1. Reads raw CSVs from ../Data/ (handles yfinance multi-level headers)
  2. Flattens column names to single-level (Open, High, Low, Close, Volume)
  3. Engineers features: lags, returns, moving averages, volatility, etc.
  4. Drops NaN rows introduced by lag windows
  5. Removes return outliers (>4-sigma)
  6. Saves cleaned CSVs to ../Data/cleaned/{TICKER}_cleaned.csv
Output:   Per-stock cleaned CSV files with ~29 columns and a _pipeline_summary.csv
          in ../Data/cleaned/ with a processing log.
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data"
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
os.makedirs(CLEAN_DIR, exist_ok=True)


def read_yfinance_csv(filepath):
    """Read a yfinance CSV with multi-level headers and return a clean single-level DataFrame."""

    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)

    df.columns = df.columns.get_level_values(0)

    for col in ["Close", "High", "Low", "Open"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].astype(int)

    return df


def add_features(df):
    """Engineer feature columns: lags, returns, log returns, moving averages, volatility, ranges."""

    df = df.copy()
    for col in ["Close", "High", "Low", "Open"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag7"] = df[col].shift(7)
            df[f"{col}_lag21"] = df[col].shift(21)

    if "Close" in df.columns:
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["returns_lag1"] = df["returns"].shift(1)
        df["returns_lag7"] = df["returns"].shift(7)

        df["ma_5"] = df["Close"].rolling(window=5).mean()
        df["ma_20"] = df["Close"].rolling(window=20).mean()
        df["ma_50"] = df["Close"].rolling(window=50).mean()

        df["volatility_5"] = df["returns"].rolling(window=5).std()
        df["volatility_20"] = df["returns"].rolling(window=20).std()

        df["high_low_range"] = df["High"] - df["Low"]
        df["close_open_change"] = df["Close"] - df["Open"]

        df["volume_ma_5"] = df["Volume"].rolling(window=5).mean()

    return df


def clean_pipeline(filepath):
    """Run the full cleaning + feature engineering pipeline on a single CSV file."""

    ticker = os.path.splitext(os.path.basename(filepath))[0].replace("_stock_data", "")

    df = read_yfinance_csv(filepath)

    print(f"  {ticker}: {len(df)} rows, columns={list(df.columns)}")

    df = add_features(df)

    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped:
        print(f"  {ticker}: dropped {dropped} rows with NaN")

    outliers_before = len(df)
    for col in ["returns", "log_returns"]:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            df = df[(np.abs(df[col] - mean) <= 4 * std)]
    outliers_removed = outliers_before - len(df)
    if outliers_removed:
        print(f"  {ticker}: removed {outliers_removed} return outliers (4-sigma)")

    df = df.sort_index()

    clean_path = os.path.join(CLEAN_DIR, f"{ticker}_cleaned.csv")
    df.to_csv(clean_path)
    print(f"  Saved: {clean_path} ({len(df)} rows)")

    return df


def process_all():
    """Process all *_stock_data.csv files in DATA_DIR through the pipeline and save a summary."""

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_stock_data.csv")))
    print(f"Found {len(csv_files)} CSV files to process\n")

    summary = []
    for fp in csv_files:
        ticker = os.path.splitext(os.path.basename(fp))[0].replace("_stock_data", "")
        try:
            df = clean_pipeline(fp)
            summary.append({
                "ticker": ticker,
                "rows": len(df),
                "date_start": str(df.index.min().date()),
                "date_end": str(df.index.max().date()),
                "columns": len(df.columns),
            })
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            summary.append({"ticker": ticker, "error": str(e)})

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(CLEAN_DIR, "_pipeline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    process_all()
