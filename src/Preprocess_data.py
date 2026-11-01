"""
Preprocess_data.py
==================
Purpose:  Cleans and featurizes raw yfinance CSV files for model consumption.
Pipeline:
  1. Reads raw CSVs from ../Data/ (handles yfinance multi-level headers)
  2. Optionally merges leakage-safe ^GSPC/^VIX market features
  3. Flattens column names to single-level (Open, High, Low, Close, Volume)
  4. Engineers features: technical indicators (RSI, MACD, EMA, ATR, Bollinger, OBV, ROC),
     lags, returns, moving averages, volatility, etc.
  5. Drops NaN rows introduced by lag windows
  6. Removes return outliers (>4-sigma)
  7. Saves cleaned CSVs to ../Data/cleaned/{TICKER}_cleaned.csv
Output:   Per-stock cleaned CSV files and a _pipeline_summary.csv
          in ../Data/cleaned/ with a processing log.

Run:
    python Preprocess_data.py
    python Preprocess_data.py --with-market --out-dir ../Data/cleaned_mkt
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import ta

from Market_features import (
    MARKET_FEATURE_COLS,
    attach_market_features,
    build_market_features,
    load_index_closes,
)

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
    """Engineer feature columns: technical indicators, lags, returns, moving averages, volatility."""

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Trend Indicators ---
    df["sma_10"] = ta.trend.sma_indicator(close, window=10)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)

    # --- Momentum Indicators ---
    df["rsi_14"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    df["roc_10"] = ta.momentum.roc(close, window=10)

    # --- Volatility Indicators ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # --- Volume Indicators ---
    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["volume_roc"] = ta.volume.volume_price_volume_index(close, volume) if False else volume.pct_change(periods=5)

    # --- Price Lags ---
    for col in ["Close", "High", "Low", "Open"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag7"] = df[col].shift(7)
            df[f"{col}_lag21"] = df[col].shift(21)

    # --- Returns ---
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["returns_lag1"] = df["returns"].shift(1)
    df["returns_lag7"] = df["returns"].shift(7)

    # --- Simple Moving Averages (additional) ---
    df["ma_5"] = close.rolling(window=5).mean()
    df["ma_20"] = close.rolling(window=20).mean()
    df["ma_50"] = close.rolling(window=50).mean()

    # --- Volatility ---
    df["volatility_5"] = df["returns"].rolling(window=5).std()
    df["volatility_20"] = df["returns"].rolling(window=20).std()

    # --- Range Features ---
    df["high_low_range"] = high - low
    df["close_open_change"] = close - df["Open"]

    # --- Volume MA ---
    df["volume_ma_5"] = volume.rolling(window=5).mean()

    return df


def clean_pipeline(filepath, with_market=False, out_dir=None, mkt_feats=None):
    """Run the full cleaning + feature engineering pipeline on a single CSV file."""

    ticker = os.path.splitext(os.path.basename(filepath))[0].replace("_stock_data", "")
    out_dir = out_dir or CLEAN_DIR
    os.makedirs(out_dir, exist_ok=True)

    df = read_yfinance_csv(filepath)

    print(f"  {ticker}: {len(df)} rows, columns={list(df.columns)}")

    if with_market:
        if mkt_feats is None:
            mkt_feats = build_market_features(load_index_closes())
        before = len(df)
        df = attach_market_features(df, mkt_feats)
        print(f"  {ticker}: merged {len(MARKET_FEATURE_COLS)} market features "
              f"({before} rows preserved)")

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

    clean_path = os.path.join(out_dir, f"{ticker}_cleaned.csv")
    df.to_csv(clean_path)
    print(f"  Saved: {clean_path} ({len(df)} rows)")

    return df


def process_all(with_market=False, out_dir=None):
    """Process all *_stock_data.csv files in DATA_DIR through the pipeline and save a summary."""

    out_dir = out_dir or CLEAN_DIR
    os.makedirs(out_dir, exist_ok=True)
    mkt_feats = build_market_features(load_index_closes()) if with_market else None

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_stock_data.csv")))
    print(f"Found {len(csv_files)} CSV files to process "
          f"({'with' if with_market else 'without'} market features)\n")
    if with_market:
        print(f"Market features: {', '.join(MARKET_FEATURE_COLS)}\n")

    summary = []
    for fp in csv_files:
        ticker = os.path.splitext(os.path.basename(fp))[0].replace("_stock_data", "")
        try:
            df = clean_pipeline(fp, with_market=with_market, out_dir=out_dir,
                                mkt_feats=mkt_feats)
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
    summary_path = os.path.join(out_dir, "_pipeline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-market", action="store_true",
                        help="merge ^GSPC/^VIX market features from Data/index/")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: Data/cleaned)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(with_market=args.with_market, out_dir=args.out_dir)
