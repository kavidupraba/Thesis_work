"""
Backfill_data.py
================
Purpose:  Extend the raw price history backwards to 2018-01-01 so that the
          thesis can stratify the analysis into pre-COVID / COVID / post-COVID
          regimes (the cleaned feature window currently starts mid-March 2020,
          at the start of the COVID crash, because of the 50-day indicator
          warm-up).

Method:
    For each of the 28 raw *_stock_data.csv files, download daily OHLCV for
    2018-01-01 .. 2019-12-31 with yfinance using the same adjustment policy as
    the existing files (split/dividend adjusted), and prepend the rows to the
    existing CSV, preserving its exact text format:

        Price,Close,High,Low,Open,Volume
        Ticker,<TK>,<TK>,<TK>,<TK>,<TK>
        Date,,,,,
        <data rows...>

    The download end date is pinned to 2020-01-01 (first trading day strictly
    before the existing files, which begin 2020-01-02), so the 2020-2026 data
    is never re-downloaded or modified. Existing results therefore remain
    valid; the pilot/test windows are unchanged.

Outputs:
    Overwrites Data/*_stock_data.csv (prepended rows only),
    prints a boundary-continuity check for each ticker.
"""

import os
import time
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data"
START = "2018-01-01"
END = "2020-01-01"  # exclusive; first trading day of 2020 is 2020-01-02

COL_ORDER = ["Close", "High", "Low", "Open", "Volume"]


def fetch_backfill(ticker):
    df = yf.download(ticker, start=START, end=END, auto_adjust=True,
                     progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Price" in df.columns:
        df = df.drop(columns=["Price"])
    df = df[COL_ORDER]
    df.index.name = "Date"
    return df.sort_index()


def read_existing(path):
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(how="all")
    df = df[df.index.notna()]
    df.columns = df.columns.get_level_values(0)
    df = df[COL_ORDER]
    return df


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_stock_data.csv")))
    print(f"Found {len(files)} raw files to backfill\n")

    report = []
    for fp in files:
        ticker = os.path.splitext(os.path.basename(fp))[0].replace("_stock_data", "")
        existing = read_existing(fp)
        # Keep only the real 2020+ rows as the anchor scale (any pre-2020 rows
        # already in the file are dropped: they were fetched at an inconsistent
        # adjustment baseline and will be replaced by the rebased backfill).
        existing = existing[existing.index >= pd.Timestamp("2020-01-02")]
        anchor = existing.index.min()
        if existing.empty:
            print(f"  {ticker}: no 2020+ rows, skipping")
            continue

        new = fetch_backfill(ticker)
        if new.empty:
            print(f"  {ticker}: backfill empty, skipping")
            continue

        # ----- Rebase the backfilled block onto the existing series' scale ----
        # A fresh yfinance fetch adjusts history using corporate actions up to
        # today, whereas the existing rows were fetched earlier; scales can
        # therefore differ by up to several percent at the boundary. To keep
        # the 2020-2026 window byte-identical and the feature series
        # continuous across the boundary, we rescale the backfill so that its
        # final close matches the close on the first 2020 row in the existing
        # file.
        factor = float(existing.loc[anchor, "Close"]) / float(new.iloc[-1]["Close"])
        for c in ["Close", "High", "Low", "Open"]:
            new[c] = new[c] * factor

        merged = pd.concat([new, existing])
        merged = merged.loc[~merged.index.duplicated(keep="last")].sort_index()

        # Boundary continuity (backfilled last row vs existing first row)
        boundary = existing.index.min()
        prev_close = merged.loc[:boundary].iloc[-2]["Close"]
        curr_close = merged.loc[boundary]["Close"]
        drift = abs(prev_close / curr_close - 1.0)

        # Write back in the exact original format
        tk = ticker
        with open(fp, "w") as fh:
            fh.write("Price,Close,High,Low,Open,Volume\n")
            fh.write(f"Ticker,{tk},{tk},{tk},{tk},{tk}\n")
            fh.write("Date,,,,,\n")
            for date, row in merged.iterrows():
                fh.write(
                    f"{date.date()},{row['Close']:.12g},{row['High']:.12g},"
                    f"{row['Low']:.12g},{row['Open']:.12g},{int(row['Volume'])}\n"
                )

        report.append({
            "Ticker": ticker,
            "rows_added": len(new),
            "new_start": str(new.index.min().date()),
            "old_start": str(existing.index.min().date()),
            "boundary_drift": drift,
        })
        print(f"  {ticker}: added {len(new)} rows "
              f"({new.index.min().date()}..{new.index.max().date()}), "
              f"boundary drift {drift:.6f}")
        time.sleep(0.5)

    print("\nSaved:", os.path.join(DATA_DIR, "_backfill_report.csv"))
    pd.DataFrame(report).to_csv(os.path.join(DATA_DIR, "_backfill_report.csv"),
                                index=False)
    print(pd.DataFrame(report).to_string(index=False))


if __name__ == "__main__":
    main()