"""
Enhance_features.py
===================
Purpose:  Extend each stock's feature set with Tier-1 market-relative and calendar
          features computed from the full 28-stock universe (no new downloads).

New features (all known at the close of day t; target is close[t+1] > close[t]):
  market_adj_ret_1   : stock's day-t return minus equal-weight market (ex-self) return
  market_adj_ret_5   : trailing 5-day cumulative stock return minus market's
  market_adj_ret_20  : trailing 20-day cumulative stock return minus market's
  rel_mom_10         : 10-day stock momentum minus 10-day market momentum
  sector_adj_ret_1   : stock day-t return minus equal-weight sector return
  sector_adj_ret_5   : trailing 5-day stock return minus sector's
  mom_5, mom_10, mom_20 : simple close-relative-shift momentum at 5/10/20 days
  vol_adj_ret_1      : day-t return scaled by trailing 20-day realised volatility
  pos_vs_sma20       : (close / SMA20 - 1) relative position
  dow, month         : calendar seasonality

Output:   ../Data/cleaned_rel/{TICKER}_rel.csv  (original columns + new features)
"""

import os
import glob
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
CLEAN_DIR = os.path.join(BASE_DIR, "Data", "cleaned")
REL_DIR = os.path.join(BASE_DIR, "Data", "cleaned_rel")
os.makedirs(REL_DIR, exist_ok=True)

SECTORS = {
    "Technology": ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM"],
    "Finance": ["JPM", "GS", "V", "MA", "BAC"],
    "Energy": ["XOM", "CVX", "COP"],
    "Healthcare": ["JNJ", "PFE", "UNH"],
    "Consumer": ["WMT", "PG", "KO", "DIS"],
    "Industrials": ["CAT", "GE", "BA"],
}
TICKER_TO_SECTOR = {t: s for s, ts in SECTORS.items() for t in ts}

ALL_TICKERS = sorted(TICKER_TO_SECTOR)


def load_clean(ticker):
    fp = os.path.join(CLEAN_DIR, f"{ticker}_cleaned.csv")
    df = pd.read_csv(fp, index_col="Date", parse_dates=True)
    return df


def build_market_frames(tickers):
    """Return a DataFrame of daily returns per ticker, and the market/sector
    equal-weight return series (computed excluding the stock itself as needed)."""
    rets = {}
    for tk in tickers:
        df = load_clean(tk)
        rets[tk] = df["Close"].pct_change()
    ret_frame = pd.DataFrame(rets)
    return ret_frame


def market_return_excluding(ret_frame, ticker):
    """Equal-weight market daily return computed WITHOUT the given ticker."""
    others = [c for c in ret_frame.columns if c != ticker]
    return ret_frame[others].mean(axis=1, skipna=True)


def sector_return(ret_frame, sector, ticker):
    """Equal-weight sector daily return computed WITHOUT the given ticker."""
    members = [c for c in ret_frame.columns if TICKER_TO_SECTOR[c] == sector and c != ticker]
    if not members:
        return pd.Series(np.nan, index=ret_frame.index)
    return ret_frame[members].mean(axis=1, skipna=True)


def cumulative(s):
    return (1 + s).cumprod() - 1


def enhance(ticker, ret_frame):
    df = load_clean(ticker)

    close = df["Close"]
    ret1 = close.pct_change()
    mkt = market_return_excluding(ret_frame, ticker)
    sec = sector_return(ret_frame, TICKER_TO_SECTOR[ticker], ticker)

    cum_ret = cumulative(ret1)
    cum_mkt = cumulative(mkt)
    cum_sec = cumulative(sec.fillna(0))

    df["market_adj_ret_1"] = ret1 - mkt
    df["market_adj_ret_5"] = (cum_ret - cum_mkt).diff(5)   # 5-day relative cumulative drift
    df["market_adj_ret_20"] = (cum_ret - cum_mkt).diff(20)
    df["rel_mom_10"] = ((1 + cum_ret).diff(10)) - ((1 + cum_mkt).diff(10))
    df["sector_adj_ret_1"] = ret1 - sec
    df["sector_adj_ret_5"] = (cum_ret - cum_sec).diff(5)

    df["mom_5"] = close / close.shift(5) - 1
    df["mom_10"] = close / close.shift(10) - 1
    df["mom_20"] = close / close.shift(20) - 1

    vol20 = ret1.rolling(20).std()
    df["vol_adj_ret_1"] = ret1 / vol20.replace(0, np.nan)

    sma20 = close.rolling(20).mean()
    df["pos_vs_sma20"] = close / sma20 - 1

    idx = df.index
    df["dow"] = idx.dayofweek
    df["month"] = idx.month

    df = df.dropna()
    out = os.path.join(REL_DIR, f"{ticker}_rel.csv")
    df.to_csv(out)
    print(f"  {ticker}: {len(df)} rows, {len(df.columns)} cols -> {out}")
    return df


def process_all():
    print(f"Processing {len(ALL_TICKERS)} tickers into {REL_DIR}\n")
    ret_frame = build_market_frames(ALL_TICKERS)
    for tk in sorted(TICKER_TO_SECTOR):
        try:
            enhance(tk, ret_frame)
        except Exception as e:
            print(f"  ERROR {tk}: {e}")


if __name__ == "__main__":
    process_all()