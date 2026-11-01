"""Market_features.py
===================
Leakage-safe market-context features built from the two index downloads:

    Data/index/sp500_stock_data.csv   ^GSPC
    Data/index/vix_stock_data.csv     ^VIX

Timing contract
---------------
A feature row at date ``t`` may use information observable at the close of
day ``t`` and earlier.  The pipeline target is ``Close[t+1] > Close[t]``, so
same-day index closes are allowed (exactly like the existing ``Close[t]``,
``rsi_14[t]``, ... features).  This module never uses ``shift(-k)``,
``bfill``, centered windows, or future index rows.

Usage
-----
    from Market_features import attach_market_features, load_index_closes, \
        build_market_features

    mkt = build_market_features(load_index_closes())
    df = attach_market_features(stock_df, mkt)

``stock_df`` must have a DatetimeIndex named ``Date`` and a ``Close`` column.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
INDEX_DIR = os.path.join(BASE_DIR, "Data", "index")
SP500_PATH = os.path.join(INDEX_DIR, "sp500_stock_data.csv")
VIX_PATH = os.path.join(INDEX_DIR, "vix_stock_data.csv")

# Columns produced by build_market_features()
MARKET_SERIES_COLS = [
    "mkt_ret_1",
    "mkt_ret_5",
    "mkt_ret_20",
    "mkt_vol_20",
    "mkt_vol_60",
    "mkt_above_sma50",
    "mkt_drawdown_60",
    "vix_level",
    "vix_log",
    "vix_chg_1",
    "vix_chg_5",
    "vix_z_20",
    "vix_high",
    "vix_stress",
]

# Columns produced by attach_market_features() in addition to the above.
MARKET_STOCK_COLS = [
    "excess_ret_1",
    "excess_ret_5",
    "excess_ret_20",
    "beta_60",
    "corr_60",
]

MARKET_FEATURE_COLS = MARKET_SERIES_COLS + MARKET_STOCK_COLS


def _read_index_csv(path: str) -> pd.Series:
    """Read a yfinance index CSV that has a three-row header.

    The downloaded files look like::

        Price,Close,High,Low,Open,Volume
        Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC
        Date,,,,,
        2019-01-02,2510.03,...

    ``header=[0, 1]`` reads the two header levels; the extra ``Date`` row is
    coerced to ``NaT`` and dropped.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Index file not found: {path}\nRun src/Download_index_data.py first."
        )

    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

    if "Close" not in df.columns:
        raise ValueError(f"No Close column in {path}; columns={list(df.columns)}")

    s = df["Close"].astype(float)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def load_index_closes() -> pd.DataFrame:
    """Return a DatetimeIndex frame with ``spx_close`` and ``vix_close``."""
    spx = _read_index_csv(SP500_PATH).rename("spx_close")
    vix = _read_index_csv(VIX_PATH).rename("vix_close")
    out = pd.concat([spx, vix], axis=1, sort=True).sort_index()
    return out


def build_market_features(idx: pd.DataFrame) -> pd.DataFrame:
    """Build market-only features from a frame of ``spx_close``/``vix_close``.

    Every value at row ``t`` uses only rows ``<= t``.
    """
    if not {"spx_close", "vix_close"}.issubset(idx.columns):
        raise ValueError("index frame must contain spx_close and vix_close")

    idx = idx.sort_index()
    spx = idx["spx_close"].astype(float)
    vix = idx["vix_close"].astype(float)

    # ^VIX occasionally reports a stale quote on days the equity market is
    # closed (e.g. US holidays). The union of the two series would otherwise
    # create a phantom row with a missing SPX close, making the next day's
    # return NaN and forward-filling a stale value. Keep SPX trading days only.
    traded = spx.notna()
    if not traded.any():
        raise ValueError("spx_close is entirely missing")
    idx, spx, vix = idx.loc[traded], spx.loc[traded], vix.loc[traded]
    spx_ret = spx.pct_change()

    f = pd.DataFrame(index=idx.index)
    f["mkt_ret_1"] = spx_ret
    f["mkt_ret_5"] = spx.pct_change(5)
    f["mkt_ret_20"] = spx.pct_change(20)
    f["mkt_vol_20"] = spx_ret.rolling(20, min_periods=15).std()
    f["mkt_vol_60"] = spx_ret.rolling(60, min_periods=40).std()

    sma50 = spx.rolling(50, min_periods=40).mean()
    f["mkt_above_sma50"] = np.where(sma50.isna(), np.nan, (spx > sma50).astype(float))
    roll_max_60 = spx.rolling(60, min_periods=40).max()
    f["mkt_drawdown_60"] = spx / roll_max_60 - 1.0

    f["vix_level"] = vix
    f["vix_log"] = np.log(vix.where(vix > 0))
    f["vix_chg_1"] = vix.pct_change()
    f["vix_chg_5"] = vix.pct_change(5)
    mu = vix.rolling(20, min_periods=15).mean()
    sd = vix.rolling(20, min_periods=15).std()
    f["vix_z_20"] = (vix - mu) / sd.replace(0, np.nan)
    f["vix_high"] = np.where(vix.isna(), np.nan, (vix >= 20).astype(float))
    f["vix_stress"] = np.where(vix.isna(), np.nan, (vix >= 30).astype(float))

    return f


def attach_market_features(stock_df: pd.DataFrame, mkt_feats: pd.DataFrame) -> pd.DataFrame:
    """Left-join market features onto a stock frame and add stock-relative ones.

    Missing index dates are forward-filled with a short limit (never
    backward-filled), so a holiday gap cannot import future information.
    """
    df = stock_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()

    df = df.join(mkt_feats, how="left")
    cols = list(mkt_feats.columns)
    df[cols] = df[cols].ffill(limit=5)

    stock_ret = df["Close"].pct_change()
    df["excess_ret_1"] = stock_ret - df["mkt_ret_1"]
    df["excess_ret_5"] = df["Close"].pct_change(5) - df["mkt_ret_5"]
    df["excess_ret_20"] = df["Close"].pct_change(20) - df["mkt_ret_20"]

    cov = stock_ret.rolling(60, min_periods=40).cov(df["mkt_ret_1"])
    var = df["mkt_ret_1"].rolling(60, min_periods=40).var()
    df["beta_60"] = cov / var.replace(0, np.nan)
    df["corr_60"] = stock_ret.rolling(60, min_periods=40).corr(df["mkt_ret_1"])

    return df


def build_for_stock(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: load the index files once and attach to a stock."""
    return attach_market_features(stock_df, build_market_features(load_index_closes()))


if __name__ == "__main__":
    closes = load_index_closes()
    feats = build_market_features(closes)
    print(f"Index rows: {len(closes)}")
    print(f"Date range: {closes.index.min().date()} .. {closes.index.max().date()}")
    print(f"Market series features: {len(MARKET_SERIES_COLS)}")
    print(feats.tail(3).to_string())
