"""Test_market_features.py
========================
Blocking leakage audit for the ^GSPC/^VIX feature integration.

Checks
------
1. Truncated market build: features computed on index rows <= t must equal the
   full-history build at t (catches future-looking rolling windows).
2. Truncated stock merge: merging on stock rows <= t must equal the full merge
   at t (catches join/ffill lookahead).
3. AST source audit: no ``bfill``, no negative ``shift``, no centered windows.
4. Coverage: after the first valid market row, no gaps in market features
   (forward-fill limit is 5 days).
5. Correlation sanity: same-day market return correlates with same-day stock
   return; next-day market return does not. A deliberately leaky shift(-1)
   control shows the strong correlation that the honest pipeline must not see.

Run:
    .venv\\Scripts\\python.exe src\\Test_market_features.py
Exit code 0 = pass, 1 = failure.
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd

from Market_features import (
    MARKET_FEATURE_COLS,
    MARKET_SERIES_COLS,
    SP500_PATH,
    VIX_PATH,
    attach_market_features,
    build_market_features,
    load_index_closes,
)
from Preprocess_data import DATA_DIR, read_yfinance_csv

CUT_POINTS = [300, 600, 1000, 1500]
INDEX_CUTS = [300, 600, 1000, 1500, 1800]
TOL = 1e-9

failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def allclose_rows(a: pd.Series, b: pd.Series, cols) -> tuple[bool, str]:
    av = a[cols].to_numpy(dtype=float)
    bv = b[cols].to_numpy(dtype=float)
    ok = np.allclose(av, bv, rtol=0, atol=TOL, equal_nan=True)
    if ok:
        return True, ""
    bad = [c for c, x, y in zip(cols, av, bv)
           if not np.isclose(x, y, rtol=0, atol=TOL, equal_nan=True)]
    return False, f"mismatched columns: {bad[:5]}"


def truncated_market_check(closes: pd.DataFrame) -> None:
    full = build_market_features(closes)
    for cut in INDEX_CUTS:
        if cut >= len(closes):
            continue
        part = build_market_features(closes.iloc[:cut])
        date = closes.index[cut - 1]
        ok, detail = allclose_rows(full.loc[date], part.iloc[-1], MARKET_SERIES_COLS)
        check(f"market build truncated at row {cut} ({date.date()})", ok, detail)


def truncated_stock_check(tickers: list[str], closes: pd.DataFrame) -> None:
    mkt = build_market_features(closes)
    for tk in tickers:
        raw = read_yfinance_csv(os.path.join(DATA_DIR, f"{tk}_stock_data.csv"))
        full = attach_market_features(raw, mkt)
        for cut in CUT_POINTS:
            if cut >= len(raw):
                continue
            part = attach_market_features(raw.iloc[:cut], mkt)
            date = raw.index[cut - 1]
            ok, detail = allclose_rows(full.loc[date], part.iloc[-1], MARKET_FEATURE_COLS)
            if not ok:
                check(f"{tk}: truncated stock merge at row {cut}", ok, detail)
                return
        print(f"  [PASS] {tk}: truncated stock merge at {CUT_POINTS}")


def source_ast_check() -> None:
    path = os.path.join(os.path.dirname(__file__), "Market_features.py")
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "bfill":
                bad.append(f"bfill at line {node.lineno}")
            if attr == "shift":
                for arg in node.args:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        bad.append(f"negative shift at line {node.lineno}")
            for kw in node.keywords:
                if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    bad.append(f"center=True at line {node.lineno}")
    check("no bfill / negative shift / centered windows", not bad, "; ".join(bad))


def coverage_check(tickers: list[str], closes: pd.DataFrame) -> None:
    mkt = build_market_features(closes)
    worst = 0.0
    gaps = 0
    for tk in tickers:
        raw = read_yfinance_csv(os.path.join(DATA_DIR, f"{tk}_stock_data.csv"))
        df = attach_market_features(raw, mkt)
        valid = df[MARKET_FEATURE_COLS].notna().all(axis=1)
        if not valid.any():
            check(f"{tk}: any valid market rows", False)
            return
        first = valid.idxmax()
        after = df.loc[first:][MARKET_FEATURE_COLS].isna().sum().sum()
        gaps += int(after)
        worst = max(worst, float((~valid).sum()))
    check("no market-feature gaps after first valid row", gaps == 0,
          f"{gaps} missing cells; max warm-up rows={int(worst)}")


def correlation_check(tickers: list[str], closes: pd.DataFrame) -> None:
    mkt = build_market_features(closes)
    same, nxt, leaky = [], [], []
    for tk in tickers:
        raw = read_yfinance_csv(os.path.join(DATA_DIR, f"{tk}_stock_data.csv"))
        df = attach_market_features(raw, mkt)
        ret = df["Close"].pct_change()
        same.append(ret.corr(df["mkt_ret_1"]))
        nxt.append(ret.shift(-1).corr(df["mkt_ret_1"]))
        # Deliberate negative control: tomorrow's market return vs tomorrow's
        # stock return. This must be strong; the honest feature must not be.
        leaky.append(ret.shift(-1).corr(df["mkt_ret_1"].shift(-1)))

    same_m, nxt_m, leaky_m = map(lambda xs: float(np.nanmean(xs)), (same, nxt, leaky))
    check("same-day stock/market correlation is positive", same_m > 0.25,
          f"mean={same_m:.3f}")
    check("next-day stock/market correlation is near zero", abs(nxt_m) < 0.15,
          f"mean={nxt_m:.3f}")
    check("leaky shift(-1) control shows strong correlation", leaky_m > 0.25,
          f"mean={leaky_m:.3f}")


def main() -> int:
    print("Market feature leakage audit")
    print("=" * 60)
    print(f"SP500: {SP500_PATH}")
    print(f"VIX:   {VIX_PATH}")

    closes = load_index_closes()
    print(f"Index rows: {len(closes)} "
          f"({closes.index.min().date()} .. {closes.index.max().date()})")
    missing = [c for c in ("spx_close", "vix_close") if c not in closes.columns]
    check("index frames loaded", not missing, f"missing={missing}")
    if failures:
        return 1

    # Non-trading rows (e.g. a stale VIX quote on a US holiday) must be dropped
    # before returns are computed; otherwise the next day's return is NaN and
    # gets forward-filled with an older value.
    feats = build_market_features(closes)
    n_ret_nan = int(feats["mkt_ret_1"].isna().sum())
    check("index build drops non-trading rows", n_ret_nan == 1,
          f"mkt_ret_1 NaN rows={n_ret_nan} (expected 1: the first row)")

    tickers = sorted(
        f.replace("_stock_data.csv", "")
        for f in os.listdir(DATA_DIR) if f.endswith("_stock_data.csv")
    )
    print(f"Universe: {len(tickers)} stocks\n")

    print("1) Truncated market build")
    truncated_market_check(closes)

    print("\n2) Truncated stock merge")
    truncated_stock_check(tickers, closes)

    print("\n3) Source audit")
    source_ast_check()

    print("\n4) Coverage")
    coverage_check(tickers, closes)

    print("\n5) Correlation sanity / leaky control")
    correlation_check(tickers, closes)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
