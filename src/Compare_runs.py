"""Compare_runs.py
================
Paired A/B comparison of two Full_run.py walk-forward runs (e.g. technical-only
baseline vs technical+market).

Both runs must share the identical protocol (HORIZON/STEP/MIN_TRAIN/SEEDS and
model configs); only the feature matrix differs.  This script joins the saved
out-of-sample predictions on ``(Stock, Date)`` and reports:

  1. Pooled accuracy / balanced accuracy / MCC / AUC per model for both runs.
  2. Delta (market - baseline) for each metric with a **date-block bootstrap**
     CI.  Market features are identical across stocks on a given date, so the
     effective sample size is the number of dates, not stock-days; resampling
     whole dates preserves that dependence.
  3. Paired McNemar on the exact same stock-dates per model.
  4. Per-stock deltas and the count of stocks that improve.

Usage:
    python src/Compare_runs.py --baseline baseline --market market
    python src/Compare_runs.py --baseline baseline --market market --n-boot 5000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2 as _chi2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

MODELS = {"LR": "p_LR", "RF": "p_RF", "XGB": "p_XGB"}
METRICS = ("Accuracy", "BalancedAccuracy", "MCC", "AUC")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p > 0.5
    out = {
        "Accuracy": float(accuracy_score(y, pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
    }
    try:
        out["AUC"] = float(roc_auc_score(y, p))
    except ValueError:
        out["AUC"] = float("nan")
    return out


def mcnemar_exact(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(min(2.0 * binomtest(min(b, c), b + c, 0.5).pvalue, 1.0))


def mcnemar_chi2(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(1 - _chi2.cdf(stat, 1))


# ---------------------------------------------------------------------------
# Date-block bootstrap
# ---------------------------------------------------------------------------
def date_blocks(df: pd.DataFrame) -> list[np.ndarray]:
    """Positional index arrays, one per trading date (sorted)."""
    return [g.index.to_numpy() for _, g in df.groupby("Date", sort=True)]


def bootstrap_deltas(y, p_base, p_mkt, blocks, n_boot, rng):
    """Return dict metric -> array of (market - baseline) bootstrap deltas."""
    n_dates = len(blocks)
    draws = np.empty((n_boot, len(METRICS)), dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n_dates, n_dates)
        idx = np.concatenate([blocks[i] for i in pick])
        yb, pb, pm = y[idx], p_base[idx], p_mkt[idx]
        mb = compute_metrics(yb, pb)
        mm = compute_metrics(yb, pm)
        draws[b] = [mm[m] - mb[m] for m in METRICS]
    return {m: draws[:, j] for j, m in enumerate(METRICS)}


def boot_summary(deltas: np.ndarray) -> dict:
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0))
    return {"Delta": float(np.mean(deltas)), "ci_lo": float(lo),
            "ci_hi": float(hi), "p_boot": float(min(p, 1.0))}


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def load_predictions(tag: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f"full28_wf_predictions_{tag}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing predictions for tag '{tag}': {path}\n"
            f"Run: python src/Full_run.py --tag {tag} --data-dir <dir>"
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df["Stock"] = df["Stock"].astype(str)
    return df


def compare(base_tag: str, mkt_tag: str, n_boot: int = 2000, seed: int = 42) -> None:
    base = load_predictions(base_tag)
    mkt = load_predictions(mkt_tag)

    joined = base.merge(
        mkt, on=["Stock", "Date"], how="inner", suffixes=("_base", "_mkt")
    )
    if joined.empty:
        raise ValueError("No overlapping (Stock, Date) rows between the two runs.")

    n_base_only = len(base) - len(joined)
    n_mkt_only = len(mkt) - len(joined)

    # Drop rows where any model probability is missing.
    prob_cols = [f"{c}_{s}" for c in MODELS.values() for s in ("base", "mkt")]
    joined = joined.dropna(subset=prob_cols).reset_index(drop=True)

    print("=" * 74)
    print(f"BASELINE: {base_tag}   rows={len(base)}")
    print(f"MARKET  : {mkt_tag}   rows={len(mkt)}")
    print(f"JOINED  : {len(joined)} stock-dates, {joined['Date'].nunique()} unique dates, "
          f"{joined['Stock'].nunique()} stocks")
    print(f"UNMATCHED: {n_base_only} baseline-only, {n_mkt_only} market-only")
    print("=" * 74)

    # Sanity: labels must agree where both runs produce them.
    y_mismatch = int((joined["y_base"] != joined["y_mkt"]).sum())
    if y_mismatch:
        print(f"WARNING: {y_mismatch} rows have different y labels between runs.")

    y = joined["y_mkt"].to_numpy(dtype=int)
    blocks = date_blocks(joined)
    rng = np.random.default_rng(seed)

    # ---- Pooled metrics -------------------------------------------------
    pooled_rows, delta_rows, mc_rows = [], [], []
    for model, col in MODELS.items():
        p_b = joined[f"{col}_base"].to_numpy(dtype=float)
        p_m = joined[f"{col}_mkt"].to_numpy(dtype=float)
        mb = compute_metrics(y, p_b)
        mm = compute_metrics(y, p_m)

        for metric in METRICS:
            pooled_rows.append({
                "Model": model, "Metric": metric, "n": len(y),
                "Baseline": mb[metric], "Market": mm[metric],
                "Delta": mm[metric] - mb[metric],
            })

        deltas = bootstrap_deltas(y, p_b, p_m, blocks, n_boot, rng)
        for metric in METRICS:
            s = boot_summary(deltas[metric])
            s.update({"Model": model, "Metric": metric, "n": len(y),
                      "Baseline": mb[metric], "Market": mm[metric],
                      "n_boot": n_boot})
            delta_rows.append(s)

        correct_b = (p_b > 0.5).astype(int) == y
        correct_m = (p_m > 0.5).astype(int) == y
        b = int(np.sum(correct_b & ~correct_m))
        c = int(np.sum(~correct_b & correct_m))
        mc_rows.append({
            "Model": model, "n": len(y),
            "base_correct_mkt_wrong": b, "base_wrong_mkt_correct": c,
            "acc_base": mb["Accuracy"], "acc_market": mm["Accuracy"],
            "p_exact": mcnemar_exact(b, c), "p_chi2": mcnemar_chi2(b, c),
        })

    pooled = pd.DataFrame(pooled_rows)
    deltas_df = pd.DataFrame(delta_rows)
    mc_df = pd.DataFrame(mc_rows)

    # ---- Per-stock deltas ----------------------------------------------
    stock_rows = []
    for (stock,), g in joined.groupby(["Stock"], sort=True):
        yv = g["y_mkt"].to_numpy(dtype=int)
        for model, col in MODELS.items():
            mb = compute_metrics(yv, g[f"{col}_base"].to_numpy(dtype=float))
            mm = compute_metrics(yv, g[f"{col}_mkt"].to_numpy(dtype=float))
            stock_rows.append({
                "Stock": stock, "Model": model, "n": len(yv),
                "acc_base": mb["Accuracy"], "acc_market": mm["Accuracy"],
                "d_acc": mm["Accuracy"] - mb["Accuracy"],
                "d_bacc": mm["BalancedAccuracy"] - mb["BalancedAccuracy"],
                "d_mcc": mm["MCC"] - mb["MCC"],
                "d_auc": mm["AUC"] - mb["AUC"],
            })
    per_stock = pd.DataFrame(stock_rows)

    # ---- Date-level paired accuracy ------------------------------------
    date_rows = []
    for date, g in joined.groupby("Date", sort=True):
        yv = g["y_mkt"].to_numpy(dtype=int)
        for model, col in MODELS.items():
            ab = float(np.mean(((g[f"{col}_base"].to_numpy() > 0.5).astype(int)) == yv))
            am = float(np.mean(((g[f"{col}_mkt"].to_numpy() > 0.5).astype(int)) == yv))
            date_rows.append({"Date": date, "Model": model,
                              "acc_base": ab, "acc_market": am, "d_acc": am - ab})
    per_date = pd.DataFrame(date_rows)

    # ---- Print ----------------------------------------------------------
    show = pooled.pivot(index="Model", columns="Metric", values=["Baseline", "Market", "Delta"])
    print("\nPOOLED METRICS (market - baseline)")
    print(show.round(4).to_string())

    print("\nDELTA WITH DATE-BLOCK BOOTSTRAP (95% CI)")
    cols = ["Model", "Metric", "Baseline", "Market", "Delta", "ci_lo", "ci_hi", "p_boot"]
    print(deltas_df[cols].round(4).to_string(index=False))

    print("\nPAIRED McNEMAR (same stock-dates)")
    print(mc_df.round(4).to_string(index=False))

    print("\nPER-STOCK ACCURACY DELTA (how many of "
          f"{per_stock['Stock'].nunique()} stocks improve)")
    for model in MODELS:
        s = per_stock[per_stock["Model"] == model]
        wins = int((s["d_acc"] > 0).sum())
        losses = int((s["d_acc"] < 0).sum())
        ties = int((s["d_acc"] == 0).sum())
        p_sign = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
        print(f"  {model:4s}: wins={wins:2d} losses={losses:2d} ties={ties:2d} "
              f"mean d_acc={s['d_acc'].mean():+.4f} sign-test p={p_sign:.3f}")

    print("\nPER-DATE ACCURACY DELTA")
    for model in MODELS:
        s = per_date[per_date["Model"] == model]
        print(f"  {model:4s}: mean d_acc={s['d_acc'].mean():+.4f} "
              f"dates better={int((s['d_acc'] > 0).sum())} "
              f"worse={int((s['d_acc'] < 0).sum())}")

    # ---- Save -----------------------------------------------------------
    suffix = f"{base_tag}_vs_{mkt_tag}"
    pooled.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_pooled.csv"), index=False)
    deltas_df.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_deltas.csv"), index=False)
    mc_df.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_mcnemar.csv"), index=False)
    per_stock.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_per_stock.csv"), index=False)
    per_date.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_per_date.csv"), index=False)
    print(f"\nSaved: Results/compare_{suffix}_*.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="baseline", help="baseline run tag")
    p.add_argument("--market", default="market", help="market run tag")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap draws")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    compare(a.baseline, a.market, n_boot=a.n_boot, seed=a.seed)
