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
  5. Naive benchmarks (always-up at the base rate, and a prior-threshold rule)
     so model accuracy is read against the ~53% drift baseline, not against 50%.
  6. Proper scores (Brier, log-loss) and skill against the expanding-window
     ``p_prior`` recorded by Full_run.py, with date-block bootstrap CIs.
  7. Calibration (quantile reliability, ECE, slope of y ~ logit(p)) and a power
     block (minimum detectable effect) so "null" is not read as "zero".

Brier and LogLoss are losses: a negative delta means the market run is better.

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
# Accuracy-style metrics are higher-is-better; Brier and LogLoss are losses, so a
# NEGATIVE delta means the market run is better.  The printout says so explicitly.
METRICS = ("Accuracy", "BalancedAccuracy", "MCC", "AUC", "Brier", "LogLoss")
LOSS_METRICS = ("Brier", "LogLoss")
EPS = 1e-6


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p > 0.5
    out = {
        "Accuracy": float(accuracy_score(y, pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "Brier": float(np.mean((y - p) ** 2)),
        "LogLoss": _logloss(y, p),
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
# Benchmarks, proper scoring, calibration
# ---------------------------------------------------------------------------
def calibration_fit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope/intercept of ``y ~ logit(p)`` by Newton-Raphson (unpenalised).

    A well-calibrated model has slope 1; a model with no information has slope 0.
    """
    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-12)
        grad = X.T @ (y - mu)
        hess = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[1]), float(beta[0])


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Quantile-binned reliability table and expected calibration error."""
    order = np.argsort(p, kind="mergesort")
    rows, ece = [], 0.0
    for k, idx in enumerate(np.array_split(order, n_bins), start=1):
        if len(idx) == 0:
            continue
        p_mean = float(np.mean(p[idx]))
        obs = float(np.mean(y[idx]))
        ece += (len(idx) / len(p)) * abs(obs - p_mean)
        rows.append({"bin": k, "n": int(len(idx)),
                     "p_min": float(np.min(p[idx])), "p_max": float(np.max(p[idx])),
                     "p_mean": p_mean, "obs_rate": obs,
                     "diff": obs - p_mean})
    return pd.DataFrame(rows), float(ece)


def bootstrap_scoring(y, p, prior, blocks, n_boot, rng):
    """Paired date-block bootstrap for skill, calibration and the naive bench."""
    n_dates = len(blocks)
    out = {k: np.empty(n_boot, dtype=float)
           for k in ("d_acc_up", "d_brier_up", "bss", "lls", "slope")}
    for b in range(n_boot):
        pick = rng.integers(0, n_dates, n_dates)
        idx = np.concatenate([blocks[i] for i in pick])
        yb, pb, prb = y[idx], p[idx], prior[idx]
        acc = float(np.mean((pb > 0.5).astype(int) == yb))
        mse_m = float(np.mean((yb - pb) ** 2))
        mse_p = float(np.mean((yb - prb) ** 2))
        out["d_acc_up"][b] = acc - float(np.mean(yb))
        out["d_brier_up"][b] = mse_m - mse_p
        out["bss"][b] = 1.0 - mse_m / mse_p
        out["lls"][b] = 1.0 - _logloss(yb, pb) / _logloss(yb, prb)
        out["slope"][b] = calibration_fit(yb, pb)[0]
    return out


def score_run(y, p, prior, blocks, n_boot, seed) -> dict:
    """Point estimates + bootstrap CIs for one (run, model) pair."""
    acc = float(np.mean((p > 0.5).astype(int) == y))
    brier = float(np.mean((y - p) ** 2))
    brier_prior = float(np.mean((y - prior) ** 2))
    ll = _logloss(y, p)
    ll_prior = _logloss(y, prior)
    slope, intercept = calibration_fit(y, p)
    _, ece = reliability_table(y, p)
    boot = bootstrap_scoring(y, p, prior, blocks, n_boot,
                             np.random.default_rng(seed))
    ci = {k: tuple(float(v) for v in np.percentile(a, [2.5, 97.5]))
          for k, a in boot.items()}
    return {
        "accuracy": acc,
        "always_up_accuracy": float(np.mean(y)),
        "d_acc_vs_always_up": acc - float(np.mean(y)),
        "d_acc_ci_lo": ci["d_acc_up"][0], "d_acc_ci_hi": ci["d_acc_up"][1],
        "prior_threshold_accuracy": float(
            np.mean((prior > 0.5).astype(int) == y)),
        "brier": brier,
        "brier_prior": brier_prior,
        "d_brier_vs_prior": brier - brier_prior,
        "d_brier_ci_lo": ci["d_brier_up"][0], "d_brier_ci_hi": ci["d_brier_up"][1],
        "brier_skill": 1.0 - brier / brier_prior,
        "bss_ci_lo": ci["bss"][0], "bss_ci_hi": ci["bss"][1],
        "logloss": ll,
        "logloss_prior": ll_prior,
        "logloss_skill": 1.0 - ll / ll_prior,
        "lls_ci_lo": ci["lls"][0], "lls_ci_hi": ci["lls"][1],
        "calib_slope": slope, "calib_intercept": intercept,
        "calib_slope_ci_lo": ci["slope"][0], "calib_slope_ci_hi": ci["slope"][1],
        "ece": ece,
    }


def mde_table(se_pp: float, n_dates: int) -> pd.DataFrame:
    """Independent dates needed for 80% power (two-sided alpha=0.05, z=2.80)."""
    rows = []
    for eff in (0.010, 0.005, 0.003, 0.002):
        se_target = eff / 2.80
        n_req = n_dates * (se_pp / 100.0 / se_target) ** 2
        rows.append({"effect_pp": eff * 100, "dates_needed": int(round(n_req)),
                     "trading_years": round(n_req / 252, 1)})
    return pd.DataFrame(rows)


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
    deltas_all = {}
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
        deltas_all[model] = deltas
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

    # ---- Benchmarks, proper scores, calibration -------------------------
    # p_prior is the expanding-window training up-share for the fold predicting
    # each row, so it is strictly out of sample and is the reference forecast for
    # Brier/log-loss skill. It is run-specific (a run that drops more warm-up rows
    # has different fold boundaries), hence one prior per run.
    for col in ("p_prior_base", "p_prior_mkt"):
        if col not in joined.columns:
            raise ValueError(
                f"'{col}' missing. Re-run Full_run.py so predictions contain "
                "p_prior (see Results/_backup_predictions_*.csv for the old files)."
            )
    prior_base = joined["p_prior_base"].to_numpy(dtype=float)
    prior_mkt = joined["p_prior_mkt"].to_numpy(dtype=float)
    n_prior_diff = int(np.sum(prior_base != prior_mkt))

    run_ids = {"base": base_tag, "mkt": mkt_tag}
    prior_cols = {"base": "p_prior_base", "mkt": "p_prior_mkt"}
    scoring_rows, calib_frames = [], []
    for run_key, tag_name in run_ids.items():
        prior = joined[prior_cols[run_key]].to_numpy(dtype=float)
        for model, col in MODELS.items():
            p = joined[f"{col}_{run_key}"].to_numpy(dtype=float)
            s = score_run(y, p, prior, blocks, n_boot, seed=seed + 1)
            s.update({"Model": model, "run": tag_name, "n": len(y)})
            scoring_rows.append(s)
            tab, _ = reliability_table(y, p)
            tab.insert(0, "Model", model)
            tab.insert(0, "run", tag_name)
            calib_frames.append(tab)
    scoring = pd.DataFrame(scoring_rows)
    calibration = pd.concat(calib_frames, ignore_index=True)
    se_acc = {m: float(np.std(deltas_all[m]["Accuracy"])) for m in MODELS}

    # ---- Print ----------------------------------------------------------
    show = pooled[pooled["Metric"].isin(["Accuracy", "BalancedAccuracy", "MCC", "AUC"])]
    show = show.pivot(index="Model", columns="Metric", values=["Baseline", "Market", "Delta"])
    print("\nPOOLED METRICS (market - baseline)")
    print(show.round(4).to_string())

    print("\nDELTA WITH DATE-BLOCK BOOTSTRAP (95% CI)")
    print("  (Brier/LogLoss are losses: a NEGATIVE delta means the market run is better)")
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

    # ---- Benchmarks / scoring / calibration / power ---------------------
    base_rate = float(np.mean(y))
    print("\nBENCHMARKS ON THE SAME STOCK-DATES")
    print(f"  always_up        : acc={base_rate:.4f}  bacc=0.5000  MCC=0.0000  AUC=0.5000")
    for run_key, tag_name in run_ids.items():
        prior = joined[prior_cols[run_key]].to_numpy(dtype=float)
        acc_p = float(np.mean((prior > 0.5).astype(int) == y))
        print(f"  prior_threshold  : [{tag_name:12s}] acc={acc_p:.4f} "
              "(predict up when p_prior > 0.5)")
    if n_prior_diff:
        print(f"  NOTE: the two runs' expanding priors differ on {n_prior_diff}/{len(y)} "
              "joined rows (different warm-up drop, hence different folds); skill "
              "scores below use each run's own prior.")

    acc_cols = ["run", "Model", "accuracy", "always_up_accuracy",
                "d_acc_vs_always_up", "d_acc_ci_lo", "d_acc_ci_hi",
                "prior_threshold_accuracy"]
    print("\nACCURACY vs NAIVE BENCHMARKS")
    print(scoring[acc_cols].round(4).to_string(index=False))

    sc_cols = ["run", "Model", "brier", "brier_prior", "brier_skill",
               "bss_ci_lo", "bss_ci_hi", "logloss", "logloss_prior",
               "logloss_skill", "lls_ci_lo", "lls_ci_hi"]
    print("\nPROPER SCORES vs EXPANDING-WINDOW PRIOR "
          "(higher skill = better; <= 0 = no information over the prior)")
    print(scoring[sc_cols].round(4).to_string(index=False))

    cal_cols = ["run", "Model", "ece", "calib_slope", "calib_slope_ci_lo",
                "calib_slope_ci_hi", "calib_intercept"]
    print("\nCALIBRATION (quantile ECE; slope of y ~ logit(p): 1=calibrated, 0=no info)")
    print(scoring[cal_cols].round(4).to_string(index=False))

    print("\nPOWER (observed date-block bootstrap SE of the accuracy delta)")
    for model in MODELS:
        print(f"  {model:4s}: SE={se_acc[model] * 100:.2f} pp -> "
              f"95% CI half-width {1.96 * se_acc[model] * 100:.2f} pp")
    se_mean_pp = float(np.mean(list(se_acc.values()))) * 100.0
    print(f"  independent date blocks = {len(blocks)}; dates needed at 80% power:")
    print(mde_table(se_mean_pp, len(blocks)).to_string(index=False,
                                                       float_format=lambda v: f"{v:.1f}"))

    print("\nMULTIPLE COMPARISONS: 3 models x 6 metrics (+ calibration) is a family; "
          "read these as descriptive, or apply Holm within a metric family.")
    print("AUC deltas use the paired date-block bootstrap; DeLong is not used.")

    # ---- Save -----------------------------------------------------------
    suffix = f"{base_tag}_vs_{mkt_tag}"
    pooled.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_pooled.csv"), index=False)
    deltas_df.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_deltas.csv"), index=False)
    mc_df.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_mcnemar.csv"), index=False)
    per_stock.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_per_stock.csv"), index=False)
    per_date.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_per_date.csv"), index=False)
    scoring.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_scoring.csv"), index=False)
    calibration.to_csv(os.path.join(RESULTS_DIR, f"compare_{suffix}_calibration.csv"), index=False)
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
