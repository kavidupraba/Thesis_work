"""Market_signal_test.py
======================
Index-level test of whether the ^GSPC / ^VIX feature block contains any signal
at all, *before* blaming the stock-level models for failing to use it.

Everything here is read-only with respect to the cleaned datasets.  The target is
the next-day S&P 500 return, built as ``spx_close.pct_change().shift(-1)`` so a
feature row at ``t`` only ever predicts ``t+1``.

Sections
--------
1. Univariate IC: Spearman IC of each market-series feature with next-day S&P
   return, with Newey-West (HAC) t-stats, per period.  HAC matters because
   volatility/return cross-products are serially correlated.
2. Event study: drawdown quintiles and VIX bins -> mean next-day S&P return in
   basis points, up-hit rate, and a moving-block bootstrap CI.
3. Index-level walk-forward: the same protocol shape as the stock models
   (60% minimum train, 20-day steps, logistic regression) on the features whose
   pre-OOS |IC| > 0.05.  Feature selection uses only 2018-2022, never the test
   window.  If index timing cannot beat the base rate, no stock-level model can.
4. Ceiling arithmetic: implied stock-level correlation (rho(stock, market) x IC),
   implied AUC, and the best achievable accuracy at the observed 53% base rate.
   This is the quantitative answer to "why is the gain < 1pp?".
5. Verdict: one line for the index signal and one for the models-vs-prior check
   (read from the latest compare_*_scoring.csv, if present).

Outputs
-------
    Results/market_signal_ic.csv
    Results/market_signal_event_study.csv
    Results/market_signal_timing.csv
    Results/market_signal_ceiling.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from Market_features import (
    MARKET_SERIES_COLS,
    build_market_features,
    load_index_closes,
)

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
CLEAN_DIR = os.environ.get("THESIS_CLEAN_DIR", os.path.join(BASE_DIR, "Data", "cleaned_mkt"))

OOS_START, OOS_END = "2023-02-06", "2026-05-29"
PRE_START, PRE_END = "2018-01-01", "2022-12-31"
FULL_START, FULL_END = "2018-01-01", "2026-12-31"

WINDOWS = [
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021-2022", "2021-01-01", "2022-12-31"),
    ("2023-2026 OOS", OOS_START, OOS_END),
    ("full", FULL_START, FULL_END),
]

MIN_TRAIN = 0.6
STEP = 20
HORIZON = 1


# ---------------------------------------------------------------------------
# Newey-West (HAC) helpers
# ---------------------------------------------------------------------------
def _maxlags(n: int) -> int:
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def nw_long_run(g: np.ndarray, maxlags: int | None = None) -> float:
    """Newey-West long-run variance estimate of a scalar series."""
    g = np.asarray(g, dtype=float)
    n = len(g)
    if maxlags is None:
        maxlags = _maxlags(n)
    s = float(np.mean(g * g))
    for lag in range(1, maxlags + 1):
        w = 1.0 - lag / (maxlags + 1.0)
        s += 2.0 * w * float(np.mean(g[lag:] * g[:-lag]))
    return max(s, 1e-24)


def hac_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS slope of y on x with a Newey-West SE.

    On ranked inputs the slope is the Spearman correlation, and the t-stat is
    the HAC-corrected t for that IC.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    sxx = float(np.mean(xc * xc))
    b = float(np.mean(xc * yc)) / sxx
    resid = yc - b * xc
    lrv = nw_long_run(xc * resid)
    se = float(np.sqrt(lrv / (len(x) * sxx ** 2)))
    return b, se, b / se


def nw_cov(scores: np.ndarray, maxlags: int | None = None) -> np.ndarray:
    """Newey-West long-run covariance matrix of score rows."""
    S = np.asarray(scores, dtype=float)
    n = len(S)
    if maxlags is None:
        maxlags = _maxlags(n)
    B = S.T @ S / n
    for lag in range(1, maxlags + 1):
        w = 1.0 - lag / (maxlags + 1.0)
        G = S[lag:].T @ S[:-lag] / n
        B += w * (G + G.T)
    return B


def logit_hac(y: np.ndarray, X: np.ndarray) -> dict:
    """Logistic regression with a Newey-West sandwich covariance.

    Features are standardised internally so the Newton step is well conditioned;
    per-coefficient t-stats are invariant to that reparameterisation.

    Returns per-coefficient t-stats and a joint Wald test for the slopes.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mu_x, sd_x = X.mean(axis=0), X.std(axis=0)
    sd_x[sd_x == 0] = 1.0
    Xs = (X - mu_x) / sd_x
    Xd = np.column_stack([np.ones(len(y)), Xs])
    beta = np.zeros(Xd.shape[1])
    for _ in range(100):
        eta = np.clip(Xd @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-9)
        A = Xd.T @ (Xd * w[:, None])
        grad = Xd.T @ (y - mu)
        try:
            step = np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break

    eta = np.clip(Xd @ beta, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(mu * (1.0 - mu), 1e-9)
    A = Xd.T @ (Xd * w[:, None])
    Ainv = np.linalg.pinv(A)
    cov = Ainv @ (len(y) * nw_cov(Xd * (y - mu)[:, None])) @ Ainv
    se = np.sqrt(np.maximum(np.diag(cov), 1e-24))
    t = beta / se

    k = Xd.shape[1] - 1
    joint_p = np.nan
    if k >= 1:
        b = beta[1:]
        V = cov[1:, 1:]
        try:
            stat = float(b @ np.linalg.pinv(V) @ b)
            from scipy.stats import chi2
            joint_p = float(1.0 - chi2.cdf(stat, k))
        except np.linalg.LinAlgError:
            pass
    return {"beta": beta, "se": se, "t": t, "joint_p": joint_p}


# ---------------------------------------------------------------------------
# Event-study bootstrap
# ---------------------------------------------------------------------------
def block_boot_mean(x: np.ndarray, n_boot: int = 2000, block: int = 5,
                    seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap of the mean (blocks of consecutive trading days)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([np.nan])
    block = min(block, max(1, n // 2))
    n_blocks = int(np.ceil(n / block))
    last_start = n - block
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, last_start + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[b] = x[idx].mean()
    return out


# ---------------------------------------------------------------------------
# Ceiling arithmetic
# ---------------------------------------------------------------------------
def implied_accuracy(r: float, base_rate: float, n_sim: int = 400_000,
                     seed: int = 0) -> tuple[float, float]:
    """Best accuracy achievable with a signal correlated ``r`` with the latent.

    Latent-direction model: Z ~ N(0,1), Y = 1{Z > z0} with P(Y=1)=base_rate,
    and X ~ N(0,1) with corr(X, Z) = r.  The accuracy-maximising rule is
    "predict up when P(Y=1|X) > 0.5", i.e. r*X > z0.  This is an upper bound:
    the true probability is known exactly, so no model can do better.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sim)
    e = rng.standard_normal(n_sim)
    x = r * z + np.sqrt(max(0.0, 1.0 - r * r)) * e
    z0 = norm.ppf(1.0 - base_rate)
    y = (z > z0).astype(int)
    pred = (r * x > z0).astype(int)
    acc = float(np.mean(pred == y))
    auc = 0.5 + np.arcsin(r) / np.pi  # equal-variance binormal identity
    return acc, auc


# ---------------------------------------------------------------------------
# Section 1: univariate IC
# ---------------------------------------------------------------------------
def ic_table(idx: pd.DataFrame, fwd_ret: pd.Series) -> pd.DataFrame:
    rows = []
    for label, start, end in WINDOWS:
        block = idx.loc[start:end]
        y = fwd_ret.reindex(block.index)
        for c in MARKET_SERIES_COLS:
            x = block[c]
            ok = x.notna() & y.notna()
            n = int(ok.sum())
            if n < 60:
                continue
            ic, se, t = hac_slope(x[ok].rank(), y[ok].rank())
            rows.append({"window": label, "feature": c, "n": n,
                         "IC": ic, "se_hac": se, "t_hac": t,
                         "absIC": abs(ic)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 2: event study
# ---------------------------------------------------------------------------
def event_study(idx: pd.DataFrame, fwd_ret: pd.Series, n_boot: int) -> pd.DataFrame:
    rng_seed = 12345
    rows = []
    frames = {
        "full": idx.loc[FULL_START:FULL_END],
        "2023-2026 OOS": idx.loc[OOS_START:OOS_END],
    }
    vix_bins = [(-np.inf, 15.0, "VIX<15"), (15.0, 20.0, "15-20"),
                (20.0, 30.0, "20-30"), (30.0, np.inf, "VIX>30")]
    for window, block in frames.items():
        r = fwd_ret.reindex(block.index)
        ok = r.notna()
        df = pd.DataFrame({"ret": r[ok], "dd": block.loc[ok, "mkt_drawdown_60"],
                           "vix": block.loc[ok, "vix_level"]}).dropna()
        if df.empty:
            continue

        dd_rank = df["dd"].rank(method="first")
        df["dd_q"] = pd.qcut(dd_rank, 5, labels=[f"Q{i}" for i in range(1, 6)])
        groups = [("drawdown_quintile", str(q), g) for q, g in df.groupby("dd_q", observed=True)]
        for lo, hi, label in vix_bins:
            g = df[(df["vix"] >= lo) & (df["vix"] < hi)]
            groups.append(("vix_bin", label, g))

        for kind, label, g in groups:
            if len(g) < 10:
                continue
            rets = g["ret"].to_numpy(dtype=float)
            draws = block_boot_mean(rets, n_boot=n_boot, block=5,
                                    seed=rng_seed + len(rows))
            lo_ci, hi_ci = np.percentile(draws, [2.5, 97.5])
            rows.append({
                "window": window, "group_type": kind, "group": label,
                "n": int(len(rets)),
                "mean_ret_bp": float(np.mean(rets) * 1e4),
                "up_hit_rate": float(np.mean(rets > 0)),
                "ci_lo_bp": float(lo_ci * 1e4),
                "ci_hi_bp": float(hi_ci * 1e4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 3: index-level walk-forward
# ---------------------------------------------------------------------------
def _wf_index(y: np.ndarray, X: np.ndarray, first_test: int) -> dict:
    """Walk-forward logistic timing with 20-day steps, training strictly before
    each test window.  ``first_test`` is the row index of the first OOS date, so
    the 2023-2026 run is trained on 2018-2023 exactly like the stock models."""
    n = len(y)
    p = np.full(n, np.nan)
    prior = np.full(n, np.nan)
    i = max(first_test, HORIZON + 50)
    while i + STEP <= n:
        tr_end = i - HORIZON
        sc = StandardScaler().fit(X[:tr_end])
        model = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
        model.fit(sc.transform(X[:tr_end]), y[:tr_end])
        p[i:i + STEP] = model.predict_proba(sc.transform(X[i:i + STEP]))[:, 1]
        prior[i:i + STEP] = float(np.mean(y[:tr_end]))
        i += STEP

    valid = ~np.isnan(p)
    yv, pv, prv = y[valid], p[valid], prior[valid]
    pred = (pv > 0.5).astype(int)
    brier = float(np.mean((yv - pv) ** 2))
    brier_prior = float(np.mean((yv - prv) ** 2))
    return {
        "n": int(len(yv)),
        "accuracy": float(accuracy_score(yv, pred)),
        "always_up_accuracy": float(np.mean(yv)),
        "d_acc_vs_always_up": float(accuracy_score(yv, pred) - np.mean(yv)),
        "balanced_accuracy": float(balanced_accuracy_score(yv, pred)),
        "auc": float(roc_auc_score(yv, pv)),
        "brier": brier,
        "brier_prior": brier_prior,
        "brier_skill": 1.0 - brier / brier_prior,
        "prior_threshold_accuracy": float(np.mean((prv > 0.5).astype(int) == yv)),
        "pred_up_share": float(np.mean(pred)),
    }


def _dedupe(cols: list[str], frame: pd.DataFrame, thresh: float = 0.95) -> list[str]:
    """Drop columns that duplicate an already-kept column (pairwise-complete corr)."""
    keep = []
    for c in cols:
        duplicate = False
        for k in keep:
            pair = frame[[c, k]].dropna()
            if len(pair) > 30 and abs(float(np.corrcoef(pair[c], pair[k])[0, 1])) >= thresh:
                duplicate = True
                break
        if not duplicate:
            keep.append(c)
    return keep


def index_timing(idx: pd.DataFrame, fwd_ret: pd.Series, ic: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Select features using ONLY the pre-OOS period, then evaluate 2023-2026.
    pre = ic[(ic["window"] == "2018-2019") | (ic["window"] == "2020") |
             (ic["window"] == "2021-2022")]
    pre_ic = pre.groupby("feature")["IC"].mean()
    pre_ic = pre_ic.reindex(pre_ic.abs().sort_values(ascending=False).index)
    # Rule: rank by |mean signed IC| over 2018-2019, 2020 and 2021-2022, take the
    # top 6, drop duplicate representations (|corr| > 0.95, e.g. vix_log ~
    # vix_level), keep the top 5.  Selection uses only 2018-2022; the evaluation
    # window is never touched.
    chosen = _dedupe(list(pre_ic.head(6).index), idx.loc[PRE_START:PRE_END])[:5]
    if len(chosen) < 2:
        chosen = list(pre_ic.head(5).index)

    y_all = fwd_ret.reindex(idx.index)
    X_all = idx[chosen]
    ok = y_all.notna() & X_all.notna().all(axis=1)
    y = (y_all[ok].to_numpy(dtype=float) > 0.0).astype(int)
    X = X_all[ok].to_numpy(dtype=float)
    dates = ok.index[ok]

    rows = []
    specs = [("2023-2026 OOS", OOS_START), ("full 2018-2026", FULL_START)]
    for label, start in specs:
        first = (int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(start))))
                 if label == "2023-2026 OOS" else int(len(y) * MIN_TRAIN))
        if len(y) - first < 100:
            continue
        d = _wf_index(y, X, first_test=first)
        d["window"] = label
        d["features"] = ", ".join(chosen)
        d["n_features"] = len(chosen)
        d["first_test_date"] = str(dates[min(first, len(dates) - 1)].date())
        rows.append(d)
    return pd.DataFrame(rows), chosen


# ---------------------------------------------------------------------------
# Section 4 helpers: rho(stock, market) from the cleaned panel
# ---------------------------------------------------------------------------
def stock_market_corr(clean_dir: str) -> dict:
    cors = []
    for path in sorted(glob.glob(os.path.join(clean_dir, "*_cleaned.csv"))):
        d = pd.read_csv(path, index_col="Date", parse_dates=True)
        if "mkt_ret_1" not in d.columns:
            continue
        r = d["Close"].pct_change().loc[OOS_START:OOS_END]
        m = d["mkt_ret_1"].loc[OOS_START:OOS_END]
        c = r.corr(m)
        if np.isfinite(c):
            cors.append(c)
    if not cors:
        return {"mean": np.nan, "max": np.nan, "n": 0}
    return {"mean": float(np.mean(cors)), "max": float(np.max(cors)), "n": len(cors)}


def stock_base_rate() -> float | None:
    """OOS up-rate of the stock panel, read from the baseline predictions."""
    path = os.path.join(RESULTS_DIR, "full28_wf_predictions_baseline.csv")
    try:
        d = pd.read_csv(path, usecols=["y"])
    except Exception:
        return None
    return float(d["y"].mean()) if len(d) else None


def models_beat_prior() -> tuple[bool | None, str]:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "compare_*_scoring.csv")))
    if not files:
        return None, "compare_*_scoring.csv not found (run Compare_runs.py first)"
    lines = []
    any_positive = False
    for path in files:
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            ok = np.isfinite(r.get("brier_skill", np.nan)) and r["brier_skill"] > 0
            any_positive |= bool(ok)
            lines.append(f"{os.path.basename(path)}: {r['run']}/{r['Model']} "
                         f"BSS={r['brier_skill']:+.4f} LLS={r['logloss_skill']:+.4f}")
    return any_positive, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_boot: int = 2000) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    closes = load_index_closes()
    idx = build_market_features(closes)
    spx = closes["spx_close"].astype(float)
    fwd_ret = spx.pct_change().shift(-1)

    print("=" * 78)
    print("INDEX-LEVEL MARKET SIGNAL TEST")
    print(f"SPX rows: {len(idx)}  {idx.index.min().date()} .. {idx.index.max().date()}")
    print(f"OOS window: {OOS_START} .. {OOS_END}")
    print("=" * 78)

    # ---- 1) Univariate IC ------------------------------------------------
    ic = ic_table(idx, fwd_ret)
    ic.to_csv(os.path.join(RESULTS_DIR, "market_signal_ic.csv"), index=False)
    print("\n1) UNIVARIATE IC vs NEXT-DAY S&P RETURN (Spearman, HAC t)")
    show = ic[ic["window"] == "2023-2026 OOS"].sort_values("absIC", ascending=False)
    print(show[["feature", "n", "IC", "t_hac", "absIC"]].round(4).to_string(index=False))
    print("\n   Full-sample IC:")
    full = ic[ic["window"] == "full"].sort_values("absIC", ascending=False)
    print(full[["feature", "n", "IC", "t_hac"]].round(4).to_string(index=False))

    stable = (ic.groupby("feature")["IC"]
              .agg(["mean", "min", "max"])
              .assign(sign_stable=lambda d: (d["min"] > 0) | (d["max"] < 0)))
    print("\n   Per-window IC range (stable sign = same sign in every window):")
    print(stable.sort_values("mean", key=lambda s: s.abs(), ascending=False).round(4).to_string())

    # ---- 2) Event study --------------------------------------------------
    ev = event_study(idx, fwd_ret, n_boot)
    ev.to_csv(os.path.join(RESULTS_DIR, "market_signal_event_study.csv"), index=False)
    print("\n2) EVENT STUDY (next-day S&P return, bp; moving-block 95% CI)")
    for window in ev["window"].unique():
        print(f"\n   [{window}]")
        print(ev[ev["window"] == window]
              [["group_type", "group", "n", "mean_ret_bp", "up_hit_rate",
                "ci_lo_bp", "ci_hi_bp"]].round(2).to_string(index=False))
    print("\n   (drawdown Q1 = deepest drawdown, Q5 = at/near the 60-day high)")

    # ---- 3) Index-level walk-forward ------------------------------------
    timing, chosen = index_timing(idx, fwd_ret, ic)
    timing.to_csv(os.path.join(RESULTS_DIR, "market_signal_timing.csv"), index=False)
    print("\n3) INDEX-LEVEL WALK-FORWARD TIMING")
    print(f"   features selected on 2018-2022 only (top 5 distinct by |mean IC|): {chosen}")
    if timing.empty:
        print("   not enough rows")
    else:
        print(timing[["window", "first_test_date", "n", "accuracy", "always_up_accuracy",
                      "d_acc_vs_always_up", "balanced_accuracy", "auc",
                      "brier", "brier_prior", "brier_skill",
                      "prior_threshold_accuracy", "pred_up_share"]]
              .round(4).to_string(index=False))
        if timing["first_test_date"].nunique() == 1:
            print(f"   note: the 60% split of the available series also starts on "
                  f"{timing['first_test_date'].iloc[0]}, so both rows test the same "
                  "window; this is a consistency check, not an extra experiment.")

    # ---- 4) Ceiling arithmetic ------------------------------------------
    print("\n4) CEILING ARITHMETIC (why a weak signal cannot buy 1pp of accuracy)")
    rho = stock_market_corr(CLEAN_DIR)
    print(f"   mean corr(stock, mkt_ret_1) over OOS, {rho['n']} stocks = {rho['mean']:.3f} "
          f"(max {rho['max']:.3f})")
    oos_ic = ic[ic["window"] == "2023-2026 OOS"]
    # Use only features whose IC is individually credible (|HAC t| >= 2); taking
    # the raw maximum would let vix_stress (IC 0.08, t 0.5, ~10 stress days)
    # overstate the ceiling.
    cred = oos_ic[oos_ic["t_hac"].abs() >= 2.0]
    if cred.empty:
        cred = oos_ic.sort_values("absIC", ascending=False).head(2)
    ic_hi = float(cred["absIC"].max())
    ic_mid = float(cred["absIC"].mean())
    print("   credible features (|HAC t| >= 2): "
          + ", ".join(f"{r.feature} ({r.IC:+.3f})" for r in cred.itertuples()))
    base_rate = stock_base_rate()
    if base_rate is None:
        base_rate = float((fwd_ret.reindex(idx.loc[OOS_START:OOS_END].index) > 0).mean())
        print("   NOTE: baseline predictions not readable; using the index up-rate "
              "as the base rate.")
    print(f"   stock OOS up-rate (baseline predictions) = {base_rate:.4f}")
    ceil_rows = []
    for ic_name, ic_val in [("max |IC| with |t|>=2", ic_hi),
                            ("mean |IC| with |t|>=2", ic_mid)]:
        for rho_name, rho_val in [("mean stock corr", rho["mean"]),
                                  ("max stock corr", rho["max"])]:
            r = rho_val * ic_val
            acc, auc = implied_accuracy(r, base_rate)
            ceil_rows.append({
                "IC_basis": ic_name, "IC": ic_val, "rho_basis": rho_name,
                "rho": rho_val, "implied_stock_corr": r,
                "implied_AUC": auc, "best_accuracy": acc,
                "base_rate": base_rate, "gain_vs_always_up_pp": 100 * (acc - base_rate),
            })
    ceil = pd.DataFrame(ceil_rows)
    ceil.to_csv(os.path.join(RESULTS_DIR, "market_signal_ceiling.csv"), index=False)
    print(f"   observed OOS up-rate = {base_rate:.4f}")
    print(ceil[["IC_basis", "IC", "rho_basis", "implied_stock_corr", "implied_AUC",
                "best_accuracy", "gain_vs_always_up_pp"]].round(4).to_string(index=False))
    print("   NOTE: at a 53% up-rate the accuracy-maximising rule is almost always "
          "'up',\n   so the same correlation buys far less accuracy than the balanced "
          "case,\n   where the bound would be arcsin(r)/pi (~1.7pp for r=0.055).")

    # ---- 5) Verdict ------------------------------------------------------
    print("\n5) VERDICT")
    oos_block = idx.loc[OOS_START:OOS_END]
    y_oos = fwd_ret.reindex(oos_block.index)
    ok = y_oos.notna() & oos_block[chosen].notna().all(axis=1)
    oos_fit = logit_hac((y_oos[ok].to_numpy(dtype=float) > 0).astype(int),
                        oos_block.loc[ok, chosen].to_numpy(dtype=float))
    full_block = idx.loc[FULL_START:FULL_END]
    y_full = fwd_ret.reindex(full_block.index)
    okf = y_full.notna() & full_block[chosen].notna().all(axis=1)
    full_fit = logit_hac((y_full[okf].to_numpy(dtype=float) > 0).astype(int),
                         full_block.loc[okf, chosen].to_numpy(dtype=float))
    print("   multivariate logit, HAC sandwich:")
    for label, fit in [("OOS 2023-2026", oos_fit), ("full 2018-2026", full_fit)]:
        ts = ", ".join(f"{f}={t:+.2f}" for f, t in zip(chosen, fit["t"][1:]))
        print(f"     [{label}] t: {ts} | max|t|={np.max(np.abs(fit['t'][1:])):.2f} "
              f"| joint Wald p={fit['joint_p']:.4f}")

    signal_yes = bool((oos_fit["joint_p"] < 0.05) or (full_fit["joint_p"] < 0.05))
    print(f"\n   signal at index level (multivariate joint test): "
          f"{'YES' if signal_yes else 'NO'}")
    beat, detail = models_beat_prior()
    if beat is None:
        print(f"   models beat prior: UNKNOWN ({detail})")
    else:
        print(f"   models beat prior (any BSS > 0 across saved comparisons): "
              f"{'YES' if beat else 'NO'}")
        for line in detail.splitlines():
            print("     " + line)

    ev_excl = int(((ev["ci_lo_bp"] > 0) | (ev["ci_hi_bp"] < 0)).sum())
    print("\n   DECISION GATE (plan section 8)")
    print(f"     BSS <= 0 for all models .............. "
          f"{'YES' if beat is False else ('NO' if beat else 'UNKNOWN')}")
    print(f"     multivariate HAC joint p >= 0.05 ..... "
          f"{'YES' if not signal_yes else 'NO'} (p={min(oos_fit['joint_p'], full_fit['joint_p']):.3f})")
    print(f"     event-study CIs include zero ........ "
          f"{len(ev) - ev_excl}/{len(ev)} cells "
          f"({ev_excl} exclude zero; ~5% expected by chance at 95%)")

    if beat is False and not signal_yes:
        print("     => CLOSE the market-features thread: report the negative result "
              "with the\n        ceiling arithmetic and the power table.")
    elif beat:
        print("     => PROCEED to Phase 2 (regime-covering OOS, then relative-to-SPX "
              "target).")
    else:
        print("     => Run Compare_runs.py first, then re-apply the gate.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=2000,
                   help="bootstrap draws for the event study")
    return p.parse_args()


if __name__ == "__main__":
    main(n_boot=parse_args().n_boot)
