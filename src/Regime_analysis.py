"""
Regime_analysis.py
==================
Purpose:  Regime-stratified analysis (pre-COVID / COVID / post-COVID) on the
          backfilled 2018-2026 data, leakage-free. Answers RQ4's remaining
          component: do predictive skill and SHAP explanation structure differ
          across market regimes?

Regimes (calendar-based):
    Pre-COVID : < 2020-01-01
    COVID     : 2020-01-01 .. 2021-12-31
    Post-COVID: >= 2022-01-01

Part 1 - Predictive performance per regime.
    Within each regime, the daily-direction walk-forward protocol from the
    report's Section sec:validation is applied to the regime slice only:
    expanding train end at i-HORIZON (labels strictly before the test fold),
    per-fold StandardScaler fit, 3 seeds averaged, 20-day test windows,
    min_train = 60% of the regime slice. This keeps every forecast
    strictly out-of-sample with respect to time. Aggregate accuracy and ROC
    AUC per regime x model (pooled over the 10 pilot stocks).

Part 2 - SHAP explanation structure per regime.
    Within each regime, a chronological 80/20 split is used to train the three
    models and compute SHAP on the held-out fifth of the regime
    (methodology identical to Section sec:shapmethod). Mean absolute SHAP is
    averaged across stocks to give a per-regime feature ranking; the top-5
    sets across regimes are compared with the Jaccard overlap and Spearman
    rank correlation (as in Section sec:shapresults).

Outputs:
    Results/regime_performance.csv          per-stock x regime x model metrics
    Results/regime_performance_summary.csv  pooled per regime x model
    Results/regime_shap_rankings.csv        per-regime x model top-5 sets + ranks
    Results/regime_shap_agreement.csv       cross-regime pairwise agreement
    Plots/regime_accuracy.png               pooled accuracy by regime x model
    Plots/regime_shap_agreement.png         top-5 Jaccard per model x regime pair
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the SHAP training configurations and explainer from SHAP_analysis.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from SHAP_analysis import (load_and_prepare, model_logreg, model_rf, model_xgb,
                           compute_shap, STOCKS)

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PLOTS_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RS = 42
HORIZON = 1
STEP = 20
MIN_TRAIN = 0.6
SEEDS = 3

REGIMES = {
    "Pre-COVID": (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31")),
    "COVID": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
    "Post-COVID": (pd.Timestamp("2022-01-01"), pd.Timestamp("2030-01-01")),
}

MODELS_WF = {
    "LR": lambda Xtr, ytr, s: LogisticRegression(
        C=1.0, max_iter=3000, random_state=RS + s).fit(Xtr, ytr),
    "RF": lambda Xtr, ytr, s: RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=RS + s, n_jobs=-1).fit(Xtr, ytr),
    "XGB": lambda Xtr, ytr, s: XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        use_label_encoder=False, random_state=RS + s, n_jobs=-1).fit(Xtr, ytr),
}


# ---------------------------------------------------------------------------
# PART 1: within-regime leakage-free walk-forward performance
# ---------------------------------------------------------------------------
def regime_slice(data, start, end):
    df, X, y, fc = data
    mask = (df.index >= start) & (df.index <= end)
    return df[mask], X[mask], y[mask], fc


def wf_regime(X, y):
    """Leakage-free daily walk-forward on a single regime slice. Returns
    (prob per model dict, valid_y). Mean probability over SEEDS per fold."""
    n = len(y)
    start = int(n * MIN_TRAIN)
    prob = {"LR": np.full(n, np.nan), "RF": np.full(n, np.nan), "XGB": np.full(n, np.nan)}
    i = start
    while i + STEP <= n:
        tr_end = i - HORIZON
        if tr_end < 50:
            i += STEP
            continue
        sc = StandardScaler().fit(X[:tr_end])
        Xtr_z = sc.transform(X[:tr_end])
        Xte_z = sc.transform(X[i:i + STEP])
        Xte = X[i:i + STEP]
        for s in range(SEEDS):
            for name, factory in MODELS_WF.items():
                if name == "LR":
                    m = factory(Xtr_z, y[:tr_end], s)
                    p = m.predict_proba(Xte_z)[:, 1]
                else:
                    m = factory(X[:tr_end], y[:tr_end], s)
                    p = m.predict_proba(Xte)[:, 1]
                prob[name][i:i + STEP] = np.nansum(
                    [prob[name][i:i + STEP], p / SEEDS], axis=0)
        i += STEP
    valid = ~np.isnan(prob["LR"])
    return prob, y[valid]


def part1():
    perf_rows = []
    for regime, (a, b) in REGIMES.items():
        acc = {"LR": [], "RF": [], "XGB": []}
        auc = {"LR": [], "RF": [], "XGB": []}
        for tk in STOCKS:
            data = load_and_prepare(tk)
            if data is None:
                continue
            df, X, y, fc = data
            sub_df, sub_X, sub_y, _ = regime_slice(data, a, b)
            if len(sub_y) < 100:
                print(f"  {regime} {tk}: too short ({len(sub_y)})")
                continue
            prob, yv = wf_regime(sub_X, sub_y)
            valid_mask = ~np.isnan(prob["LR"])
            dates = sub_df.index[valid_mask]
            n = len(yv)
            for name in prob:
                acc_v, auc_v = np.nan, np.nan
                p_ok = prob[name][valid_mask]
                if n:
                    pred = (p_ok > 0.5).astype(int)
                    acc_v = accuracy_score(yv, pred)
                if n and len(np.unique(yv)) > 1:
                    auc_v = roc_auc_score(yv, p_ok)
                acc[name].append(acc_v)
                auc[name].append(auc_v)
                perf_rows.append({
                    "Regime": regime, "Stock": tk, "Model": name,
                    "N_days": n,
                    "Start": str(dates[0].date()) if len(dates) else "",
                    "End": str(dates[-1].date()) if len(dates) else "",
                    "Accuracy": acc_v, "ROC_AUC": auc_v,
                })
        print(f"[{regime}] mean acc: " + ", ".join(
            f"{k}={np.nanmean(v):.3f}" for k, v in acc.items()) +
              " | mean AUC: " + ", ".join(
            f"{k}={np.nanmean(v):.3f}" for k, v in auc.items()))

    out = pd.DataFrame(perf_rows)
    out.to_csv(os.path.join(RESULTS_DIR, "regime_performance.csv"), index=False)

    pooled = out.groupby(["Regime", "Model"])[["Accuracy", "ROC_AUC"]].mean()
    pooled = pooled.reset_index()
    pooled.to_csv(os.path.join(RESULTS_DIR, "regime_performance_summary.csv"),
                  index=False)
    print("\nPooled summary:\n", pooled.to_string(index=False))

    # Plot
    models_ordered = ["LR", "RF", "XGB"]
    regimes_ordered = ["Pre-COVID", "COVID", "Post-COVID"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.26
    for k, m in enumerate(models_ordered):
        vals = [pooled[(pooled.Regime == r) & (pooled.Model == m)]["Accuracy"].iloc[0]
                for r in regimes_ordered]
        ax.bar([i + k * width for i in range(len(regimes_ordered))], vals,
               width, label=m, color=["#4c72b0", "#dd8452", "#55a868"][k])
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks([i + width for i in range(len(regimes_ordered))])
    ax.set_xticklabels(regimes_ordered)
    ax.set_ylabel("Pooled accuracy")
    ax.set_ylim(0.4, 0.6)
    ax.set_title("Regime-stratified accuracy (leakage-free walk-forward)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "regime_accuracy.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# PART 2: per-regime SHAP rankings and their cross-regime agreement
# ---------------------------------------------------------------------------
def shap_rankings_regime(data, start, end, feature_cols):
    """Per-regime per-model mean |SHAP| per feature from an 80/20 split."""
    df, X, y, _ = data
    mask = (df.index >= start) & (df.index <= end)
    Xr, yr = X[mask], y[mask]
    split = int(len(Xr) * 0.8)
    if split < 60 or len(Xr) - split < 10:
        return None
    Xtr, Xte, ytr, yte = Xr[:split], Xr[split:], yr[:split], yr[split:]
    out = {}
    for name, factory in [("LR", model_logreg), ("RF", model_rf), ("XGB", model_xgb)]:
        try:
            model, scaler = factory(Xtr, ytr)
            vals = compute_shap(model, scaler, Xtr, Xte)
            out[name] = pd.Series(np.abs(vals).mean(axis=0), index=feature_cols)
        except Exception as e:
            print(f"    SHAP error {name}: {e}")
    return out


def topk_sets(series_store, features, k=5):
    """Return top-k feature sets per model from pooled mean-|SHAP| dict."""
    sets = {}
    ranks = {}
    for m, s in series_store.items():
        # pool across stocks: we receive a DataFrame features x stocks
        pooled = s.mean(axis=1)
        sets[m] = set(pooled.nlargest(k).index)
        ranks[m] = pooled.rank(ascending=False)
    return sets, ranks


def part2():
    # per-regime pooled importance: DataFrame[feature x stock] per model
    storage = {}  # regime -> model -> dict stock -> Series
    for regime, (a, b) in REGIMES.items():
        storage[regime] = {m: {} for m in ["LR", "RF", "XGB"]}
        for tk in STOCKS:
            data = load_and_prepare(tk)
            if data is None:
                continue
            df, X, y, fc = data
            res = shap_rankings_regime(data, a, b, fc)
            if res is None:
                continue
            for m, s in res.items():
                storage[regime][m][tk] = s

    # Build pooled feature x stock frames per regime x model
    sets_by_regime = {}
    ranking_by_regime = {}
    for regime, models_dict in storage.items():
        sets_by_regime[regime] = {}
        ranking_by_regime[regime] = {}
        for m, stocks in models_dict.items():
            if not stocks:
                continue
            pooled = pd.DataFrame(stocks)          # features x stocks
            sets_by_regime[regime][m] = set(pooled.mean(axis=1).nlargest(5).index)
            ranking_by_regime[regime][m] = pooled.mean(axis=1).rank(ascending=False)

    # ExpXort the top-5 sets per regime x model
    rows = []
    for regime, models_dict in sets_by_regime.items():
        for m, s in models_dict.items():
            rows.append({"Regime": regime, "Model": m,
                         "Top5": "|".join(sorted(s, key=str))})
    rank_df = pd.DataFrame(rows)
    rank_df.to_csv(os.path.join(RESULTS_DIR, "regime_shap_rankings.csv"),
                   index=False)
    print("\nPer-regime top-5 SHAP sets:\n", rank_df.to_string(index=False))

    # Cross-regime agreement per model
    regimes = list(sets_by_regime.keys())
    agree_rows = []
    for m in ["LR", "RF", "XGB"]:
        for i in range(len(regimes)):
            for j in range(i + 1, len(regimes)):
                ra, rb = regimes[i], regimes[j]
                sa = sets_by_regime[ra].get(m, set())
                sb = sets_by_regime[rb].get(m, set())
                jac = len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan
                # Spearman on union of the two top-5 sets
                union = sorted(sa | sb)
                rho = np.nan
                if len(union) >= 3 and m in ranking_by_regime[ra] and m in ranking_by_regime[rb]:
                    va = ranking_by_regime[ra][m].reindex(union).values
                    vb = ranking_by_regime[rb][m].reindex(union).values
                    rho = spearmanr(va, vb).statistic
                agree_rows.append({"Model": m, "Regime_A": ra, "Regime_B": rb,
                                   "Jaccard": jac, "Spearman": rho})
    agree_df = pd.DataFrame(agree_rows)
    agree_df.to_csv(os.path.join(RESULTS_DIR, "regime_shap_agreement.csv"),
                    index=False)
    print("\nCross-regime SHAP agreement:\n", agree_df.to_string(index=False))

    # Plot: Jaccard per model x regime pair
    fig, ax = plt.subplots(figsize=(9, 5))
    pairs = [(f"{r[1]['Regime_A'][:5]}-{r[1]['Regime_B'][:5]}", r)
             for r in agree_df.groupby(["Regime_A", "Regime_B"])]
    import collections
    labels = []
    data = {m: [] for m in ["LR", "RF", "XGB"]}
    for (ra, rb), g in agree_df.groupby(["Regime_A", "Regime_B"]):
        labels.append(f"{ra[:7]}\n{rb[:7]}")
        for m in ["LR", "RF", "XGB"]:
            v = g[g.Model == m]["Jaccard"].iloc[0]
            data[m].append(v)
    x = np.arange(len(labels))
    w = 0.26
    for k, m in enumerate(["LR", "RF", "XGB"]):
        ax.bar(x + k * w, data[m], w, label=m, color=["#4c72b0", "#dd8452", "#55a868"][k])
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Jaccard of top-5 sets")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-Regime Top-5 SHAP Agreement")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "regime_shap_agreement.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("REGIME ANALYSIS (pre-COVID / COVID / post-COVID), leakage-free")
    print("=" * 70)
    print("\n--- PART 1: walk-forward performance ---")
    part1()
    print("\n--- PART 2: SHAP rankings per regime ---")
    part2()
    print("\nDone. Saved CSVs to Results/, plots to Plots/.")