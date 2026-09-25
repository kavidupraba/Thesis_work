"""
Spearman_consistency.py
=======================
Purpose:  Quantify explanation stability across model architectures (RQ3):
          Spearman rank correlation between the Random Forest and XGBoost
          SHAP-based feature-importance rankings.

          For each pilot stock we load the already-computed per-test-row SHAP
          values (Results/shap/shap_values/*.npy), derive the per-feature mean
          absolute contribution, and compare the RF and XGBoost rankings with:

            1) Per-stock Spearman rho on the full 23-feature ranking.
            2) Per-stock Spearman rho restricted to the union of the two
               top-5 feature sets (the thesis's "top-5 ranking agreement").
            3) Jaccard overlap of the two top-5 sets.
            4) A pooled across-stock comparison (mean |SHAP| per feature
               averaged over stocks) for both models.

          The thesis targets a top-5 Spearman rho above 0.7 as evidence that
          explanations agree across architectures.

Method notes:
          - Rankings are derived from mean |SHAP| over the test set per stock,
            mirroring exactly how global importance was computed in
            SHAP_analysis.py (Table: tab:shap).
          - Feature order is taken from the same load_and_prepare() used when
            the SHAP arrays were produced, so columns align with the .npy
            arrays.
          - rho is not defined when the union of top-5 sets has length < 3;
            such stocks are reported as NaN and excluded from the average.

Outputs:
  Results/spearman_consistency.csv        per-stock rho (full 23 + top-5 union),
                                          Jaccard, and agreement flags
  Results/spearman_pooled.csv             pooled full-23 rho and top-5 agreement
  Plots/spearman_top5_agreement.png       bar chart of per-stock Jaccard overlap
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SHAP_analysis import load_and_prepare, STOCKS

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
SHAP_VALUES_DIR = os.path.join(BASE_DIR, "Results", "shap", "shap_values")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PLOTS_DIR = os.path.join(BASE_DIR, "Plots")

MODEL_A = "RandomForest"
MODEL_B = "XGBoost"
TOPK = 5


def per_stock_ranking(ticker, model_name, feature_cols):
    """Mean |SHAP| per feature for one stock/model, ranked 1 = most important."""
    path = os.path.join(SHAP_VALUES_DIR, f"{ticker}_{model_name}.npy")
    values = np.load(path)
    mean_abs = np.abs(values).mean(axis=0)
    ranking = pd.Series(mean_abs, index=feature_cols)
    return ranking


def topk_spearman(rank_a, rank_b, k):
    """Spearman rho between the two rankings, restricted to the union of the
    k most important features of each ranking."""
    top_a = set(rank_a.nlargest(k).index)
    top_b = set(rank_b.nlargest(k).index)
    union = sorted(top_a | top_b)
    if len(union) < 3:
        return np.nan, 0.0, 0.0, union
    sub_a = rank_a.loc[union]
    sub_b = rank_b.loc[union]
    rho, _ = spearmanr(sub_a.values, sub_b.values)
    if np.isnan(rho):
        rho = np.nan
    jaccard = len(top_a & top_b) / len(top_a | top_b)
    return rho, jaccard, len(union), union


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rows = []

    all_a = {}
    all_b = {}
    feature_cols_global = None

    for ticker in STOCKS:
        data = load_and_prepare(ticker)
        if data is None:
            continue
        df, X, y, feature_cols = data
        feature_cols_global = feature_cols

        rank_a = per_stock_ranking(ticker, MODEL_A, feature_cols)
        rank_b = per_stock_ranking(ticker, MODEL_B, feature_cols)

        # 1) full 23-feature Spearman rho
        rho_full, _ = spearmanr(rank_a.values, rank_b.values)
        # 2) top-5-union Spearman rho + 3) Jaccard
        rho_top, jac, n_union, union = topk_spearman(rank_a, rank_b, TOPK)

        rows.append({
            "Stock": ticker,
            "rho_full_23": rho_full,
            "rho_top5_union": rho_top,
            "top5_union_size": n_union,
            "top5_union_features": ";".join(union),
            "jaccard_top5": jac,
            "top5_overlap": jac > 0.5,
        })

        all_a[ticker] = rank_a  # raw importance, not ranking
        all_b[ticker] = rank_b

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "spearman_consistency.csv"), index=False)

    # ---------------------------------------------------------------------
    # POOLED ACROSS STOCKS (keep feature alignment; do NOT pre-sort)
    # ---------------------------------------------------------------------
    pooled_a = pd.concat(all_a, axis=1).mean(axis=1)
    pooled_b = pd.concat(all_b, axis=1).mean(axis=1)

    rho_pooled_full, _ = spearmanr(pooled_a.values, pooled_b.values)
    rho_pooled_top, jac_pooled, n_union_p, union_p = topk_spearman(
        pooled_a, pooled_b, TOPK)

    top_a = list(pooled_a.nlargest(TOPK).index)
    top_b = list(pooled_b.nlargest(TOPK).index)

    pooled_summary = pd.DataFrame({
        "metric": ["rho_full_23", "rho_top5_union", "jaccard_top5",
                   "top5_RandomForest", "top5_XGBoost"],
        "value": [rho_pooled_full, rho_pooled_top, jac_pooled,
                  "|".join(top_a), "|".join(top_b)],
    })
    pooled_summary.to_csv(os.path.join(RESULTS_DIR, "spearman_pooled.csv"),
                          index=False)

    pd.set_option("display.width", 220)
    print("=" * 70)
    print("SPEARMAN CROSS-MODEL EXPLANATION CONSISTENCY (RF vs XGBoost)")
    print("=" * 70)
    print(df[["Stock", "rho_full_23", "rho_top5_union", "jaccard_top5",
              "top5_overlap"]].to_string(index=False))
    print()
    print(f"--- POOLED ACROSS {len(df)} STOCKS ---")
    print(pooled_summary[["metric", "value"]].to_string(index=False))
    print()
    print(f"Mean rho_full_23        : {df['rho_full_23'].mean():.3f} "
          f"(std {df['rho_full_23'].std():.3f})")
    print(f"Mean rho_top5_union     : {df['rho_top5_union'].mean():.3f} "
          f"(std {df['rho_top5_union'].std():.3f})")
    print(f"Mean Jaccard top5       : {df['jaccard_top5'].mean():.3f}")

    # ---------------------------------------------------------------------
    # PLOT: per-stock Jaccard overlap
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2a9d8f" if v else "#e76f51" for v in df["jaccard_top5"] > 0.5]
    ax.bar(df["Stock"], df["jaccard_top5"], color=colors)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jaccard overlap of top-5 feature sets")
    ax.set_title("Top-5 SHAP Feature Set Agreement: Random Forest vs XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "spearman_top5_agreement.png"), dpi=150)
    plt.close()

    print("\nSaved -> Results/spearman_consistency.csv, Results/spearman_pooled.csv")
    print("       -> Plots/spearman_top5_agreement.png")


if __name__ == "__main__":
    main()