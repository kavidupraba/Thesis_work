"""
Cross_stock_stability.py
========================
Purpose:  Assess explanation stability across stocks (RQ4). Using the same
          per-stock mean |SHAP| data that underlies Table tab:shap
          (Results/shap/feature_importance_by_stock.csv), we quantify, for
          each model, how stable the per-feature importance ranking is when
          the stock changes:

            1) Rank spread: per-feature mean and std of rank across the 10
               stocks, and the observed min/max rank.
            2) Top-5 membership: how many stocks rank each feature inside
               their top-5 (0..10). A feature whose rank is stable should be
               in the top-5 for most stocks if it matters at all.
            3) Coefficient of variation of mean |SHAP| across stocks.
            4) Pairwise stock agreement: Spearman rho between the rankings of
               every pair of stocks, averaged per model. High and stable
               values would indicate explanations transfer across stocks.

Outputs:
  Results/cross_stock_stability.csv       per-model, per-feature stats
  Results/cross_stock_pairwise_rho.csv    mean pairwise stock rho per model
  Plots/cross_stock_top5_membership.png   top-5 membership frequency (all models)
  Plots/cross_stock_pairwise_rho.png      boxplot of pairwise stock rho per model
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

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
SHAP_DIR = os.path.join(BASE_DIR, "Results", "shap")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PLOTS_DIR = os.path.join(BASE_DIR, "Plots")

MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
N_STOCKS = 10
TOPK = 5


def load_by_stock():
    path = os.path.join(SHAP_DIR, "feature_importance_by_stock.csv")
    df = pd.read_csv(path)
    return df


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_by_stock()
    stocks = sorted(df["Stock"].unique())
    print(f"Dropping anomalies... stocks: {len(stocks)}, rows: {len(df)}")

    rows = []
    pairwise_rows = []

    fig, ax = plt.subplots(figsize=(9, 6))
    box_data = []
    box_labels = []
    x_pos = 0

    for model in MODELS:
        sub = df[df["Model"] == model]
        pivot = sub.pivot_table(index="Feature", columns="Stock",
                                values="MeanAbsSHAP", aggfunc="mean")
        pivot = pivot[stocks]
        n_feat, n_stk = pivot.shape
        if n_feat != 23 or n_stk != N_STOCKS:
            print(f"  WARNING: {model} matrix is {n_feat}x{n_stk}")

        # Per-stock rankings (1 = most important)
        ranks = pivot.rank(ascending=False, method="average")

        # Per-feature statistics
        mean_abs_mean = pivot.mean(axis=1)
        mean_abs_std = pivot.std(axis=1)
        cv = mean_abs_std / mean_abs_mean.replace(0, np.nan)
        rank_mean = ranks.mean(axis=1)
        rank_std = ranks.std(axis=1)
        rank_min = ranks.min(axis=1)
        rank_max = ranks.max(axis=1)
        top5_membership = (ranks <= TOPK).sum(axis=1)

        pooled_rank = mean_abs_mean.rank(ascending=False)  # pooled/across-stock rank

        for feat in pivot.index:
            rows.append({
                "Model": model,
                "Feature": feat,
                "MeanAbsSHAP_mean": mean_abs_mean[feat],
                "MeanAbsSHAP_std": mean_abs_std[feat],
                "CoefVar": cv[feat],
                "Rank_mean": rank_mean[feat],
                "Rank_std": rank_std[feat],
                "Rank_min": rank_min[feat],
                "Rank_max": rank_max[feat],
                "Top5_membership": int(top5_membership[feat]),
                "Pooled_rank": int(pooled_rank[feat]),
            })

        # Pairwise stock agreement (Spearman between stock rankings)
        rhos = []
        for i in range(n_stk):
            for j in range(i + 1, n_stk):
                r, _ = spearmanr(pivot.iloc[:, i].values, pivot.iloc[:, j].values)
                if not np.isnan(r):
                    rhos.append(r)
        mean_rho = float(np.mean(rhos))
        pairwise_rows.append({
            "Model": model,
            "n_pairs": len(rhos),
            "mean_pairwise_rho": mean_rho,
            "std_pairwise_rho": float(np.std(rhos)),
            "min_pairwise_rho": float(np.min(rhos)),
            "max_pairwise_rho": float(np.max(rhos)),
        })
        box_data.append(rhos)
        box_labels.append(model)

        # Plot on shared axis: top-5 membership
        top5 = (ranks <= TOPK).sum(axis=1).sort_values()
        y = np.arange(len(top5))
        ax.barh(y + x_pos * 0.9, top5.values / N_STOCKS, height=0.28,
                label=model)
        x_pos += 1

    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(os.path.join(RESULTS_DIR, "cross_stock_stability.csv"),
                        index=False)

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(os.path.join(RESULTS_DIR, "cross_stock_pairwise_rho.csv"),
                       index=False)

    pd.set_option("display.width", 220)
    print("=" * 72)
    print("CROSS-STOCK EXPLANATION STABILITY (per-stock SHAP rankings)")
    print("=" * 72)
    print(pairwise_df.to_string(index=False))
    print()

    # Compact per-model summary: mean |SHAP| CV + top-5 membership stats
    for model in MODELS:
        m = stability_df[stability_df["Model"] == model]
        print(f"--- {model} ---")
        print(f"  mean CV of mean|SHAP|        : {m['CoefVar'].mean():.3f}")
        print(f"  mean Rank_std               : {m['Rank_std'].mean():.3f}")
        print(f"  features in top-5 for >=8/10 stocks: "
              f"{int((m['Top5_membership'] >= 8).sum())} "
              f"(>=5/10: {int((m['Top5_membership'] >= 5).sum())})")
        print(f"  top-5 pooled features       : "
              f"{list(m.loc[m['Pooled_rank'] <= 5, 'Feature'])}")
        print(f"  top-5 membership counts     : "
              f"{list(m.loc[m['Pooled_rank'] <= 5, 'Top5_membership'])}")
        print()

    # -----------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("Share of stocks ranking feature in top-5")
    ax.set_ylabel("Feature (pooled-ranked per model)")
    ax.set_title("Top-5 SHAP Feature Membership Across Stocks (by model)")
    ax.legend(loc="upper left")
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_stock_top5_membership.png"), dpi=150)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bp = ax2.boxplot(box_data, showmeans=True)
    ax2.set_xticklabels(box_labels)
    ax2.axhline(0, color="grey", ls="--", lw=1)
    ax2.set_ylabel("Pairwise stock Spearman rho (full 23-feature ranking)")
    ax2.set_title("Cross-Stock Ranking Agreement Within Each Model")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_stock_pairwise_rho.png"), dpi=150)
    plt.close()

    print("Saved -> Results/cross_stock_stability.csv, "
          "Results/cross_stock_pairwise_rho.csv")
    print("       -> Plots/cross_stock_top5_membership.png, "
          "Plots/cross_stock_pairwise_rho.png")


if __name__ == "__main__":
    main()