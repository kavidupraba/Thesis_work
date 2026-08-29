"""
SHAP_analysis.py
================
Purpose:  Generate SHAP (SHapley Additive exPlanations) values for the three
          existing trained classifiers (Logistic Regression, Random Forest,
          XGBoost) using the *current* pipeline unchanged.

          Task 1 & Task 3 (supervisor work items):
            - Task 1: Produce SHAP values for the existing trained models
                      using the existing preprocessing, feature set, model
                      configurations, and evaluation methodology.
            - Task 3: Build a feature-importance table (feature name,
                      description, category, SHAP contribution) from the
                      SHAP results and existing feature definitions.

Important:  This script deliberately reuses the exact feature engineering
            (EXCLUDE_RAW set), chronological 80/20 split, and model
            hyperparameters defined in main.py. Nothing is changed apart from
            what SHAP itself needs (an explainer object per model).

Outputs:
  Results/shap/shap_values/{TICKER}_{MODEL}.npy        raw SHAP values (test set)
  Results/shap/feature_importance_by_stock.csv         per-stock mean |SHAP| per feature
  Results/shap/feature_importance_summary.csv          Task 3 table (across-stock mean)
  Plots/shap_global_importance_{model}.png             global importance bar charts
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# CONFIGURATION (mirrors main.py exactly)
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
DATA_DIR = os.path.join(BASE_DIR, "Data")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PLOTS_DIR = os.path.join(BASE_DIR, "Plots")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap")
SHAP_VALUES_DIR = os.path.join(SHAP_DIR, "shap_values")

TEST_SIZE = 0.2
RANDOM_STATE = 42

STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]

os.makedirs(SHAP_VALUES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# FEATURE METADATA (used for Task 3 feature-importance table)
# ---------------------------------------------------------------------------
FEATURE_META = {
    "sma_10":          ("Trend",  "10-day Simple Moving Average of Close"),
    "sma_50":          ("Trend",  "50-day Simple Moving Average of Close"),
    "ema_12":          ("Trend",  "12-day Exponential Moving Average of Close"),
    "ema_26":          ("Trend",  "26-day Exponential Moving Average of Close"),
    "rsi_14":          ("Momentum", "14-day Relative Strength Index"),
    "macd":            ("Momentum", "MACD line (12/26 EMA difference)"),
    "macd_signal":     ("Momentum", "MACD signal line (9-day EMA of MACD)"),
    "macd_histogram":  ("Momentum", "MACD histogram (MACD - signal)"),
    "roc_10":          ("Momentum", "10-day Rate of Change of Close"),
    "bb_upper":        ("Volatility", "Bollinger Band upper band (20-day, 2 std)"),
    "bb_lower":        ("Volatility", "Bollinger Band lower band (20-day, 2 std)"),
    "bb_width":        ("Volatility", "Bollinger Band width ((upper-lower)/mid)"),
    "atr_14":          ("Volatility", "14-day Average True Range"),
    "volume_roc":      ("Volume",  "Volume Rate of Change (5-day pct change)"),
    "returns_lag1":    ("Other",   "1-day lagged daily return"),
    "returns_lag7":    ("Other",   "7-day lagged daily return"),
    "ma_5":            ("Other",   "5-day Simple Moving Average of Close"),
    "ma_20":           ("Other",   "20-day Simple Moving Average of Close"),
    "ma_50":           ("Other",   "50-day Simple Moving Average of Close"),
    "volatility_5":    ("Volatility", "5-day rolling std of returns"),
    "volatility_20":   ("Volatility", "20-day rolling std of returns"),
    "high_low_range":  ("Other",   "Daily price range (High - Low)"),
    "close_open_change": ("Other", "Intraday price change (Close - Open)"),
}


# ---------------------------------------------------------------------------
# 1. LOAD DATA AND PREPARE FEATURES (identical logic to main.py)
# ---------------------------------------------------------------------------
def load_and_prepare(ticker):
    """Load cleaned data and prepare X, y for classification (same as main.py)."""
    filepath = os.path.join(CLEAN_DIR, f"{ticker}_cleaned.csv")
    if not os.path.exists(filepath):
        print(f"  WARNING: {ticker} cleaned file not found, skipping.")
        return None

    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)

    TARGET_COL = "Close"
    EXCLUDE_RAW = {TARGET_COL, "Open", "High", "Low", "Volume",
                    "Close_lag1", "Close_lag7", "Close_lag21",
                    "High_lag1", "High_lag7", "High_lag21",
                    "Low_lag1", "Low_lag7", "Low_lag21",
                    "Open_lag1", "Open_lag7", "Open_lag21",
                    "returns", "log_returns", "obv", "volume_ma_5"}

    df["target"] = (df[TARGET_COL].shift(-1) > df[TARGET_COL]).astype(int)
    df = df.dropna(subset=["target"])

    df = df.dropna()
    if len(df) == 0:
        print(f"  WARNING: {ticker} has no data after cleaning, skipping.")
        return None

    feature_cols = [c for c in df.columns if c not in EXCLUDE_RAW and c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    return df, X, y, feature_cols


# ---------------------------------------------------------------------------
# 2. TRAIN MODELS AND COMPUTE SHAP VALUES
# ---------------------------------------------------------------------------
def model_logreg(X_train, y_train):
    """Return a fitted Logistic Regression.(same config as main.py)"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_s, y_train)
    return model, scaler


def model_rf(X_train, y_train):
    """Return a fitted Random Forest.(same config as main.py)"""
    model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                   min_samples_leaf=5, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, None


def model_xgb(X_train, y_train):
    """Return a fitted XGBoost.(same config as main.py)"""
    n_up = y_train.sum()
    n_down = len(y_train) - n_up
    scale_pos = n_down / n_up if n_up > 0 else 1
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
                          n_jobs=-1, eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)
    return model, None


def compute_shap(model, scaler, X_train, X_test):
    """Compute SHAP values for the given fitted model on the test set."""
    if isinstance(model, LogisticRegression):
        # Linear explainer needs the scaled data (model was trained on it).
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        masker = shap.maskers.Independent(X_train_s, max_samples=X_train_s.shape[0])
        explainer = shap.LinearExplainer(model, masker,
                                         feature_perturbation="interventional")
        values = explainer.shap_values(X_test_s)
    else:
        # TreeExplainer is exact for RF and XGBoost; no background needed.
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_test)

    # Binary classifiers may return a list or a 3D (samples, features, classes)
    # array. In both cases select the LAST class (UP / class 1).
    if isinstance(values, list):
        values = values[-1]
    if values.ndim == 3:
        values = values[..., -1]

    return np.asarray(values)


# ---------------------------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SHAP ANALYSIS — EXPLAINABLE ML FOR STOCK PREDICTION")
    print("=" * 60)
    print()

    by_stock_rows = []

    for ticker in STOCKS:
        print("-" * 50)
        print(f"Processing {ticker}...")
        data = load_and_prepare(ticker)
        if data is None:
            continue

        df, X, y, feature_cols = data

        split_idx = int(len(df) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        models = {
            "LogisticRegression": model_logreg,
            "RandomForest": model_rf,
            "XGBoost": model_xgb,
        }

        for model_name, factory in models.items():
            try:
                model, scaler = factory(X_train, y_train)
                shap_values = compute_shap(model, scaler, X_train, X_test)
            except Exception as e:
                print(f"  {model_name:25s} | SHAP ERROR: {e}")
                continue

            # Save raw SHAP values (test set) — Task 1 deliverable.
            np.save(os.path.join(SHAP_VALUES_DIR, f"{ticker}_{model_name}.npy"),
                    shap_values)

            # Mean absolute contribution per feature (global explanation).
            mean_abs = np.abs(shap_values).mean(axis=0)
            for i, feat in enumerate(feature_cols):
                by_stock_rows.append({
                    "Stock": ticker,
                    "Model": model_name,
                    "Feature": feat,
                    "MeanAbsSHAP": mean_abs[i],
                })

            print(f"  {model_name:25s} | test rows: {len(X_test)} | "
                  f"features: {len(feature_cols)} | SHAP OK")

        print()

    if not by_stock_rows:
        print("No results collected.")
        return

    by_stock_df = pd.DataFrame(by_stock_rows)
    by_stock_df.to_csv(os.path.join(SHAP_DIR, "feature_importance_by_stock.csv"),
                       index=False)
    print("=" * 60)
    print(f"Saved per-stock importance -> Results/shap/feature_importance_by_stock.csv")

    # ---------------------------------------------------------------------
    # TASK 3: FEATURE-IMPORTANCE TABLE (across-stock mean |SHAP| per model)
    # ---------------------------------------------------------------------
    pivot = by_stock_df.pivot_table(index="Feature", columns="Model",
                                    values="MeanAbsSHAP", aggfunc="mean")

    table_rows = []
    for feat in feature_cols:
        cat = FEATURE_META.get(feat, ("Other", ""))[0]
        desc = FEATURE_META.get(feat, ("Other", ""))[1]
        lr = pivot.loc[feat, "LogisticRegression"] if feat in pivot.index else np.nan
        rf = pivot.loc[feat, "RandomForest"] if feat in pivot.index else np.nan
        xgb = pivot.loc[feat, "XGBoost"] if feat in pivot.index else np.nan
        table_rows.append({
            "Feature": feat,
            "Category": cat,
            "Description": desc,
            "LR_MeanAbsSHAP": lr,
            "RF_MeanAbsSHAP": rf,
            "XGBoost_MeanAbsSHAP": xgb,
            "Overall_MeanAbsSHAP": np.nanmean([lr, rf, xgb]),
        })

    table_df = pd.DataFrame(table_rows)
    table_df = table_df.sort_values("Overall_MeanAbsSHAP", ascending=False)

    # Rank each feature within each model (1 = most important).
    for model in ["LR", "RF", "XGBoost"]:
        col = f"{model}_MeanAbsSHAP"
        table_df[f"{model}_Rank"] = table_df[col].rank(ascending=False).astype(int)

    table_path = os.path.join(SHAP_DIR, "feature_importance_summary.csv")
    table_df.to_csv(table_path, index=False)
    print(f"Saved Task 3 table            -> {table_path}")
    print()

    pd.set_option("display.width", 200)
    print(table_df[["Feature", "Category", "LR_MeanAbsSHAP", "RF_MeanAbsSHAP",
                    "XGBoost_MeanAbsSHAP", "Overall_MeanAbsSHAP"]].to_string(index=False))

    # ---------------------------------------------------------------------
    # SUPPORTING PLOTS: global importance per model
    # ---------------------------------------------------------------------
    for model in ["LogisticRegression", "RandomForest", "XGBoost"]:
        sub = by_stock_df[by_stock_df["Model"] == model]
        mean_imp = sub.groupby("Feature")["MeanAbsSHAP"].mean().sort_values()
        diff = feature_cols  # full feature ordering for consistent y-axis

        plt.figure(figsize=(9, 8))
        plt.barh(range(len(mean_imp)), mean_imp.values)
        plt.yticks(range(len(mean_imp)), mean_imp.index)
        plt.xlabel("Mean |SHAP| value (across stocks, test set)")
        plt.title(f"Global Feature Importance — {model} (SHAP)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"shap_global_importance_{model}.png"),
                    dpi=150)
        plt.close()

    print("\nSaved global importance plots -> Plots/")
    print(f"\nDone. SHAP values stored in Results/shap/shap_values/")


if __name__ == "__main__":
    main()