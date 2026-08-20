"""
main.py
=======
Purpose:  Orchestrate the full pipeline:
          1. Preprocess raw data (if not already done)
          2. Train and evaluate LR, RF, XGBoost classifiers on multiple stocks
          3. Collect all results into a summary CSV for the mid-term report
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
DATA_DIR = os.path.join(BASE_DIR, "Data")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PLOTS_DIR = os.path.join(BASE_DIR, "Plots")

TEST_SIZE = 0.2
RANDOM_STATE = 42

STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. PREPROCESS DATA (run if cleaned files missing)
# ---------------------------------------------------------------------------
def preprocess_data():
    """Run preprocessing if cleaned files don't exist."""
    import Preprocess_data
    if not os.path.exists(os.path.join(CLEAN_DIR, "_pipeline_summary.csv")):
        print("=" * 60)
        print("STEP 1: Preprocessing raw data...")
        print("=" * 60)
        Preprocess_data.process_all()
    else:
        print("Cleaned data already exists. Skipping preprocessing.\n")


# ---------------------------------------------------------------------------
# 2. LOAD DATA AND PREPARE FEATURES
# ---------------------------------------------------------------------------
def load_and_prepare(ticker):
    """Load cleaned data and prepare X, y for classification."""
    filepath = os.path.join(CLEAN_DIR, f"{ticker}_cleaned.csv")
    if not os.path.exists(filepath):
        print(f"  WARNING: {ticker} cleaned file not found, skipping.")
        return None

    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)

    TARGET_COL = "Close"
    # Exclude raw price columns (non-stationary) — only keep derived indicators
    EXCLUDE_RAW = {TARGET_COL, "Open", "High", "Low", "Volume",
                    "Close_lag1", "Close_lag7", "Close_lag21",
                    "High_lag1", "High_lag7", "High_lag21",
                    "Low_lag1", "Low_lag7", "Low_lag21",
                    "Open_lag1", "Open_lag7", "Open_lag21",
                    "returns", "log_returns", "obv", "volume_ma_5"}

    df["target"] = (df[TARGET_COL].shift(-1) > df[TARGET_COL]).astype(int)
    df = df.dropna(subset=["target"])

    # Drop any remaining NaN rows in features
    df = df.dropna()
    if len(df) == 0:
        print(f"  WARNING: {ticker} has no data after cleaning, skipping.")
        return None

    feature_cols = [c for c in df.columns if c not in EXCLUDE_RAW and c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    return df, X, y, feature_cols


# ---------------------------------------------------------------------------
# 3. TRAIN AND EVALUATE A SINGLE MODEL
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba):
    """Compute all classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_evaluate_lr(X_train, y_train, X_test, y_test):
    """Logistic Regression with scaling."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    return evaluate(y_test, y_pred, y_proba)


def train_evaluate_rf(X_train, y_train, X_test, y_test):
    """Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_pred, y_proba)


def train_evaluate_xgb(X_train, y_train, X_test, y_test):
    """XGBoost Classifier."""
    n_up = y_train.sum()
    n_down = len(y_train) - n_up
    scale_pos = n_down / n_up if n_up > 0 else 1

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_pred, y_proba)


# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPLAINABLE ML FOR STOCK PREDICTION — MID-TERM PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Preprocess
    preprocess_data()

    # Step 2: Run models on all stocks
    all_results = []

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

        train_start = df.index[0].date()
        train_end = df.index[split_idx - 1].date()
        test_start = df.index[split_idx].date()
        test_end = df.index[-1].date()

        print(f"  Train: {len(X_train)} samples ({train_start} to {train_end})")
        print(f"  Test:  {len(X_test)} samples ({test_start} to {test_end})")

        # Train each model
        models = {
            "LogisticRegression": train_evaluate_lr,
            "RandomForest": train_evaluate_rf,
            "XGBoost": train_evaluate_xgb,
        }

        for model_name, train_fn in models.items():
            try:
                metrics = train_fn(X_train, y_train, X_test, y_test)
                result = {
                    "Stock": ticker,
                    "Model": model_name,
                    "Train_Size": len(X_train),
                    "Test_Size": len(X_test),
                    "Test_Start": test_start,
                    "Test_End": test_end,
                    **{f"test_{k}": v for k, v in metrics.items()},
                }
                all_results.append(result)
                print(f"  {model_name:25s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f} | MCC: {metrics['mcc']:.4f}")
            except Exception as e:
                print(f"  {model_name:25s} | ERROR: {e}")

        print()

    # Step 5: Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(RESULTS_DIR, "model_comparison_midterm.csv")
        results_df.to_csv(results_path, index=False)
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        # Summary table: mean metrics per model
        summary = results_df.groupby("Model")[
            ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_mcc", "test_roc_auc"]
        ].agg(["mean", "std"])
        print(summary.to_string())
        print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
