"""
XGBoost_model.py
================
Purpose:  Train an XGBoost Classifier to predict the DIRECTION of
          the next-day stock price movement (Up / Down) using engineered features.
Output:   Console prints of train/test classification metrics, confusion matrix,
          and feature importance chart.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
)

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------
TICKER = "AAPL"
CLEAN_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data\cleaned"
PLOTS_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Plots"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------------------------
filepath = os.path.join(CLEAN_DIR, f"{TICKER}_cleaned.csv")
if not os.path.exists(filepath):
    print(f"ERROR: Cleaned file not found: {filepath}")
    print("Run Preprocess_data.py first.")
    sys.exit(1)

df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
print(f"Loaded {TICKER}: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
print()


# ---------------------------------------------------------------------------
# 2. DEFINE FEATURES (X) AND BINARY TARGET (y)
# ---------------------------------------------------------------------------
TARGET_COL = "Close"
EXCLUDE = {TARGET_COL, "returns", "log_returns"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

df["target"] = (df[TARGET_COL].shift(-1) > df[TARGET_COL]).astype(int)
df = df.dropna(subset=["target"])

X = df[FEATURE_COLS].values
y = df["target"].values

n_up = y.sum()
n_down = len(y) - n_up
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: Next-day direction (1=UP, 0=DOWN)")
print(f"Class distribution — UP: {n_up} ({n_up/len(y):.1%}), DOWN: {n_down} ({n_down/len(y):.1%})")
print(f"X shape: {X.shape}, y shape: {y.shape}")


# ---------------------------------------------------------------------------
# 3. CHRONOLOGICAL TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
split_idx = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dates = df.index[:split_idx]
test_dates = df.index[split_idx:]

print(f"\nTrain: {len(X_train)} samples ({train_dates[0].date()} to {train_dates[-1].date()})")
print(f"Test:  {len(X_test)} samples ({test_dates[0].date()} to {test_dates[-1].date()})")
print()


# ---------------------------------------------------------------------------
# 4. TRAIN XGBOOST CLASSIFIER
# ---------------------------------------------------------------------------
scale_pos = n_down / n_up if n_up > 0 else 1

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="logloss",
    use_label_encoder=False,
)

print("Training XGBoost Classifier...")
model.fit(X_train, y_train)
print("Training complete.\n")


# ---------------------------------------------------------------------------
# 5. EVALUATE ON TRAIN AND TEST
# ---------------------------------------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]


def evaluate_classifier(y_true, y_pred, y_proba, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"  {label} Metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-score:  {f1:.4f}")
    print(f"    MCC:       {mcc:.4f}")
    print(f"    ROC-AUC:   {auc:.4f}")
    print()
    print(f"  {label} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["DOWN", "UP"]))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc, "auc": auc}

print("=" * 55)
train_metrics = evaluate_classifier(y_train, y_train_pred, y_train_proba, "Train")
print("=" * 55)
test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_proba, "Test")
print("=" * 55)


# ---------------------------------------------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, label in zip(
    axes, [y_train, y_test], [y_train_pred, y_test_pred], ["Train", "Test"],
):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{label} Confusion Matrix ({TICKER} - XGBoost)")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_xgb_confusion_matrix.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\nTop 10 Most Important Features:")
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12}")
print("-" * 43)
for i in range(min(10, len(FEATURE_COLS))):
    idx = indices[i]
    print(f"{i+1:<6} {FEATURE_COLS[idx]:<25} {importances[idx]:.4f}")

top_n = 15
top_indices = indices[:top_n]
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[top_indices][::-1], align="center")
plt.yticks(range(top_n), [FEATURE_COLS[i] for i in top_indices][::-1])
plt.xlabel("Feature Importance (Gain)")
plt.title(f"{TICKER} — XGBoost Feature Importance (Top {top_n})")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_xgb_feature_importance.png"), dpi=150)
plt.close()


print(f"\nDone. All plots saved to: {PLOTS_DIR}")
