"""
RandomForest_model.py
=====================
Purpose:  Train a Random Forest Regressor to predict the next-day closing price
          of a stock using the engineered features from Preprocess_data.py.
How it works:
  - Loads cleaned CSV from ../Data/cleaned/{TICKER}_cleaned.csv
  - Uses all feature columns (lags, returns, MAs, volatility, etc.) to predict
    the next-day Close price (Close_t+1).
  - Splits data chronologically (time series — no random shuffle) into train/test.
  - Trains a RandomForestRegressor (ensemble of decision trees).
  - Evaluates predictions with regression metrics and plots feature importance.
Output:   Console prints of train/test metrics, a feature importance chart, and
          actual-vs-predicted scatter plot.
"""

import os
import glob
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")       # Non-interactive backend — no display window needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# 0. CONFIGURATION — change these as needed
# ---------------------------------------------------------------------------
TICKER = "AAPL"              # Stock to model
CLEAN_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data\cleaned"
PLOTS_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Plots"
TEST_SIZE = 0.2              # Reserve the last 20% of rows for testing
N_ESTIMATORS = 200           # Number of trees in the forest
MAX_DEPTH = 10               # Maximum tree depth (None = unlimited)
RANDOM_STATE = 42            # Seed for reproducibility

os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------------------------
# The cleaned CSV has a column for Date as index and ~29 feature columns.
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
# 2. DEFINE FEATURES (X) AND TARGET (y)
# ---------------------------------------------------------------------------
# Target: predict TOMORROW's closing price. We shift Close backward by 1 day
# so that row t has the target value = Close at t+1. The last row will have
# NaN as the target (no tomorrow to predict) so we drop it.
TARGET_COL = "Close"
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

df["target"] = df[TARGET_COL].shift(-1)   # Close_t+1 becomes the target for row t
df = df.dropna(subset=["target"])          # Drop the last row that has no target

X = df[FEATURE_COLS].values   # Shape: (n_samples, n_features)
y = df["target"].values       # Shape: (n_samples,)

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: next-day Close price")
print(f"X shape: {X.shape}, y shape: {y.shape}")


# ---------------------------------------------------------------------------
# 3. CHRONOLOGICAL TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
# Time series data MUST NOT be shuffled randomly — that would leak future
# information into the training set. We keep the order and take the last
# TEST_SIZE fraction of rows as the test set.
split_idx = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dates = df.index[:split_idx]
test_dates = df.index[split_idx:]

print(f"\nTrain: {len(X_train)} samples ({train_dates[0].date()} to {train_dates[-1].date()})")
print(f"Test:  {len(X_test)} samples ({test_dates[0].date()} to {test_dates[-1].date()})")
print()


# ---------------------------------------------------------------------------
# 4. TRAIN RANDOM FOREST REGRESSOR
# ---------------------------------------------------------------------------
# Random Forest is an ensemble of decision trees. Each tree sees a random
# subset of rows and columns, and the final prediction is the average across
# all trees. This reduces overfitting compared to a single tree.
#
# Hyperparameters tuned:
#   - n_estimators: how many trees (more = better but slower)
#   - max_depth: how deep each tree can grow (prevents overfitting)
#   - min_samples_leaf: minimum rows per leaf (also prevents overfitting)
model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,              # Use all CPU cores
)

print("Training Random Forest Regressor...")
model.fit(X_train, y_train)
print("Training complete.")
print()


# ---------------------------------------------------------------------------
# 5. EVALUATE ON TRAIN AND TEST
# ---------------------------------------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def evaluate(y_true, y_pred, label):
    """Print regression metrics for a given set of predictions."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"  {label} Metrics:")
    print(f"    RMSE:  ${rmse:.4f}")
    print(f"    MAE:   ${mae:.4f}")
    print(f"    MAPE:  {mape:.2f}%")
    print(f"    R²:    {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

print("=" * 50)
train_metrics = evaluate(y_train, y_train_pred, "Train")
print()
test_metrics = evaluate(y_test, y_test_pred, "Test")
print("=" * 50)
print()


# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (WHAT MATTERS MOST TO THE MODEL)
# ---------------------------------------------------------------------------
# Random Forest natively computes how much each column reduces impurity
# across all splits/trees. Higher = more important for prediction.
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"Top 10 Most Important Features:")
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12}")
print("-" * 43)
for i in range(min(10, len(FEATURE_COLS))):
    idx = indices[i]
    print(f"{i+1:<6} {FEATURE_COLS[idx]:<25} {importances[idx]:.4f}")

# Bar chart of top-10 feature importance
top_n = 15
top_indices = indices[:top_n]
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[top_indices][::-1], align="center")
plt.yticks(range(top_n), [FEATURE_COLS[i] for i in top_indices][::-1])
plt.xlabel("Feature Importance")
plt.title(f"{TICKER} — Random Forest Feature Importance (Top {top_n})")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_rf_feature_importance.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 7. ACTUAL vs PREDICTED SCATTER PLOT
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_test_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1, label="Perfect prediction")
plt.xlabel("Actual Close Price ($)")
plt.ylabel("Predicted Close Price ($)")
plt.title(f"{TICKER} — Actual vs Predicted (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_rf_actual_vs_predicted.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 8. PREDICTION OVER TIME
# ---------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_test, label="Actual", linewidth=1.5)
plt.plot(test_dates, y_test_pred, label="Predicted", linewidth=1.5, alpha=0.8)
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.title(f"{TICKER} — Test Set: Actual vs Predicted Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_rf_time_series.png"), dpi=150)
plt.close()


print("\nDone. All plots saved to:", PLOTS_DIR)
