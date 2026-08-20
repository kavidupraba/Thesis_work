"""
LogisticRegression_model.py
===========================
Purpose:  Train a Logistic Regression classifier to predict the DIRECTION of
          the next-day stock price movement (Up / Down) using the engineered
          features from Preprocess_data.py.
How it works:
  - Loads cleaned CSV from ../Data/cleaned/{TICKER}_cleaned.csv
  - Creates a binary target: 1 if tomorrow's Close > today's Close, else 0.
  - Uses all feature columns (lags, returns, MAs, volatility, etc.).
  - Splits data chronologically into train/test.
  - Scales features (Logistic Regression is sensitive to scale).
  - Trains a LogisticRegression classifier.
  - Evaluates with accuracy, precision, recall, F1, and confusion matrix.
  - Shows model coefficients (interpretable as feature importance).
Output:   Console prints of train/test classification metrics, confusion matrix,
          and coefficient bar chart.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")       # Non-interactive backend — no display window needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# ---------------------------------------------------------------------------
# 0. CONFIGURATION — change these as needed
# ---------------------------------------------------------------------------
TICKER = "AAPL"              # Stock to model
CLEAN_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Data\cleaned"
PLOTS_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject\Plots"
TEST_SIZE = 0.2              # Reserve the last 20% of rows for testing
RANDOM_STATE = 42            # Seed for reproducibility

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
# We exclude the raw Close and target-related columns from features to avoid
# data leakage (the model shouldn't see today's Close to predict direction).
# Target: 1 if Close goes UP tomorrow, 0 if DOWN or unchanged.
TARGET_COL = "Close"

# Exclude columns that would give the model an "unfair advantage":
#   - Close itself (we're predicting its movement)
#   - returns / log_returns (they encode today's change relative to yesterday,
#     which is nearly the same as the target if we're not careful)
EXCLUDE = {TARGET_COL, "returns", "log_returns"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

# Create binary target: tomorrow's Close > today's Close → 1, else 0.
df["target"] = (df[TARGET_COL].shift(-1) > df[TARGET_COL]).astype(int)
df = df.dropna(subset=["target"])   # drop last row with no tomorrow

X = df[FEATURE_COLS].values
y = df["target"].values

# Check class balance
n_up = y.sum()
n_down = len(y) - n_up
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: Next-day direction (1=UP, 0=DOWN)")
print(f"Class distribution — UP: {n_up} ({n_up/len(y):.1%}), DOWN: {n_down} ({n_down/len(y):.1%})")
print(f"X shape: {X.shape}, y shape: {y.shape}")


# ---------------------------------------------------------------------------
# 3. CHRONOLOGICAL TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
# Strictly chronological — no shuffle — to avoid time leakage.
split_idx = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dates = df.index[:split_idx]
test_dates = df.index[split_idx:]

print(f"\nTrain: {len(X_train)} samples ({train_dates[0].date()} to {train_dates[-1].date()})")
print(f"Test:  {len(X_test)} samples ({test_dates[0].date()} to {test_dates[-1].date()})")
print()


# ---------------------------------------------------------------------------
# 4. STANDARDIZE FEATURES (Z-SCORE NORMALIZATION)
# ---------------------------------------------------------------------------
# Logistic Regression uses gradient descent internally. Features with very
# different magnitudes (e.g. Volume in millions vs returns ~0.01) will cause
# the optimizer to struggle. We subtract the mean and divide by standard
# deviation so every feature has mean=0, std=1.
#
# Fit the scaler ONLY on the training set to prevent test data from leaking
# into the training process.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
# Logistic Regression models the probability that y=1 as:
#   P(y=1 | X) = 1 / (1 + exp(- (w₀ + w₁x₁ + ... + wₙxₙ)))
# where the coefficients w are learned by maximizing the log-likelihood.
#
# Hyperparameters:
#   - C = inverse of regularization strength (smaller = stronger regularization)
#   - class_weight='balanced' automatically adjusts weights inversely to class
#     frequency, helping if UP/DOWN are imbalanced.
#   - max_iter: more iterations ensure convergence.
model = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Training Logistic Regression...")
model.fit(X_train_scaled, y_train)
print("Training complete.\n")


# ---------------------------------------------------------------------------
# 6. EVALUATE ON TRAIN AND TEST
# ---------------------------------------------------------------------------
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Probabilities for the positive class (UP)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]


def evaluate_classifier(y_true, y_pred, y_proba, label):
    """Print classification metrics for a set of predictions."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"  {label} Metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}  (of predicted UP, how many were correct)")
    print(f"    Recall:    {rec:.4f}  (of actual UP days, how many predicted)")
    print(f"    F1-score:  {f1:.4f}  (harmonic mean of precision & recall)")
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
# 7. CONFUSION MATRIX
# ---------------------------------------------------------------------------
# Shows counts of: true DOWN, false UP, false DOWN, true UP.
# Diagonal = correct predictions; off-diagonal = errors.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_true, y_pred, label in zip(
    axes,
    [y_train, y_test],
    [y_train_pred, y_test_pred],
    ["Train", "Test"],
):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{label} Confusion Matrix ({TICKER})")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_lr_confusion_matrix.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 8. MODEL COEFFICIENTS (INTERPRETABLE FEATURE IMPORTANCE)
# ---------------------------------------------------------------------------
# Logistic Regression coefficients tell us the log-odds change in the target
# for a one-standard-deviation increase in each feature. A positive coefficient
# means the feature pushes the prediction toward UP; negative pushes toward DOWN.
coefs = model.coef_.flatten()
coef_df = pd.DataFrame({"Feature": FEATURE_COLS, "Coefficient": coefs})
coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)

print("\nTop 10 Most Influential Features (by |coefficient|):")
print(f"{'Rank':<6} {'Feature':<25} {'Coefficient':<12}")
print("-" * 43)
for i in range(min(10, len(coef_df))):
    row = coef_df.iloc[i]
    print(f"{i+1:<6} {row['Feature']:<25} {row['Coefficient']:+.4f}")

# Bar chart: top positive and negative coefficients
top_coefs = pd.concat([
    coef_df.head(8),
    coef_df.tail(8),
])
plt.figure(figsize=(10, 7))
colors = ["green" if c > 0 else "red" for c in top_coefs["Coefficient"]]
plt.barh(range(len(top_coefs)), top_coefs["Coefficient"], color=colors, alpha=0.7)
plt.yticks(range(len(top_coefs)), top_coefs["Feature"])
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Coefficient (log-odds)")
plt.title(f"{TICKER} — Logistic Regression Coefficients\n(Positive = pushes prediction UP, Negative = DOWN)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_lr_coefficients.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 9. PREDICTION ACCURACY OVER TIME
# ---------------------------------------------------------------------------
# Visualize where the model gets it right vs wrong over the test period.
results = pd.DataFrame({
    "Date": test_dates,
    "Actual": y_test,
    "Predicted": y_test_pred,
    "Correct": y_test == y_test_pred,
}, index=test_dates)

plt.figure(figsize=(14, 4))
colors = ["green" if c else "red" for c in results["Correct"]]
plt.scatter(results["Date"], results["Actual"], c=colors, s=15, alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Actual Direction (1=UP, 0=DOWN)")
plt.title(f"{TICKER} — Test Set: Correct (green) vs Incorrect (red) Predictions")
plt.yticks([0, 1], ["DOWN", "UP"])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{TICKER}_lr_prediction_timeline.png"), dpi=150)
plt.close()


print(f"\nDone. All plots saved to: {PLOTS_DIR}")
