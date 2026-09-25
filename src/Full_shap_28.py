"""Global SHAP feature importance across all 28 stocks (80/20 chronological
split, same pipeline as SHAP_analysis.py). Outputs:
  Results/shap/feature_importance_28_by_stock.csv
  Results/shap/feature_importance_28_summary.csv   (pooled mean |SHAP|)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import shap

sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject\src")
from SHAP_analysis import load_and_prepare, model_logreg, model_rf, model_xgb
from sklearn.linear_model import LogisticRegression

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
DATA_DIR = os.path.join(BASE_DIR, "Data", "cleaned")
SHAP_DIR = os.path.join(BASE_DIR, "Results", "shap")
os.makedirs(SHAP_DIR, exist_ok=True)

STOCKS = sorted(f.replace("_cleaned.csv", "")
                for f in os.listdir(DATA_DIR) if f.endswith("_cleaned.csv"))
MODELS = {"LogisticRegression": model_logreg, "RandomForest": model_rf, "XGBoost": model_xgb}
TEST_SIZE = 0.2


def build_explainer(model, scaler, X_train):
    if isinstance(model, LogisticRegression):
        Xs = scaler.transform(X_train)
        masker = shap.maskers.Independent(Xs, max_samples=Xs.shape[0])
        exp = shap.LinearExplainer(model, masker, feature_perturbation="interventional")
        return exp
    return shap.TreeExplainer(model)


def main():
    rows = []
    done = set()
    partial = os.path.join(SHAP_DIR, "feature_importance_28_partial.csv")
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        done = set(prev["Stock"])
        rows = prev.to_dict("records")

    for tk in STOCKS:
        if tk in done:
            continue
        data = load_and_prepare(tk)
        if data is None:
            continue
        df, X, y, fc = data
        split = int(len(df) * (1 - TEST_SIZE))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        for name, factory in MODELS.items():
            model, scaler = factory(Xtr, ytr)
            exp = build_explainer(model, scaler, Xtr)
            Xte_m = scaler.transform(Xte) if scaler is not None else Xte
            if isinstance(model, LogisticRegression):
                v = exp.shap_values(Xte_m)
            else:
                v = exp.shap_values(Xte)
            v = np.asarray(v)
            if v.ndim == 3:
                v = v[..., -1]
            mabs = np.mean(np.abs(v), axis=0)
            for j, fname in enumerate(fc):
                rows.append({"Stock": tk, "Model": name, "Feature": fname,
                             "MeanAbsSHAP": round(float(mabs[j]), 6)})
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(f"[{len(done) + 1}/{len(STOCKS)}] {tk} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SHAP_DIR, "feature_importance_28_by_stock.csv"), index=False)
    summary = df.groupby(["Model", "Feature"])["MeanAbsSHAP"].mean().reset_index()
    summary.to_csv(os.path.join(SHAP_DIR, "feature_importance_28_summary.csv"), index=False)

    # top-10 pooled per model
    for m in MODELS:
        top = summary[summary["Model"] == m].nlargest(10, "MeanAbsSHAP")
        print(f"\n{m} pooled top-10:")
        print(top.to_string(index=False))
    if os.path.exists(partial):
        os.remove(partial)
    print("\nSaved: feature_importance_28_by_stock.csv, feature_importance_28_summary.csv")


if __name__ == "__main__":
    main()