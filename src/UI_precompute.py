"""UI_precompute.py
==================
Precomputes, on the current 2018-2026 cleaned data (chronological 80/20
split), the per-model test-window arrays the Flask UI needs: dates, true
labels, up-probabilities, explainer base value, SHAP values, raw feature
rows, and feature names. Saves one npz per (stock, model) under
Results/ui_cache/ plus a manifest.json describing available dates.

Serves the same pilot stocks and models as the rest of the pipeline.
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject\src")
from SHAP_analysis import load_and_prepare, model_logreg, model_rf, model_xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
CACHE_DIR = os.path.join(BASE_DIR, "Results", "ui_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODELS = {"LogisticRegression": model_logreg, "RandomForest": model_rf, "XGBoost": model_xgb}
STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]
TEST_SIZE = 0.2


def build_explainer(model, scaler, X_train):
    if isinstance(model, LogisticRegression):
        Xs = scaler.transform(X_train)
        masker = shap.maskers.Independent(Xs, max_samples=Xs.shape[0])
        exp = shap.LinearExplainer(model, masker, feature_perturbation="interventional")
        base = np.asarray(exp.expected_value)
        if base.ndim > 0:
            base = base[-1]
        return exp, float(base)
    exp = shap.TreeExplainer(model)
    base = np.asarray(exp.expected_value)
    if base.ndim > 0:
        base = base[-1]
    return exp, float(base)


def main():
    manifest = {"stocks": STOCKS, "models": list(MODELS), "window": {}}
    for tk in STOCKS:
        data = load_and_prepare(tk)
        if data is None:
            continue
        df, X, y, fc = data
        split = int(len(df) * (1 - TEST_SIZE))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        dates = [d.strftime("%Y-%m-%d") for d in df.index[split:]]
        manifest["window"][tk] = {"dates": dates}
        for name, factory in MODELS.items():
            model, scaler = factory(Xtr, ytr)
            exp, base = build_explainer(model, scaler, Xtr)
            Xte_m = scaler.transform(Xte) if scaler is not None else Xte
            probs = model.predict_proba(Xte_m)[:, 1]
            if isinstance(model, LogisticRegression):
                vals = exp.shap_values(Xte_m)
            else:
                vals = exp.shap_values(Xte)
            vals = np.asarray(vals)
            if vals.ndim == 3:
                vals = vals[..., -1]
            path = os.path.join(CACHE_DIR, f"{tk}_{name}.npz")
            np.savez_compressed(path, dates=np.array(dates, dtype=object),
                                y=yte.astype(int), prob=probs.astype(float),
                                base=np.float64(base), shap=vals.astype(float),
                                Xraw=Xte.astype(float), feats=np.array(fc, dtype=object))
        print(f"{tk}: {len(dates)} test days", flush=True)

    with open(os.path.join(CACHE_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print("Saved ui_cache for", len(STOCKS), "stocks x", len(MODELS), "models")


if __name__ == "__main__":
    main()