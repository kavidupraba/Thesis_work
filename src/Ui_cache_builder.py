"""Ui_cache_builder.py
Prepares a fast, pickle-free cache for the Flask explanation UI on the
10-stock pilot. For each stock x model it retrains the model on the current
(2018-2026) data with the chronological 80/20 split, and stores everything the
UI needs so that request time needs no model or explainer:

  Results/ui_cache/{TICKER}_{MODEL}.npz
      shap_vals   (n_test, n_feat)  SHAP values on the test set
      data        (n_test, n_feat)  feature values fed to SHAP (scaled for LR)
      probs       (n_test,)          predicted P(up)
      preds       (n_test,)          hard predictions
      true        (n_test,)          true directions
      dates       (n_test,)          test dates as strings
      base        ()                 explainer expected value
  Results/ui_cache/meta.json
      feature lists + base rate per model per stock
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import shap

sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject\src")
from SHAP_analysis import load_and_prepare, model_logreg, model_rf, model_xgb
from sklearn.linear_model import LogisticRegression

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
CACHE_DIR = os.path.join(BASE_DIR, "Results", "ui_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]
MODELS = {"LogisticRegression": model_logreg, "RandomForest": model_rf, "XGBoost": model_xgb}
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
    meta = {"stocks": {}, "models": list(MODELS)}
    for tk in STOCKS:
        data = load_and_prepare(tk)
        if data is None:
            print(f"  skip {tk}")
            continue
        df, X, y, fc = data
        split = int(len(df) * (1 - TEST_SIZE))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        dates = [d.strftime("%Y-%m-%d") for d in df.index[split:]]
        meta["stocks"][tk] = {"features": fc, "n_train": int(split), "n_test": int(len(Xte))}
        for name, factory in MODELS.items():
            model, scaler = factory(Xtr, ytr)
            exp, base = build_explainer(model, scaler, Xtr)
            Xte_m = scaler.transform(Xte) if scaler is not None else Xte
            probs = model.predict_proba(Xte_m)[:, 1]
            preds = (probs > 0.5).astype(int)
            if isinstance(model, LogisticRegression):
                v = exp.shap_values(Xte_m)
            else:
                v = exp.shap_values(Xte)
            v = np.asarray(v)
            if v.ndim == 3:
                v = v[..., -1]
            np.savez_compressed(
                os.path.join(CACHE_DIR, f"{tk}_{name}.npz"),
                shap_vals=v.astype(np.float32),
                data=(Xte_m if scaler is not None else Xte).astype(np.float32),
                probs=np.asarray(probs, dtype=np.float32),
                preds=np.asarray(preds, dtype=np.int8),
                true=np.asarray(yte, dtype=np.int8),
                dates=np.asarray(dates, dtype=object),
                base=np.asarray(base, dtype=np.float32),
            )
            meta["stocks"][tk][name] = {"base": round(float(base), 4)}
            print(f"  {tk} {name}  n_test={len(Xte)}  base={base:.4f}", flush=True)

    with open(os.path.join(CACHE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nWrote 30 npz + meta.json -> {CACHE_DIR}")


if __name__ == "__main__":
    main()