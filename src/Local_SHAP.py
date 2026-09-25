"""Local SHAP waterfall explanations.
Selects representative test predictions (a correct UP call, a correct DOWN call,
and a high-confidence error) per model per stock and saves SHAP waterfall plots
to Plots/shap_waterfall/ together with a CSV of the selected cases."""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import shap

sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject\src")
from SHAP_analysis import load_and_prepare, model_logreg, model_rf, model_xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
PLOT_STOCKS = ["AAPL", "JNJ", "TSLA"]
MODELS = {"LogisticRegression": model_logreg, "RandomForest": model_rf, "XGBoost": model_xgb}
TEST_SIZE = 0.2
OUT_DIR = os.path.join(BASE_DIR, "Plots", "shap_waterfall")
os.makedirs(OUT_DIR, exist_ok=True)


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


def pick_case(idx_arr, probs, yte, kind):
    """From candidate row indices return the single most representative one."""
    if len(idx_arr) == 0:
        return None
    if kind == "correct_up":
        return idx_arr[int(np.argmax(probs[idx_arr]))]
    if kind == "correct_down":
        return idx_arr[int(np.argmin(probs[idx_arr]))]
    if kind == "error_up":
        return idx_arr[int(np.argmax(probs[idx_arr]))]
    if kind == "error_down":
        return idx_arr[int(np.argmin(probs[idx_arr]))]
    return idx_arr[0]


def main():
    rows = []
    for tk in PLOT_STOCKS:
        data = load_and_prepare(tk)
        if data is None:
            continue
        df, X, y, fc = data
        split = int(len(df) * (1 - TEST_SIZE))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        dates = df.index[split:]

        for name, factory in MODELS.items():
            model, scaler = factory(Xtr, ytr)
            exp, base = build_explainer(model, scaler, Xtr)

            Xte_m = scaler.transform(Xte) if scaler is not None else Xte
            probs = model.predict_proba(Xte_m)[:, 1]
            pred = (probs > 0.5).astype(int)

            if isinstance(model, LogisticRegression):
                vals = exp.shap_values(Xte_m)
            else:
                vals = exp.shap_values(Xte)
            vals = np.asarray(vals)
            if vals.ndim == 3:
                vals = vals[..., -1]

            cases = [
                ("up", (pred == 1) & (yte == 1), "correct UP"),
                ("down", (pred == 0) & (yte == 0), "correct DOWN"),
                ("err", (pred == 1) & (yte == 0), "error (predicted UP)"),
            ]
            for tag, mask, label in cases:
                idx = pick_case(np.where(mask)[0], probs, yte, tag)
                if idx is None:
                    continue
                date = dates[idx].strftime("%Y-%m-%d")
                data_row = Xte_m[idx] if scaler is not None else Xte[idx]
                expl = shap.Explanation(values=vals[idx], base_values=base,
                                        data=data_row, feature_names=fc)
                fig = shap.plots.waterfall(expl, max_display=14, show=False)
                path = os.path.join(OUT_DIR, f"waterfall_{tk}_{name}_{tag}.png")
                if hasattr(fig, "figure"):
                    fig.figure.savefig(path, dpi=150, bbox_inches="tight")
                else:
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                plt_close(fig)
                rows.append({"Stock": tk, "Model": name, "Case": label,
                             "Date": date, "True": int(yte[idx]),
                             "Pred": int(pred[idx]), "Prob_UP": round(float(probs[idx]), 4),
                             "BaseRate": round(base, 4), "Plot": os.path.basename(path)})
                print("  %s %s %-18s %s prob=%.3f -> %s" % (tk, name, label, date, probs[idx], os.path.basename(path)))

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "local_shap_cases.csv"), index=False)
    print("\nSaved case table -> Plots/shap_waterfall/local_shap_cases.csv")


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close()


if __name__ == "__main__":
    main()