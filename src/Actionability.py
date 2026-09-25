"""
Actionability.py
================
Last pipeline stage: map model probabilities (and top SHAP contributions) into
practitioner-readable trading signals, then test whether acting on those
signals has any out-of-sample edge.

Signal rule (probability thresholds, standard practice):
    strong_up    prob >= 0.60
    up           0.55 <= prob < 0.60
    neutral      0.45 <= prob < 0.55
    down         0.40 <= prob < 0.45
    strong_down  prob <  0.40

For every (pilot stock x model, 80/20 split, current 2018-2026 data) the script:
  1. builds the signal per test day,
  2. records the realised next-day return,
  3. evaluates, per model and signal class, average realised return and hit
     rate (fraction of days the realised move matched the sign of the signal),
  4. flags the top SHAP drivers behind each signal.

Given the validated noise floor the expectation is flat performance across
signal classes; the output quantifies exactly that.

Outputs:
  Results/actionability_signals.csv      per-day signal log
  Results/actionability_eval.csv         realised return / hit rate by signal class
  Plots/actionability_return_by_signal.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject\src")
from SHAP_analysis import load_and_prepare, model_logreg, model_rf, model_xgb, FEATURE_META
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PLOTS_DIR = os.path.join(BASE_DIR, "Plots")
SIGNALS_CSV = os.path.join(RESULTS_DIR, "actionability_signals.csv")
EVAL_CSV = os.path.join(RESULTS_DIR, "actionability_eval.csv")

MODELS = {"LogisticRegression": model_logreg, "RandomForest": model_rf, "XGBoost": model_xgb}
STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]
TEST_SIZE = 0.2


def signal_of(prob):
    if prob >= 0.60:
        return "strong_up"
    if prob >= 0.55:
        return "up"
    if prob >= 0.45:
        return "neutral"
    if prob >= 0.40:
        return "down"
    return "strong_down"


SIGN = {"strong_up": 1.0, "up": 1.0, "neutral": 0.0, "down": -1.0, "strong_down": -1.0}
ORDER = ["strong_up", "up", "neutral", "down", "strong_down"]


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
    log_rows = []
    for tk in STOCKS:
        data = load_and_prepare(tk)
        if data is None:
            continue
        df, X, y, fc = data
        split = int(len(df) * (1 - TEST_SIZE))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        dates = df.index[split:]

        # realised next-day return for test rows (row i has return on date i+1)
        returns = df["returns"].values
        realised = np.roll(returns, -1)[split:]
        realised[-1] = np.nan

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

            for i in range(len(Xte)):
                p = float(probs[i])
                sig = signal_of(p)
                r = realised[i]
                if np.isnan(r):
                    continue
                hit = bool(np.sign(r) == SIGN[sig]) if SIGN[sig] != 0 else None
                # actual SHAP drivers behind the signal
                order = np.argsort(-np.abs(vals[i]))[:3]
                top = "; ".join([str(fc[j]) for j in order])
                log_rows.append({
                    "Stock": tk, "Model": name, "Date": dates[i].strftime("%Y-%m-%d"),
                    "Prob_UP": round(p, 4), "Signal": sig,
                    "Realised_Return_pct": round(r * 100.0, 3),
                    "Hit": hit, "Top_SHAP": top,
                })

    log = pd.DataFrame(log_rows)
    log.to_csv(SIGNALS_CSV, index=False)

    # evaluation: realised return and hit rate by model and signal class
    agg = log[log["Signal"] != "neutral"].copy()
    agg["sign_ret"] = agg["Realised_Return_pct"] * agg["Signal"].map(SIGN)
    rows = []
    for name in MODELS:
        sub = agg[agg["Model"] == name]
        if len(sub) == 0:
            continue
        rows.append({
            "Model": name,
            "n_signals": len(sub),
            "n_neutral": int(((log["Model"] == name) & (log["Signal"] == "neutral")).sum()),
            "hit_rate": round(float((sub["Hit"] == True).mean()), 4),
            "mean_ret_strong_up_pct": round(float(sub[sub["Signal"] == "strong_up"]["sign_ret"].mean()), 4),
            "mean_ret_strong_down_pct": round(float(sub[sub["Signal"] == "strong_down"]["sign_ret"].mean()), 4),
            "mean_ret_all_signals_pct": round(float(sub["sign_ret"].mean()), 4),
        })
    for cls in ORDER:
        for name in MODELS:
            sub = log[(log["Model"] == name) & (log["Signal"] == cls)]
            if len(sub) == 0:
                continue
            rows.append({
                "Model": name, "SignalClass": cls, "n": int(len(sub)),
                "mean_ret_pct": round(float(sub["Realised_Return_pct"].mean()), 4),
            })
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(EVAL_CSV, index=False)

    # figure: mean realised return by signal class per model
    piv = eval_df[eval_df["SignalClass"].notna()].pivot(index="SignalClass", columns="Model", values="mean_ret_pct")
    piv = piv.reindex(ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean realised next-day return (%)")
    ax.set_title("Realised return by signal class (no actionable edge expected)")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actionability_return_by_signal.png"), dpi=150)
    plt.close(fig)

    print(f"Saved -> {SIGNALS_CSV} ({len(log)} rows)")
    print(f"Saved -> {EVAL_CSV}")
    print(log[log["Signal"] != "neutral"].groupby(["Model", "Signal"]).size().unstack(fill_value=0))
    print("\nPer-model hit rates and mean directional return (of signalled days):")
    print(eval_df[eval_df["SignalClass"].isna()].to_string(index=False))


if __name__ == "__main__":
    main()