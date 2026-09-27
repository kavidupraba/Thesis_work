"""Full-scale leakage-free walk-forward on all 28 stocks (HORIZON=1).

Refreshes the headline validation numbers on the extended 2018-2026 data:
  Results/full28_wf_metrics.csv       per-stock, per-model acc/auc/mcc
  Results/full28_wf_pooled.csv        pooled acc/auc/mcc + McNemar pairs

Protocol identical to Significance_test.py: step=20, min_train=0.6, three
seeds averaged on probabilities, per-fold StandardScaler, train rows strictly
before i-HORIZON. Checkpointed per stock (Results/full28_wf_partial.csv)."""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from scipy.stats import binomtest, chi2 as _chi2
from xgboost import XGBClassifier

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
DATA_DIR = os.path.join(BASE_DIR, "Data", "cleaned")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PARTIAL = os.path.join(RESULTS_DIR, "full28_wf_partial.csv")

RS = 42
HORIZON = 1
STEP = 20
MIN_TRAIN = 0.6
SEEDS = 3

STOCKS = sorted(
    f.replace("_cleaned.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_cleaned.csv")
)

EXCLUDE = {"Close", "Open", "High", "Low", "Volume",
           "Close_lag1", "Close_lag7", "Close_lag21",
           "High_lag1", "High_lag7", "High_lag21",
           "Low_lag1", "Low_lag7", "Low_lag21",
           "Open_lag1", "Open_lag7", "Open_lag21",
           "returns", "log_returns", "obv", "volume_ma_5"}

MODELS = {
    "LR":  lambda rs, njobs: LogisticRegression(C=1.0, max_iter=3000, random_state=rs),
    "RF":  lambda rs, njobs: RandomForestClassifier(n_estimators=300, max_depth=12,
                                                    min_samples_leaf=5, random_state=rs, n_jobs=njobs),
    "XGB": lambda rs, njobs: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           eval_metric="logloss", use_label_encoder=False,
                                           random_state=rs, n_jobs=njobs),
}


def load_daily(tk):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{tk}_cleaned.csv"),
                     index_col="Date", parse_dates=True)
    df["target"] = (df["Close"].shift(-HORIZON) > df["Close"]).astype(int)
    df = df.dropna()
    fc = [c for c in df.columns if c not in EXCLUDE and c != "target"]
    return df[fc].values, df["target"].values


def wf_predictions(X, y):
    n = len(y)
    start = int(n * MIN_TRAIN)
    prob = {m: np.full(n, np.nan) for m in MODELS}
    i = start
    while i + STEP <= n:
        tr_end = i - HORIZON
        sc = StandardScaler().fit(X[:tr_end])
        Xtr_z, Xte_z = sc.transform(X[:tr_end]), sc.transform(X[i:i + STEP])
        Xte = X[i:i + STEP]
        for s in range(SEEDS):
            for m, maker in MODELS.items():
                model = maker(RS + s, -1)
                if m == "LR":
                    model.fit(Xtr_z, y[:tr_end])
                    p = model.predict_proba(Xte_z)[:, 1]
                else:
                    model.fit(X[:tr_end], y[:tr_end])
                    p = model.predict_proba(Xte)[:, 1]
                prob[m][i:i + STEP] = np.nansum([prob[m][i:i + STEP], p / SEEDS], axis=0)
        i += STEP
    valid = ~np.isnan(prob["LR"])
    return {m: prob[m][valid] for m in MODELS}, y[valid]


def mcnemar(b, c):
    if b + c == 0:
        return 1.0, 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_chi2 = float(1 - _chi2.cdf(chi2, 1))
    p_exact = min(2.0 * binomtest(min(b, c), b + c, 0.5).pvalue, 1.0)
    return p_chi2, p_exact


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = set()
    if os.path.exists(PARTIAL):
        prev = pd.read_csv(PARTIAL)
        done = set(prev["Stock"])
        rows = prev.to_dict("records")
    else:
        rows = []

    pooled = {m: {"y": [], "p": []} for m in MODELS}
    completed_stocks = set()
    for tk in STOCKS:
        if tk in done:
            completed_stocks.add(tk)
            continue
        X, y = load_daily(tk)
        probs, yv = wf_predictions(X, y)
        for m in MODELS:
            pooled[m]["y"].append(yv)
            pooled[m]["p"].append(probs[m])
            acc = accuracy_score(yv, probs[m] > 0.5)
            auc = roc_auc_score(yv, probs[m])
            mcc = matthews_corrcoef(yv, probs[m] > 0.5)
            rows.append({"Stock": tk, "Model": m, "n": int(len(yv)),
                         "Accuracy": round(float(acc), 4),
                         "BalancedAccuracy": round(float(balanced_accuracy_score(yv, probs[m] > 0.5)), 4),
                         "BaseRate": round(float(yv.mean()), 4),
                         "AUC": round(float(auc), 4),
                         "MCC": round(float(mcc), 4)})
        pd.DataFrame(rows).to_csv(PARTIAL, index=False)
        completed_stocks.add(tk)
        print(f"[{len(completed_stocks)}/28] {tk} (n={len(yv)}) done", flush=True)

    # pooled metrics + McNemar
    pool_rows = []
    for m in MODELS:
        y = np.concatenate(pooled[m]["y"])
        p = np.concatenate(pooled[m]["p"])
        pool_rows.append({"Model": m, "n": int(len(y)),
                          "Accuracy": round(float(accuracy_score(y, p > 0.5)), 4),
                          "BalancedAccuracy": round(float(balanced_accuracy_score(y, p > 0.5)), 4),
                          "BaseRate": round(float(y.mean()), 4),
                          "AUC": round(float(roc_auc_score(y, p)), 4),
                          "MCC": round(float(matthews_corrcoef(y, p > 0.5)), 4)})
    pooled_ok = {m: pd.Series(np.concatenate(pooled[m]["y"])) == pd.Series(np.concatenate(pooled[m]["y"]))}
    y_all = np.concatenate(pooled["LR"]["y"])
    pairs = [("LR", "RF"), ("LR", "XGB"), ("RF", "XGB")]
    mc = []
    for mA, mB in pairs:
        pA = (np.concatenate(pooled[mA]["p"]) > 0.5).astype(int)
        pB = (np.concatenate(pooled[mB]["p"]) > 0.5).astype(int)
        b = int(np.sum((pA == y_all) & (pB != y_all)))
        c = int(np.sum((pA != y_all) & (pB == y_all)))
        p_chi2, p_exact = mcnemar(b, c)
        mc.append({"Pair": f"{mA} vs {mB}", "b_A_only": b, "c_B_only": c,
                   "p_chi2": p_chi2, "p_exact": p_exact})

    pd.DataFrame(pool_rows).to_csv(os.path.join(RESULTS_DIR, "full28_wf_pooled.csv"), index=False)
    pd.DataFrame(mc).to_csv(os.path.join(RESULTS_DIR, "full28_mcnemar_pooled.csv"), index=False)
    pd.DataFrame(rows).to_csv(PARTIAL, index=False)
    print(pd.DataFrame(pool_rows).to_string(index=False))
    print(pd.DataFrame(mc).to_string(index=False))
    print("Saved pooled + McNemar.")


if __name__ == "__main__":
    main()