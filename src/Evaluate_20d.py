"""Evaluate 20-day (multi-week) direction target with walk-forward ensemble.
Saves Results/model_comparison_20d.csv and prints summary."""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
REL_DIR = os.path.join(BASE_DIR, "Data", "cleaned_rel")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
RANDOM_STATE = 42
HORIZON = 20
STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]

EXCLUDE = {"Close", "Open", "High", "Low", "Volume",
           "Close_lag1", "Close_lag7", "Close_lag21",
           "High_lag1", "High_lag7", "High_lag21",
           "Low_lag1", "Low_lag7", "Low_lag21",
           "Open_lag1", "Open_lag7", "Open_lag21",
           "returns", "log_returns", "obv", "volume_ma_5"}


def load(ticker):
    fp = os.path.join(REL_DIR, f"{ticker}_rel.csv")
    df = pd.read_csv(fp, index_col="Date", parse_dates=True)
    df["target"] = (df["Close"].shift(-HORIZON) > df["Close"]).astype(int)
    df = df.dropna()
    fc = [c for c in df.columns if c not in EXCLUDE and c != "target"]
    return df, df[fc].values, df["target"].values, fc


def evaluate(y_true, y_pred, y_proba):
    return dict(accuracy=accuracy_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred, zero_division=0),
                mcc=matthews_corrcoef(y_true, y_pred),
                roc_auc=roc_auc_score(y_true, y_proba))


def walk_forward(X, y, step=20, min_train=0.6, seeds=3):
    """Leakage-free expanding walk-forward.
    For a HORIZON-day target, training rows stop at i-HORIZON so no training
    label requires a close that falls inside the test fold; the scaler is fit
    per fold on the training slice only."""
    n = len(y)
    start = int(n * min_train)
    ps = {"lr": np.zeros(n), "rf": np.zeros(n), "xgb": np.zeros(n)}
    ps_valid = {"lr": np.zeros(n, dtype=bool), "rf": np.zeros(n, dtype=bool), "xgb": np.zeros(n, dtype=bool)}
    i = start
    while i + step <= n:
        tr_end = i - HORIZON
        sc = StandardScaler().fit(X[:tr_end])
        Xtr_z, Xte_z = sc.transform(X[:tr_end]), sc.transform(X[i:i+step])
        Xte = X[i:i+step]
        for s in range(seeds):
            lr = LogisticRegression(C=1.0, max_iter=3000, random_state=RANDOM_STATE + s)
            lr.fit(Xtr_z, y[:tr_end])
            ps["lr"][i:i+step] += lr.predict_proba(Xte_z)[:, 1] / seeds
            rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                        random_state=RANDOM_STATE + s, n_jobs=-1)
            rf.fit(X[:tr_end], y[:tr_end])
            ps["rf"][i:i+step] += rf.predict_proba(Xte)[:, 1] / seeds
            xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="logloss", use_label_encoder=False,
                                random_state=RANDOM_STATE + s, n_jobs=-1)
            xgb.fit(X[:tr_end], y[:tr_end])
            ps["xgb"][i:i+step] += xgb.predict_proba(Xte)[:, 1] / seeds
        ps_valid["lr"][i:i+step] = True
        ps_valid["rf"][i:i+step] = True
        ps_valid["xgb"][i:i+step] = True
        i += step

    valid = ps_valid["lr"]
    y = y[valid]
    out = {}
    for m, p in ps.items():
        p = p[valid]
        out[m] = evaluate(y, (p > 0.5).astype(int), p)
    mean_prob = np.mean([ps[m][valid] for m in ps], axis=0)
    out["Ensemble"] = evaluate(y, (mean_prob > 0.5).astype(int), mean_prob)
    return out, valid.sum()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for tk in STOCKS:
        df, X, y, fc = load(tk)
        res, n = walk_forward(X, y)
        base = y.mean()
        print("\n== %s (h=%d, base=%.3f, %d rows, %d feats) ==" % (tk, HORIZON, base, n, len(fc)))
        for m in ["lr", "rf", "xgb", "Ensemble"]:
            r = res[m]
            print("  %-8s acc=%.3f f1=%.3f mcc=%.3f auc=%.3f" % (m, r["accuracy"], r["f1"], r["mcc"], r["roc_auc"]))
        con = {"Stock": tk, "Horizon": HORIZON, "Test_Size": n, "Base_Rate": base}
        for m in ["lr", "rf", "xgb", "Ensemble"]:
            for k, v in res[m].items():
                con[f"{m.lower()}_{k}"] = v
        rows.append(con)

    out = pd.DataFrame(rows)
    for m in ["lr", "rf", "xgb", "Ensemble"]:
        pass
    out.to_csv(os.path.join(RESULTS_DIR, "model_comparison_20d.csv"), index=False)

    print("\n===== SUMMARY (mean across 10 stocks) =====")
    for m in ["lr", "rf", "xgb", "Ensemble"]:
        for k in ["accuracy", "f1", "mcc", "roc_auc"]:
            col = f"{m.lower()}_{k}"
            print("  %-8s %-9s mean=%.3f std=%.3f" % (m, k, out[col].mean(), out[col].std()))
    print("\nSaved:", os.path.join(RESULTS_DIR, "model_comparison_20d.csv"))


if __name__ == "__main__":
    main()