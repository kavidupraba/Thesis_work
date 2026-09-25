"""McNemar significance testing between LR, RF and XGBoost on leakage-free
daily-direction walk-forward predictions (10-stock pilot). Saves
Results/mcnemar_significance.csv."""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, binomtest
from xgboost import XGBClassifier

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
DATA_DIR = os.path.join(BASE_DIR, "Data", "cleaned")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
RS = 42
HORIZON = 1
STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]

EXCLUDE = {"Close", "Open", "High", "Low", "Volume",
           "Close_lag1", "Close_lag7", "Close_lag21",
           "High_lag1", "High_lag7", "High_lag21",
           "Low_lag1", "Low_lag7", "Low_lag21",
           "Open_lag1", "Open_lag7", "Open_lag21",
           "returns", "log_returns", "obv", "volume_ma_5"}


def load_daily(tk):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{tk}_cleaned.csv"),
                     index_col="Date", parse_dates=True)
    df["target"] = (df["Close"].shift(-HORIZON) > df["Close"]).astype(int)
    df = df.dropna()
    fc = [c for c in df.columns if c not in EXCLUDE and c != "target"]
    return df[fc].values, df["target"].values, df.index.values


def wf_predictions(X, y, step=20, min_train=0.6, seeds=3):
    """Leakage-free daily walk-forward; returns per-model mean-probability
    array (NaN where no fold covered the row) and per-model hard predictions."""
    n = len(y)
    start = int(n * min_train)
    prob = {"lr": np.full(n, np.nan), "rf": np.full(n, np.nan), "xgb": np.full(n, np.nan)}
    i = start
    while i + step <= n:
        tr_end = i - HORIZON  # labels known strictly before the test fold
        sc = StandardScaler().fit(X[:tr_end])
        Xtr_z, Xte_z = sc.transform(X[:tr_end]), sc.transform(X[i:i+step])
        Xte = X[i:i+step]
        for s in range(seeds):
            lr = LogisticRegression(C=1.0, max_iter=3000, random_state=RS + s)
            lr.fit(Xtr_z, y[:tr_end])
            prob["lr"][i:i+step] = np.nansum([prob["lr"][i:i+step], lr.predict_proba(Xte_z)[:, 1] / seeds], axis=0)
            rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                        random_state=RS + s, n_jobs=-1)
            rf.fit(X[:tr_end], y[:tr_end])
            prob["rf"][i:i+step] = np.nansum([prob["rf"][i:i+step], rf.predict_proba(Xte)[:, 1] / seeds], axis=0)
            xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                use_label_encoder=False, random_state=RS + s, n_jobs=-1)
            xgb.fit(X[:tr_end], y[:tr_end])
            prob["xgb"][i:i+step] = np.nansum([prob["xgb"][i:i+step], xgb.predict_proba(Xte)[:, 1] / seeds], axis=0)
        i += step
    valid = ~np.isnan(prob["lr"])
    preds = {m: (prob[m][valid] > 0.5).astype(int) for m in prob}
    return preds, y[valid]


def mcnemar(yA, yB, y):
    """McNemar contingency on two models' hard predictions."""
    n = len(y)
    okA = (yA == y)
    okB = (yB == y)
    both = np.sum(okA & okB)
    A_only = np.sum(okA & ~okB)   # b: A correct, B wrong
    B_only = np.sum(~okA & okB)   # c: B correct, A wrong
    neither = n - both - A_only - B_only
    table = np.array([[both, A_only], [B_only, neither]])
    b, c = A_only, B_only
    if b + c == 0:
        chi2, p_chi2, exact_p = np.nan, 1.0, 1.0
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
        # chi2 ~ chi-square(1), two-sided p
        p_chi2 = 1 - chi2_contingency(table, correction=True)[1]  # placeholder replaced below
        p_chi2 = 1.0  # replaced
        from scipy.stats import chi2 as _chi2
        p_chi2 = float(1 - _chi2.cdf(chi2, 1))
        k = min(b, c)
        exact_p = 2.0 * binomtest(k, b + c, 0.5).pvalue
        exact_p = min(exact_p, 1.0)
    return dict(pair_b_c=(b, c), n=n, chi2=float(chi2), p_chi2=float(p_chi2), p_exact=float(exact_p))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pairs = [("lr", "rf"), ("lr", "xgb"), ("rf", "xgb")]
    LABEL = {"lr": "LR", "rf": "RF", "xgb": "XGB"}
    rows = []
    pooled = {p: [0, 0] for p in pairs}  # (b, c) summed
    for tk in STOCKS:
        X, y, idx = load_daily(tk)
        preds, yv = wf_predictions(X, y)
        print("\n== %s (n=%d) ==" % (tk, len(yv)))
        for (mA, mB) in pairs:
            r = mcnemar(preds[mA], preds[mB], yv)
            pooled[(mA, mB)][0] += r["pair_b_c"][0]
            pooled[(mA, mB)][1] += r["pair_b_c"][1]
            print("  %s vs %s: b(correct-A-only)=%d c(correct-B-only)=%d chi2=%.3f p(chi2)=%.4f p(exact)=%.4f"
                  % (LABEL[mA], LABEL[mB], r["pair_b_c"][0], r["pair_b_c"][1], r["chi2"], r["p_chi2"], r["p_exact"]))
            rows.append({"Stock": tk, "Model_A": LABEL[mA], "Model_B": LABEL[mB],
                         "b_A_only_correct": r["pair_b_c"][0], "c_B_only_correct": r["pair_b_c"][1],
                         "n": r["n"], "chi2": r["chi2"], "p_chi2": r["p_chi2"], "p_exact": r["p_exact"]})

    print("\n===== POOLED across 10 stocks =====")
    for (mA, mB), (b, c) in pooled.items():
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c) if b + c else np.nan
        from scipy.stats import chi2 as _chi2
        p = float(1 - _chi2.cdf(chi2, 1)) if b + c else 1.0
        k = min(b, c)
        pex = min(2.0 * binomtest(k, b + c, 0.5).pvalue, 1.0) if b + c else 1.0
        print("  %s vs %s: b=%d c=%d chi2=%.3f p(chi2)=%.4f p(exact)=%.4f" % (LABEL[mA], LABEL[mB], b, c, chi2, p, pex))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "mcnemar_significance.csv"), index=False)
    print("\nSaved:", os.path.join(RESULTS_DIR, "mcnemar_significance.csv"))


if __name__ == "__main__":
    main()