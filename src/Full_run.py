"""Full-scale leakage-free walk-forward on all 28 stocks (HORIZON=1).

Refreshes the headline validation numbers on the extended 2018-2026 data:
  Results/full28_wf_metrics_{TAG}.csv       per-stock, per-model acc/auc/mcc
  Results/full28_wf_pooled_{TAG}.csv        pooled acc/auc/mcc + McNemar pairs
  Results/full28_wf_predictions_{TAG}.csv   per stock-date OOS probabilities
                                             (+ p_prior: expanding-window train
                                             up-share, fold: fold index)

Protocol identical to Significance_test.py: step=20, min_train=0.6, three
seeds averaged on probabilities, per-fold StandardScaler, train rows strictly
before i-HORIZON. Checkpointed per stock from the predictions file.

Usage (baseline vs market):
    $env:THESIS_DATA_DIR="...\\Data\\cleaned";     $env:THESIS_RUN_TAG="baseline"
    .venv\\Scripts\\python.exe src\\Full_run.py

    $env:THESIS_DATA_DIR="...\\Data\\cleaned_mkt"; $env:THESIS_RUN_TAG="market"
    .venv\\Scripts\\python.exe src\\Full_run.py

Or pass CLI flags:
    python src/Full_run.py --data-dir Data/cleaned_mkt --tag market

Ablations (integration plan §7.4):
    --feature-set technical          # baseline columns only
    --feature-set market             # market columns only
    --market-lag 1                   # robustness: shift market block +1 day

Every tag writes a ``full28_wf_config_{tag}.json`` sidecar; resuming with a
different configuration under the same tag raises instead of mixing runs.
"""

import argparse
import json
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

from Market_features import MARKET_FEATURE_COLS

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
DATA_DIR = os.environ.get("THESIS_DATA_DIR", os.path.join(BASE_DIR, "Data", "cleaned"))
TAG = os.environ.get("THESIS_RUN_TAG", "baseline")
# all | technical | market    (ablation C in the integration plan)
FEATURE_SET = os.environ.get("THESIS_FEATURE_SET", "all")
# 0 = market features observed at close t (primary), 1 = shifted one day (robustness D)
MARKET_LAG = int(os.environ.get("THESIS_MARKET_LAG", "0"))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

RS = 42
HORIZON = 1
STEP = 20
MIN_TRAIN = 0.6
SEEDS = 3

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

MODEL_COLS = {"LR": "p_LR", "RF": "p_RF", "XGB": "p_XGB"}


def load_daily(tk, data_dir, feature_set="all", market_lag=0):
    df = pd.read_csv(os.path.join(data_dir, f"{tk}_cleaned.csv"),
                     index_col="Date", parse_dates=True)

    if market_lag:
        mkt_present = [c for c in MARKET_FEATURE_COLS if c in df.columns]
        df[mkt_present] = df[mkt_present].shift(market_lag)

    df["target"] = (df["Close"].shift(-HORIZON) > df["Close"]).astype(int)
    df = df.dropna()
    fc = [c for c in df.columns if c not in EXCLUDE and c != "target"]

    mkt = set(MARKET_FEATURE_COLS)
    if feature_set == "technical":
        fc = [c for c in fc if c not in mkt]
    elif feature_set == "market":
        fc = [c for c in fc if c in mkt]
    elif feature_set != "all":
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if not fc:
        raise ValueError(f"{tk}: no features selected for feature_set={feature_set}")

    return df, df[fc].values, df["target"].values


def wf_predictions(X, y):
    n = len(y)
    start = int(n * MIN_TRAIN)
    prob = {m: np.full(n, np.nan) for m in MODELS}
    # Expanding-window up-share for the fold that predicts each row. It uses only
    # y[:tr_end], i.e. nothing at or after the test window, so it is a strictly
    # out-of-sample base-rate forecast and the reference for Brier/log-loss skill.
    prior = np.full(n, np.nan)
    fold = np.full(n, -1, dtype=int)
    i = start
    k = 0
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
        prior[i:i + STEP] = float(np.mean(y[:tr_end]))
        fold[i:i + STEP] = k
        i += STEP
        k += 1
    valid = ~np.isnan(prob["LR"])
    return ({m: prob[m][valid] for m in MODELS}, y[valid], valid,
            prior[valid], fold[valid])


def mcnemar(b, c):
    if b + c == 0:
        return 1.0, 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_chi2 = float(1 - _chi2.cdf(chi2, 1))
    p_exact = min(2.0 * binomtest(min(b, c), b + c, 0.5).pvalue, 1.0)
    return p_chi2, p_exact


def metrics(y, p):
    pred = p > 0.5
    return {
        "Accuracy": round(float(accuracy_score(y, pred)), 4),
        "BalancedAccuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "BaseRate": round(float(np.mean(y)), 4),
        "AUC": round(float(roc_auc_score(y, p)), 4),
        "MCC": round(float(matthews_corrcoef(y, pred)), 4),
    }


def main(data_dir=None, tag=None, feature_set=None, market_lag=None):
    data_dir = data_dir or DATA_DIR
    tag = tag or TAG
    feature_set = feature_set or FEATURE_SET
    market_lag = MARKET_LAG if market_lag is None else market_lag
    os.makedirs(RESULTS_DIR, exist_ok=True)

    partial = os.path.join(RESULTS_DIR, f"full28_wf_partial_{tag}.csv")
    pred_partial = os.path.join(RESULTS_DIR, f"full28_wf_predictions_{tag}.csv")

    # Guard the resume path: reusing a predictions file produced under a
    # different feature configuration would silently mix runs.
    config_path = os.path.join(RESULTS_DIR, f"full28_wf_config_{tag}.json")
    config = {"data_dir": os.path.abspath(data_dir), "tag": tag,
              "feature_set": feature_set, "market_lag": market_lag,
              "horizon": HORIZON, "step": STEP, "min_train": MIN_TRAIN,
              "seeds": SEEDS, "random_state": RS}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old != config:
            raise RuntimeError(
                f"Tag '{tag}' already has a different configuration:\n"
                f"  existing: {old}\n  requested: {config}\n"
                "Use a new --tag or delete the existing "
                f"full28_wf_*_{tag}.csv files."
            )
    else:
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

    stocks = sorted(
        f.replace("_cleaned.csv", "") for f in os.listdir(data_dir)
        if f.endswith("_cleaned.csv")
    )

    print("=" * 60)
    print(f"DATA_DIR : {data_dir}")
    print(f"TAG      : {tag}")
    print(f"FEATURES : {feature_set} (market_lag={market_lag})")
    print(f"STOCKS   : {len(stocks)}")
    print("=" * 60)

    if os.path.exists(pred_partial):
        pred_df = pd.read_csv(pred_partial)
        pred_df["Date"] = pd.to_datetime(pred_df["Date"], errors="coerce")
        if "p_prior" not in pred_df.columns or "fold" not in pred_df.columns:
            raise RuntimeError(
                f"{pred_partial} was written before p_prior/fold existed. "
                "Back it up and delete it to force a clean rerun."
            )
        done = set(pred_df["Stock"].astype(str))
        print(f"Resuming from {pred_partial} ({len(done)} stocks done)\n")
    else:
        pred_df = pd.DataFrame()
        done = set()

    completed = set(done)
    for tk in stocks:
        if tk in done:
            continue
        df, X, y = load_daily(tk, data_dir, feature_set=feature_set,
                              market_lag=market_lag)
        probs, yv, valid, prior, fold = wf_predictions(X, y)
        dates = df.index[valid]

        block = pd.DataFrame({
            "Stock": tk,
            "Date": dates,
            "y": yv.astype(int),
            "p_LR": probs["LR"],
            "p_RF": probs["RF"],
            "p_XGB": probs["XGB"],
            "p_prior": prior,
            "fold": fold.astype(int),
        })
        pred_df = pd.concat([pred_df, block], ignore_index=True)
        pred_df.to_csv(pred_partial, index=False)

        completed.add(tk)
        print(f"[{len(completed)}/{len(stocks)}] {tk} (n={len(yv)}) done", flush=True)

    # Per-stock metrics
    rows = []
    for tk, g in pred_df.groupby("Stock", sort=False):
        yv = g["y"].values
        for m, col in MODEL_COLS.items():
            rows.append({"Stock": tk, "Model": m, "n": int(len(yv)),
                         **metrics(yv, g[col].values)})
    pd.DataFrame(rows).to_csv(partial, index=False)

    # Pooled metrics + McNemar
    y_all = pred_df["y"].values
    pool_rows = []
    for m, col in MODEL_COLS.items():
        p = pred_df[col].values
        pool_rows.append({"Model": m, "n": int(len(y_all)), **metrics(y_all, p)})

    pairs = [("LR", "RF"), ("LR", "XGB"), ("RF", "XGB")]
    mc = []
    for mA, mB in pairs:
        pA = (pred_df[MODEL_COLS[mA]].values > 0.5).astype(int)
        pB = (pred_df[MODEL_COLS[mB]].values > 0.5).astype(int)
        b = int(np.sum((pA == y_all) & (pB != y_all)))
        c = int(np.sum((pA != y_all) & (pB == y_all)))
        p_chi2, p_exact = mcnemar(b, c)
        mc.append({"Pair": f"{mA} vs {mB}", "b_A_only": b, "c_B_only": c,
                   "p_chi2": p_chi2, "p_exact": p_exact})

    pd.DataFrame(pool_rows).to_csv(os.path.join(RESULTS_DIR, f"full28_wf_pooled_{tag}.csv"), index=False)
    pd.DataFrame(mc).to_csv(os.path.join(RESULTS_DIR, f"full28_mcnemar_pooled_{tag}.csv"), index=False)
    pd.DataFrame(rows).to_csv(partial, index=False)

    print("\n" + pd.DataFrame(pool_rows).to_string(index=False))
    print("\n" + pd.DataFrame(mc).to_string(index=False))
    print(f"\nSaved: {partial}")
    print(f"Saved: {pred_partial}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="directory containing {TICKER}_cleaned.csv")
    parser.add_argument("--tag", default=TAG, help="suffix for result files")
    parser.add_argument("--feature-set", default=FEATURE_SET,
                        choices=["all", "technical", "market"],
                        help="which feature block to train on (ablation C)")
    parser.add_argument("--market-lag", type=int, default=MARKET_LAG,
                        help="shift market features by N days (0=primary, 1=robustness D)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(data_dir=args.data_dir, tag=args.tag,
         feature_set=args.feature_set, market_lag=args.market_lag)
