"""Cross-sectional experiment: relative features + Rank-IC evaluation.

Two-part experiment on the 28-stock universe (2018-2026):

A) Cross-sectional relative features.  For each of the 23 model features and
   each trading date, we add the cross-sectional percentile rank and z-score of
   that feature across the stocks present on that date:
        {feature}__rank   (0..1)          {feature}__z
   The models then see 69 features (23 original + 46 relative).  Ranks/z-scores
   are point-in-time: they use only values observable on the trading date
   itself, so a test prediction for stock S on day d is informed only by other
   stocks' day-d indicator values, never by future data.  This satisfies the
   same leakage-free contract as Full_run.py.

C) Evaluation upgrade, per the Vibe-Trading factor methodology (cross-sectional
   Spearman IC / ICIR) plus class-balance reporting:
     - Pooled accuracy, balanced accuracy, AUC, MCC  (balanced accuracy + base
       rate reported so a majority class cannot masquerade as skill)
     - Rank IC: per date, Spearman correlation between the model's predicted
       P(UP) rank and next-day return rank across stocks; then mean IC, IC std,
       ICIR, % positive-IC days, and significance (t = ICIR*sqrt(n)).
     - Random-shuffle control: within each date, shuffle P(UP) across stocks
       (5 seeds), recompute IC, and test real IC minus shuffle IC against zero.

Leakage-free walk-forward protocol identical to Significance_test.py/Full_run.py:
step=20, min_train=0.6, three seeds averaged on probabilities, per-fold
StandardScaler (LR only, as in Full_run.py), train rows strictly before i-HORIZON.

Outputs (Results/cross_sectional/):
  cs_partial.csv       checkpoint (per stock)
  cs_per_stock.csv     per stock/model: n, accuracy, balanced accuracy, base rate
  cs_pooled.csv        pooled metrics
  cs_rank_ic.csv       Rank-IC/ICIR + random-control per model

Run:  .venv\\Scripts\\python.exe src\\Cross_sectional_experiment.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import spearmanr, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef,
                             balanced_accuracy_score)
from xgboost import XGBClassifier

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
DATA_DIR = os.path.join(BASE_DIR, "Data", "cleaned")
CS_DIR = os.path.join(BASE_DIR, "Data", "cross_sectional")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "cross_sectional")
PARTIAL = os.path.join(RESULTS_DIR, "cs_partial.csv")

RS = 42
HORIZON = 1
STEP = 20
MIN_TRAIN = 0.6
SEEDS = 3
N_RAND_SEEDS = 5
MIN_STOCKS_PER_DATE = 5

STOCKS = sorted(
    f.replace("_cleaned.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_cleaned.csv")
)

EXCLUDE = {"Close", "Open", "High", "Low", "Volume",
           "Close_lag1", "Close_lag7", "Close_lag21",
           "High_lag1", "High_lag7", "High_lag21",
           "Low_lag1", "Low_lag7", "Low_lag21",
           "Open_lag1", "Open_lag7", "Open_lag21",
           "returns", "log_returns", "obv", "volume_ma_5"}

BASE_FEATURES = [
    "sma_10", "sma_50", "ema_12", "ema_26", "rsi_14", "macd",
    "macd_signal", "macd_histogram", "roc_10", "bb_upper", "bb_lower",
    "bb_width", "atr_14", "volume_roc", "returns_lag1", "returns_lag7",
    "ma_5", "ma_20", "ma_50", "volatility_5", "volatility_20",
    "high_low_range", "close_open_change",
]

MODELS = {
    "LR":  lambda rs, njobs: LogisticRegression(C=1.0, max_iter=3000, random_state=rs),
    "RF":  lambda rs, njobs: RandomForestClassifier(n_estimators=300, max_depth=12,
                                                    min_samples_leaf=5, random_state=rs, n_jobs=njobs),
    "XGB": lambda rs, njobs: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           eval_metric="logloss", use_label_encoder=False,
                                           random_state=rs, n_jobs=njobs),
}


def build_cross_sectional_features():
    """Add {feature}__rank and {feature}__z columns for each base feature."""
    os.makedirs(CS_DIR, exist_ok=True)
    sheets = {tk: pd.read_csv(os.path.join(DATA_DIR, f"{tk}_cleaned.csv"),
                              index_col="Date", parse_dates=True) for tk in STOCKS}
    missing = [feat for feat in BASE_FEATURES if feat not in sheets[STOCKS[0]].columns]
    if missing:
        raise ValueError(f"Features not found: {missing}")
    idx = sorted(set().union(*[s.index for s in sheets.values()]))
    for feat in BASE_FEATURES:
        panel = pd.DataFrame(index=idx, columns=STOCKS)
        for tk in STOCKS:
            panel.loc[sheets[tk].index, tk] = sheets[tk][feat]
        rank = panel.rank(axis=1, pct=True, na_option="keep")
        mu = panel.mean(axis=1)
        s = panel.std(axis=1, ddof=1)
        s = s.replace({0: np.nan})
        z = panel.sub(mu, axis=0).div(s, axis=0)
        for tk in STOCKS:
            sheets[tk][f"{feat}__rank"] = rank[tk]
            sheets[tk][f"{feat}__z"] = z[tk]
    for tk, s in sheets.items():
        s.to_csv(os.path.join(CS_DIR, f"{tk}_cs.csv"))
    print(f"Cross-sectional features written to {CS_DIR} "
          f"({len(BASE_FEATURES)}x2 extra columns per stock).")


def load_daily_cs(tk):
    df = pd.read_csv(os.path.join(CS_DIR, f"{tk}_cs.csv"),
                     index_col="Date", parse_dates=True)
    df["target"] = (df["Close"].shift(-HORIZON) > df["Close"]).astype(int)
    df["fwd_ret"] = df["Close"].pct_change().shift(-HORIZON)
    fc = [c for c in df.columns if c not in EXCLUDE and c not in ("target", "fwd_ret")]
    df = df.dropna(subset=["target"] + fc)
    return df, fc


def wf_predictions(df, fc):
    X = df[fc].values
    y = df["target"].values
    dates = np.asarray(df.index.astype(str))
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
    return {m: prob[m][valid] for m in MODELS}, y[valid], dates[valid], df["fwd_ret"].values[valid]


def rank_ic_eval(obs):
    """obs: DataFrame with columns date, stock, p (probability), ret (next-day return)."""
    ic_series = []
    rand_rows = []
    for d, grp in obs.groupby("date"):
        grp = grp.dropna(subset=["p", "ret"])
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        ic, _ = spearmanr(grp["p"], grp["ret"])
        if np.isnan(ic):
            continue
        ic_series.append(ic)
        rrow = []
        for seed in range(N_RAND_SEEDS):
            rng = np.random.RandomState(seed + 1000)
            shuf = grp["p"].to_numpy().copy()
            rng.shuffle(shuf)
            ric, _ = spearmanr(shuf, grp["ret"].to_numpy())
            rrow.append(np.nan if np.isnan(ric) else ric)
        rand_rows.append(rrow)
    ic_series = np.asarray(ic_series)
    rand_rows = np.asarray(rand_rows, dtype=float)
    mean_ic = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1))
    icir = mean_ic / ic_std if ic_std > 0 else 0.0
    t_stat = icir * np.sqrt(len(ic_series))
    norm = __import__("scipy.stats", fromlist=["norm"]).norm
    pval_ic = 2.0 * (1 - norm.cdf(abs(t_stat)))
    rand_mean = float(rand_rows.mean()) if rand_rows.size else 0.0
    delta = ic_series - rand_rows.mean(axis=1)
    t_delta, p_delta = ttest_1samp(delta, 0.0) if len(delta) > 1 else (0.0, 1.0)
    return {
        "n_dates": len(ic_series),
        "mean_ic": round(mean_ic, 4),
        "ic_std": round(ic_std, 4),
        "icir": round(icir, 4),
        "ic_pos_ratio": round(float(np.mean(ic_series > 0)), 4),
        "t_ic": round(float(t_stat), 2),
        "p_ic": round(float(pval_ic), 4),
        "rand_mean_ic": round(rand_mean, 4),
        "t_vs_random": round(float(t_delta), 2),
        "p_vs_random": round(float(p_delta), 4),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    build_cross_sectional_features()

    done = set()
    rows = []
    if os.path.exists(PARTIAL):
        prev = pd.read_csv(PARTIAL)
        done = set(prev["Stock"])
        rows = prev.to_dict("records")

    pooled = {m: {"y": [], "p": [], "date": [], "stock": [], "ret": []} for m in MODELS}
    n_done = len(done)
    for tk in STOCKS:
        if tk in done:
            n_done += 0
            continue
        df, fc = load_daily_cs(tk)
        probs, yv, dates, fwd_ret = wf_predictions(df, fc)
        for m in MODELS:
            pooled[m]["y"].append(yv)
            pooled[m]["p"].append(probs[m])
            pooled[m]["date"].append(dates)
            pooled[m]["stock"].append(np.full(len(yv), tk))
            pooled[m]["ret"].append(fwd_ret)
            acc = accuracy_score(yv, probs[m] > 0.5)
            bal = balanced_accuracy_score(yv, probs[m] > 0.5)
            auc = roc_auc_score(yv, probs[m])
            mcc = matthews_corrcoef(yv, probs[m] > 0.5)
            rows.append({"Stock": tk, "Model": m, "n": int(len(yv)),
                         "Accuracy": round(float(acc), 4),
                         "BalancedAccuracy": round(float(bal), 4),
                         "BaseRate": round(float(yv.mean()), 4),
                         "AUC": round(float(auc), 4), "MCC": round(float(mcc), 4)})
        pd.DataFrame(rows).to_csv(PARTIAL, index=False)
        n_done += 1
        print(f"[{n_done}/28] {tk} (n={len(yv)}) done", flush=True)

    pool_rows = []
    obs_by_model = {}
    for m in MODELS:
        y = np.concatenate(pooled[m]["y"])
        p = np.concatenate(pooled[m]["p"])
        bal = balanced_accuracy_score(y, p > 0.5)
        pool_rows.append({"Model": m, "n": int(len(y)),
                          "Accuracy": round(float(accuracy_score(y, p > 0.5)), 4),
                          "BalancedAccuracy": round(float(bal), 4),
                          "BaseRate": round(float(y.mean()), 4),
                          "AUC": round(float(roc_auc_score(y, p)), 4),
                          "MCC": round(float(matthews_corrcoef(y, p > 0.5)), 4)})
        obs_by_model[m] = pd.DataFrame({
            "date": np.concatenate(pooled[m]["date"]),
            "stock": np.concatenate(pooled[m]["stock"]),
            "p": p, "ret": np.concatenate(pooled[m]["ret"]),
        })

    ic_rows = []
    for m in MODELS:
        ic_rows.append({"Model": m, **rank_ic_eval(obs_by_model[m])})

    pd.DataFrame(pool_rows).to_csv(os.path.join(RESULTS_DIR, "cs_pooled.csv"), index=False)
    pd.DataFrame(ic_rows).to_csv(os.path.join(RESULTS_DIR, "cs_rank_ic.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "cs_per_stock.csv"), index=False)
    pd.DataFrame(rows).to_csv(PARTIAL, index=False)
    print("\n=== Pooled (cross-sectional features) ===")
    print(pd.DataFrame(pool_rows).to_string(index=False))
    print("\n=== Rank-IC (Vibe-style cross-sectional evaluation) ===")
    print(pd.DataFrame(ic_rows).to_string(index=False))
    print("\nSaved cs_pooled.csv, cs_rank_ic.csv, cs_per_stock rows in cs_partial.csv.")


if __name__ == "__main__":
    main()