"""
LLM_explanations.py
===================
Task: LLM integration for the XAI pipeline.

Stage A - build_cases():
    Recomputes SHAP values for the three models on the 10-stock pilot (current
    2018-2026 cleaned data, chronological 80/20 split, same protocol as
    Local_SHAP.py), deterministically samples 50 test predictions, and exports
    a structured JSON (Results/shap/llm_cases.json) holding everything the LLM
    needs: prediction, true label, probability, base rate, top feature
    contributions with actual values, and a short market context.

Stage B - generate():
    Reads llm_cases.json, calls the Google Gemini API (google-genai) with a
    fixed system prompt, and writes one natural-language explanation per case
    to:
      Results/shap/llm_explanations.csv
      Results/shap/llm_explanations.md   (human-readable verification log)

Usage:
    python src/LLM_explanations.py                       # Stage A only
    python src/LLM_explanations.py generate              # Stage B (needs key)
Environment:
    GEMINI_API_KEY   required for Stage B
    GEMINI_MODEL     optional model id (default gemini-2.5-flash)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import shap

sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject\src")
from SHAP_analysis import (
    load_and_prepare, model_logreg, model_rf, model_xgb, FEATURE_META,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
SHAP_DIR = os.path.join(BASE_DIR, "Results", "shap")
CASES_JSON = os.path.join(SHAP_DIR, "llm_cases.json")
EXPL_CSV = os.path.join(SHAP_DIR, "llm_explanations.csv")
EXPL_MD = os.path.join(SHAP_DIR, "llm_explanations.md")

MODELS = {"LogisticRegression": model_logreg, "RandomForest": model_rf, "XGBoost": model_xgb}
STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "WMT", "XOM", "GS", "TSLA", "V"]
TEST_SIZE = 0.2
N_CASES = 50
TOP_K = 7
SEED = 42


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


def top_contribs(fc, vals, Xte, k=TOP_K):
    """Top-k features by |SHAP| with pre-scaled (raw) values and metadata."""
    order = np.argsort(-np.abs(vals))
    out = []
    for j in order[:k]:
        fname = fc[j]
        meta = FEATURE_META.get(fname, ("Other", "Technical indicator"))
        out.append({
            "feature": fname,
            "description": meta[1],
            "category": meta[0],
            "value": round(float(Xte[j]), 4),
            "shap": round(float(vals[j]), 4),
            "direction": "pushes toward UP" if vals[j] > 0 else "pushes toward DOWN",
        })
    return out


def build_cases():
    rng = np.random.RandomState(SEED)
    pool = []
    for tk in STOCKS:
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

            # market context columns available on the cleaned df
            close = df["Close"].values[split:]
            rets = df["returns"].values[split:]
            vol = df["Volume"].values[split:]

            for i in range(len(Xte)):
                d = dates[i]
                # 5-day return (safe window: we are past warm-up)
                k0 = max(0, i - 4)
                ret5 = (close[i] / close[k0] - 1.0) * 100.0 if close[k0] > 0 else 0.0
                vol_ma = float(np.nanmean(df["Volume"].values[split - 20:split]))
                pool.append({
                    "stock": tk,
                    "model": name,
                    "date": d.strftime("%Y-%m-%d"),
                    "true": "UP" if yte[i] == 1 else "DOWN",
                    "predicted": "UP" if pred[i] == 1 else "DOWN",
                    "prob_up": round(float(probs[i]), 4),
                    "base_rate": round(base, 4),
                    "correct": bool(pred[i] == yte[i]),
                    "prob_margin": round(abs(probs[i] - 0.5), 4),
                    "day_return_pct": round(rets[i] * 100.0, 3),
                    "ret5_pct": round(ret5, 3),
                    "price": round(float(close[i]), 2),
                    "volume_ratio_20d": round(float(vol[i]) / vol_ma, 3) if vol_ma > 0 else 1.0,
                    "top_features": top_contribs(fc, vals[i], Xte_m[i]),
                })

    rng.shuffle(pool)
    seen_dates = set()
    cases = []
    for case in pool:
        if len(cases) >= N_CASES:
            break
        if case["date"] in seen_dates:
            continue
        seen_dates.add(case["date"])
        cases.append(case)

    os.makedirs(SHAP_DIR, exist_ok=True)
    with open(CASES_JSON, "w", encoding="utf-8") as f:
        json.dump({"n_cases": len(cases), "cases": cases}, f, indent=2,
                  default=lambda o: float(o))

    n_corr = sum(c["correct"] for c in cases)
    stocks_used = sorted({c["stock"] for c in cases})
    models_used = sorted({c["model"] for c in cases})
    print(f"Wrote {len(cases)} cases -> {CASES_JSON}")
    print(f"  stocks: {stocks_used}")
    print(f"  models: {models_used}")
    print(f"  correct: {n_corr}/{len(cases)}, dates spread over "
          f"{min(c['date'] for c in cases)} .. {max(c['date'] for c in cases)}")


SYSTEM_PROMPT = (
    "You write precise, plain-language explanations of machine-learning model "
    "decisions for a master's thesis on explainable stock-movement prediction. "
    "The model under discussion is known, from rigorous out-of-sample testing, "
    "to perform at the chance level (about 50% accuracy); your explanation must "
    "reflect that reality and must not claim or imply skill, profitability, or "
    "predictive power. Do not give investment advice. For each case: state the "
    "model's prediction in one sentence; describe in plain terms what each of "
    "the top contributing features said on that day (what the indicator means, "
    "its actual value, and which direction it pushed the prediction), leaning "
    "on the feature descriptions provided; then close with one sentence noting "
    "what this does and does not tell us, consistent with a no-better-than-"
    "chance model. Keep each explanation between 100 and 140 words, use no "
    "bullets, and do not hedge with 'may' more than once."
)


def call_with_retry(client, model_id, sys_prompt, user_text, attempts=8):
    import time
    from google.genai import errors
    last = None
    for k in range(attempts):
        try:
            resp = client.models.generate_content(
                model=model_id, contents=(sys_prompt, user_text)
            )
            return resp.text.strip()
        except errors.APIError as e:
            last = e
            code = getattr(e, "code", None)
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = (
                isinstance(e, errors.ServerError)
                or code in (429, 500, 502, 503, 504)
                or status in (429, 500, 502, 503, 504)
            )
            if not retryable:
                raise
            wait = min(5 * (2 ** k), 90)
            print(f"    (transient {status or code}, retry {k+1}/{attempts} in {wait}s)")
            time.sleep(wait)
    raise last


def generate():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY is not set. Set it and re-run 'generate'.")
        sys.exit(1)
    from google import genai

    model_id = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
    client = genai.Client(api_key=key)

    with open(CASES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    checkpoint = os.path.join(SHAP_DIR, "llm_cases_done.json")
    done = {}
    if os.path.exists(checkpoint):
        with open(checkpoint, "r", encoding="utf-8") as f:
            done = {r["case_key"]: r for r in json.load(f)}

    rows = []
    md = ["# LLM Explanation Verification Log",
          "",
          f"{len(data['cases'])} sampled predictions, model `{model_id}`.",
          ""]
    for i, case in enumerate(data["cases"], 1):
        key = f"{case['stock']}|{case['model']}|{case['date']}"
        if key in done:
            text = done[key]["explanation"]
        else:
            text = call_with_retry(
                client, model_id, SYSTEM_PROMPT,
                f"Explain this stock-movement model prediction.\n"
                f"Stock: {case['stock']}; Date: {case['date']}; Model: {case['model']}.\n"
                f"Prediction: {case['predicted']} with probability {case['prob_up']} "
                f"(the model's chance baseline is {case['base_rate']}); the true move "
                f"was {case['true']}, so the call was "
                f"{'correct' if case['correct'] else 'incorrect'}.\n"
                f"Market context on that day: price {case['price']}, "
                f"day return {case['day_return_pct']}%, 5-day return {case['ret5_pct']}%, "
                f"volume 20-day ratio {case['volume_ratio_20d']}.\n"
                f"Top contributing features (with description, actual value, SHAP "
                f"value, and direction):\n{json.dumps(case['top_features'], indent=1)}",
            )
            done[key] = {**case, "case_key": key, "explanation": text}
            with open(checkpoint, "w", encoding="utf-8") as f:
                json.dump(list(done.values()), f, indent=1)
        rows.append({**case, "explanation": text})
        md.append(
            f"## Case {i}: {case['stock']} {case['model']} {case['date']}\n\n"
            f"- True {case['true']} / Predicted {case['predicted']} "
            f"(P(up)={case['prob_up']}, base={case['base_rate']}, "
            f"{'correct' if case['correct'] else 'WRONG'})\n"
            f"- {text}\n"
        )
        print(f"[{i:02d}/{len(data['cases'])}] {case['stock']} {case['model']} "
              f"{case['date']} ({case['predicted']}) -> {len(text)} chars")

    os.makedirs(SHAP_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(EXPL_CSV, index=False)
    with open(EXPL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"\nSaved -> {EXPL_CSV}\nSaved -> {EXPL_MD}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate()
    else:
        build_cases()