"""app.py -- SHAP explanation browser (read-only demo UI).

Run:  .venv\\Scripts\\python.exe app.py   ->  http://127.0.0.1:5000

Reads the precomputed cache in Results/ui_cache (see src/UI_precompute.py)
and, for any chosen stock / day / model, shows the model's prediction, its
probability, the top SHAP contributions as a horizontal bar chart, and a
natural-language explanation (cached for the 50 sampled cases, or generated
live from Gemini when GEMINI_API_KEY is set).

The models operate at the chance level; the UI is an explanatory demo only.
"""

import io
import os
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request, send_file

sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject\src")
from SHAP_analysis import FEATURE_META

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
CACHE_DIR = os.path.join(BASE_DIR, "Results", "ui_cache")
LLM_CSV = os.path.join(BASE_DIR, "Results", "shap", "llm_explanations.csv")

MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
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

cache = {}
dates_by_stock = {}
manifest = {}
if os.path.exists(os.path.join(CACHE_DIR, "manifest.json")):
    with open(os.path.join(CACHE_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for tk in manifest.get("stocks", []):
        dates_by_stock[tk] = manifest["window"][tk]["dates"] or []

for fname in os.listdir(CACHE_DIR):
    if not fname.endswith(".npz"):
        continue
    tk, mname = fname.replace(".npz", "").split("_", 1)
    d = np.load(os.path.join(CACHE_DIR, fname), allow_pickle=True)
    cache[(tk, mname)] = d

llm_cache = {}
if os.path.exists(LLM_CSV):
    le = pd.read_csv(LLM_CSV)
    for _, r in le.iterrows():
        llm_cache[(str(r["stock"]), str(r["model"]), str(r["date"]))] = str(r["explanation"])

app = Flask(__name__)


def load_api_key():
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    kf = os.environ.get("GEMINI_KEY_FILE", "").strip()
    if not kf:
        kf = os.path.join(os.path.expanduser("~"), ".gemini_api_key")
    try:
        with open(kf, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


API_KEY = load_api_key()


def get_case(stock, model, date):
    key = (stock, model)
    if key not in cache:
        return None, "unknown model"
    d = cache[key]
    iso = np.array([str(x) for x in d["dates"]])
    idx = np.where(iso == date)[0]
    if len(idx) == 0:
        return None, "date outside this model's test window"
    i = int(idx[0])
    base = float(d["base"])
    prob = float(d["prob"][i])
    y = int(d["y"][i])
    pred = 1 if prob > 0.5 else 0
    feats = [str(x) for x in d["feats"]]
    row = d["Xraw"][i]
    vals = d["shap"][i]
    order = np.argsort(-np.abs(vals))[:5]
    top = [{
        "feature": feats[j],
        "description": FEATURE_META.get(feats[j], ("Technical", "Indicator"))[1],
        "category": FEATURE_META.get(feats[j], ("Technical", "Indicator"))[0],
        "value": round(float(row[j]), 4),
        "shap": round(float(vals[j]), 4),
        "direction": "UP" if vals[j] > 0 else "DOWN",
    } for j in order]
    has_llm = (stock, model, str(date)) in llm_cache
    return {
        "stock": stock, "model": model, "date": str(date),
        "predicted": "UP" if pred == 1 else "DOWN",
        "true": "UP" if y == 1 else "DOWN",
        "correct": bool(pred == y),
        "prob_up": round(prob, 4),
        "confidence": round(abs(prob - 0.5), 4),
        "base_rate": round(base, 4),
        "top_features": top,
        "waterfall_url": f"/api/waterfall?stock={stock}&model={model}&date={date}",
        "has_cached_llm": has_llm,
        "cached_llm_text": llm_cache.get((stock, model, str(date))) or None,
    }, None


@app.route("/")
def index():
    stocks = json.dumps(dates_by_stock)
    opts = "".join(f'<option value="{m}">{m}</option>' for m in MODELS)
    q_stock = request.args.get("stock", "")
    q_model = request.args.get("model", "")
    q_date = request.args.get("date", "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SHAP Explanation Browser</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#1c2733}}
 .banner{{background:#b3261e;color:#fff;padding:10px 18px;font-size:14px}}
 .wrap{{max-width:1000px;margin:24px auto;padding:0 16px}}
 .card{{background:#fff;border-radius:10px;padding:18px 22px;box-shadow:0 1px 3px rgba(0,0,0,.12);margin-bottom:18px}}
 label{{font-weight:600;margin-right:8px}} select,input,button{{font-size:14px;padding:6px 10px;margin:4px 6px 4px 0;border:1px solid #c6ccd4;border-radius:6px}}
 button{{background:#1a73e8;color:#fff;border:none;cursor:pointer}} button:hover{{background:#1765cc}}
 .facts{{display:flex;gap:18px;flex-wrap:wrap}} .fact b{{display:block;font-size:11px;color:#5f6b76;text-transform:uppercase}}
 .fact span{{font-size:20px}} .badge{{padding:2px 10px;border-radius:20px;color:#fff;font-weight:600}}
 .up{{background:#1e8e3e}} .down{{background:#b3261e}} .ok{{background:#1e8e3e}} .err{{background:#b3261e}} .warn{{background:#f29900}}
 table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #e2e6ea;padding:7px 9px;text-align:left;font-size:14px}}
 img{{max-width:100%;border:1px solid #e2e6ea;border-radius:8px}}
 #llm{{white-space:pre-wrap;line-height:1.5}} .hint{{color:#5f6b76;font-size:13px}}
</style></head><body>
<div class="banner">Explanatory demo only: all models perform at the chance level (~50% accuracy). Nothing here is trading advice.</div>
<div class="wrap">
 <div class="card"><h2 style="margin-top:0">Model Prediction &amp; SHAP Explanation</h2>
  <label>Stock</label><select id="stock"></select>
  <label>Model</label><select id="model">{opts}</select>
  <label>Day</label><input type="date" id="date">
  <button onclick="loadCase()">Show explanation</button>
  <button onclick="generateLLM()" style="background:#5a6472">Ask Gemini (this day)</button>
 </div>
 <div class="card" id="case">Select a stock and day, then click "Show explanation".</div>
</div>
<script>
const DICT = {stocks};
const PRESET = {json.dumps({"stock": q_stock, "model": q_model, "date": q_date})};
const sel=document.getElementById('stock'), dateEl=document.getElementById('date');
function fmt(ds){{const el=document.getElementById('date');el.min=ds[0];el.max=ds[ds.length-1];el.value=ds[Math.min(ds.length-1,Math.floor(ds.length*0.8))];}}
Object.keys(DICT).sort().forEach(s=>{{const o=document.createElement('option');o.value=s;o.text=s;sel.appendChild(o)}});
const first=Object.keys(DICT).sort()[0]; sel.value=first; fmt(DICT[first]);
sel.onchange=()=>{{
 const ds=DICT[sel.value]||[];
 if(ds.length){{dateEl.min=ds[0];dateEl.max=ds[ds.length-1];dateEl.value=ds[0];}}
}};
if(PRESET.stock && DICT[PRESET.stock]){{sel.value=PRESET.stock; const ds=DICT[PRESET.stock]; dateEl.min=ds[0]; dateEl.max=ds[ds.length-1]; dateEl.value=PRESET.date||ds[0];}}
if(PRESET.model){{document.getElementById('model').value=PRESET.model;}}
if(PRESET.stock){{loadCase();}}
async function loadCase(){{
 const q=new URLSearchParams({{stock:sel.value,model:document.getElementById('model').value,date:document.getElementById('date').value}});
 const r=await fetch('/api/case?'+q); const c=await r.json();
 if(c.error){{document.getElementById('case').innerHTML='<b>'+c.error+'</b>';return;}}
 const f=c.top_features.map(t=>`<tr><td><b>${{t.feature}}</b><br><span class="hint">${{t.category}} &mdash; ${{t.description}}</span></td><td>${{t.value}}</td><td>${{t.shap}}</td><td><span class="badge ${{t.direction==='UP'?'up':'down'}}">${{t.direction}}</span></td></tr>`).join('');
 document.getElementById('case').innerHTML=
  `<div class="facts">
   <div class="fact"><b>Prediction</b><span><span class="badge ${{c.predicted==='UP'?'up':'down'}}">${{c.predicted}}</span></span></div>
   <div class="fact"><b>Actual</b><span><span class="badge ${{c.true==='UP'?'up':'down'}}">${{c.true}}</span></span></div>
   <div class="fact"><b>P(UP)</b><span>${{c.prob_up}}</span></div>
   <div class="fact"><b>Baseline</b><span>${{c.base_rate}}</span></div>
   <div class="fact"><b>Outcome</b><span><span class="badge ${{c.correct?'ok':'err'}}">${{c.correct?'correct':'wrong'}}</span></span></div>
  </div>
  <h3>Top-5 feature contributions</h3>
  <table><tr><th>Feature</th><th>Value</th><th>SHAP</th><th>Pushes</th></tr>${{f}}</table>
  <h3>SHAP contributions</h3><img src="${{c.waterfall_url}}">
  <div id="llm" style="margin-top:12px">${{c.cached_llm_text?`<div class="facts" style="margin-bottom:8px"><div class="fact"><b>Generated via</b><span>${{c.has_cached_llm?'cached explanation':'--'}}</span></div></div>${{c.cached_llm_text}}`:`<span class="hint">No explanation generated for this day yet &mdash; click "Ask Gemini (this day)".</span>`}}</div>`;
}}
async function generateLLM(){{
 const q=new URLSearchParams({{stock:sel.value,model:document.getElementById('model').value,date:document.getElementById('date').value}});
 const box=document.getElementById('llm'); if(!box) return;
 box.innerHTML='<span class="hint">Generating&hellip;</span>';
 const r=await fetch('/api/explain?'+q); const c=await r.json();
 box.innerHTML=c.error?'<b>'+c.error+'</b>':`<div class="facts"><div class="fact"><b>Generated via</b><span>${{c.source}}</span></div></div>${{c.text}}`;
}}
</script></body></html>"""


@app.route("/api/dates")
def api_dates():
    tk = request.args.get("stock", "")
    return jsonify({"dates": dates_by_stock.get(tk, [])})


@app.route("/api/case")
def api_case():
    stock = request.args.get("stock", "")
    model = request.args.get("model", "")
    date = request.args.get("date", "")
    case, err = get_case(stock, model, date)
    if err:
        return jsonify({"error": err})
    return jsonify(case)


@app.route("/api/waterfall")
def api_waterfall():
    stock = request.args.get("stock", "")
    model = request.args.get("model", "")
    date = request.args.get("date", "")
    if (stock, model) not in cache:
        return jsonify({"error": "unknown model"}), 404
    d = cache[(stock, model)]
    iso = np.array([str(x) for x in d["dates"]])
    idx = np.where(iso == date)[0]
    if len(idx) == 0:
        return jsonify({"error": "date outside test window"}), 404
    i = int(idx[0])
    feats = [str(x) for x in d["feats"]]
    vals = d["shap"][i]

    order = np.argsort(-np.abs(vals))[:10]
    n = len(order)
    labels = [feats[j] for j in order][::-1]
    values = [float(vals[j]) for j in order][::-1]

    pos_col = "#1e8e3e"
    neg_col = "#b3261e"
    colors = [pos_col if v >= 0 else neg_col for v in values]
    m = max(abs(v) for v in values) or 1.0

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * n + 0.6)))
    ax.barh(np.arange(n), values, color=colors, edgecolor="none", height=0.66)
    ax.axvline(0, color="#5f6b76", linewidth=0.8)
    pad = 0.02 * m
    for v, k in zip(values, np.arange(n)):
        ax.text(v + pad, k, f"{v:.4f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9, color="#1c2733")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-m * 1.22, m * 1.22)
    ax.set_xlabel("SHAP contribution (push toward UP or DOWN)", fontsize=9)
    ax.set_title(f"{stock} \u2014 {model} \u2014 {date}", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/explain")
def api_explain():
    stock = request.args.get("stock", "")
    model = request.args.get("model", "")
    date = request.args.get("date", "")
    if (stock, model, date) in llm_cache:
        return jsonify({"source": "cached", "text": llm_cache[(stock, model, date)]})
    case, err = get_case(stock, model, date)
    if err:
        return jsonify({"error": err})
    key = API_KEY
    if not key:
        return jsonify({"error": "No cached explanation and no Gemini key is configured for this session (set GEMINI_API_KEY, or put the key in ~/.gemini_api_key), so live generation is unavailable."})
    try:
        from google import genai
        import time as _time
        from google.genai import errors as _gerr
        client = genai.Client(api_key=key)
        model_id = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
        top = json.dumps(case["top_features"], indent=1)
        user_text = (
            f"Explain this stock-movement model prediction.\n"
            f"Stock: {case['stock']}; Date: {case['date']}; Model: {case['model']}.\n"
            f"Prediction: {case['predicted']} with probability {case['prob_up']} "
            f"(baseline {case['base_rate']}); the true move was {case['true']}.\n"
            f"Top contributing features:\n{top}"
        )
        last = None
        for k in range(4):
            try:
                resp = client.models.generate_content(model=model_id, contents=(SYSTEM_PROMPT, user_text))
                return jsonify({"source": "generated", "text": resp.text.strip()})
            except _gerr.APIError as e:
                last = e
                code = getattr(e, "code", None)
                status = getattr(getattr(e, "response", None), "status_code", None)
                if isinstance(e, _gerr.ServerError) or code in (429, 500, 502, 503, 504) \
                        or status in (429, 500, 502, 503, 504):
                    _time.sleep(6 * (k + 1))
                    continue
                raise
        return jsonify({"error": f"live generation failed after retries: {last}"})
    except Exception as e:
        return jsonify({"error": f"live generation failed: {e}"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)