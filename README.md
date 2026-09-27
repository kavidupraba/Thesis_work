# ML-Based Stock Price Direction Prediction with Explainable AI

A master's-thesis project that builds a complete, leakage-free pipeline for
next-day stock-movement prediction, explains the resulting models with SHAP,
paraphrases the explanations with an LLM, evaluates the signals for trading
actionability, and packages everything into a browsable Flask demo.

The three classifiers (Logistic Regression, Random Forest, XGBoost), trained
on 23 technical indicators for 28 stocks over 2018-2026, all perform at the
chance level under a strictly leakage-free walk-forward protocol
(roughly 50% balanced accuracy, ROC AUC ~ 0.50, MCC ~ 0). A cross-sectional
extension that adds daily relative rank and z-score features and scores the
daily Rank-IC reproduces the same null result. The project is therefore a
careful demonstration of how to test and falsify a momentum-style technical
signal: it traces an apparent 20-day "signal" to a walk-forward leak, corrects
it, and shows that no out-of-sample edge survives. This is consistent with the
weak form of the Efficient Market Hypothesis.

**Explanatory research demo only. Nothing in this repository is trading
advice.**

## Repository layout

```
app.py                 Flask web UI (stock / model / day -> explanation)
src/                   All pipeline and analysis scripts
Data/                  Raw + cleaned CSV data (gitignored, not committed)
Results/               Metrics, SHAP arrays, caches, LLM log (gitignored)
Plots/                 Generated figures (gitignored)
static/                UI stylesheet + precomputed waterfall PNGs
templates/             Legacy Jinja templates (the served UI is generated in app.py)
README.md
```

## Pipeline scripts (src/)

| Stage | Script | What it does |
| --- | --- | --- |
| Data | `Download_data.py` | Pulls daily OHLCV from Yahoo Finance (`auto_adjust=True`) for the 28 tickers |
| Data | `Download_index_data.py` | Pulls ^GSPC (S&P 500) index data |
| Data | `Backfill_data.py` | Extends history back to 2018-01-01 for pre-COVID/COVID/post-COVID regimes |
| Feature engineering | `Preprocess_data.py` | Cleaning, outlier removal, 23 indicators (RSI, MACD, EMA, ATR, Bollinger, ROC, lags, MA, volatility) |
| Feature engineering | `Enhance_features.py` | Market-relative and calendar feature variants |
| Modelling | `LogisticRegression_model.py`, `RandomForest_model.py`, `XGBoost_model.py` | Baseline classifiers, per-stock train/evaluate |
| Modelling | `main.py` | Orchestrates preprocess -> train -> summary CSV (10-stock pilot) |
| Validation | `Full_run.py` | **Headline numbers**: leakage-free walk-forward on all 28 stocks, pooled metrics incl. balanced accuracy and base rate |
| Validation | `Significance_test.py` | McNemar pairwise significance (10-stock pilot) |
| Validation | `Evaluate_20d.py` | 3/5/20-day horizon sensitivity, walk-forward ensembles |
| Validation | `Cross_sectional_experiment.py` | Relative rank/z features + daily Rank-IC with permutation control (negative result) |
| Explainability | `SHAP_analysis.py` | Global SHAP importance per stock/model; `FEATURE_META` shared by the UI |
| Explainability | `Full_shap_28.py` | Global SHAP at 28-stock scale |
| Explainability | `Local_SHAP.py` | Per-day waterfall plots for representative cases |
| Explainability | `Spearman_consistency.py` | Cross-model explanation agreement (RQ3) |
| Explainability | `Cross_stock_stability.py` | Cross-stock SHAP ranking stability |
| Explainability | `Regime_analysis.py` | Pre-COVID/COVID/post-COVID stratification, leakage-free (RQ4) |
| Explainability | `LLM_explanations.py` | 50 sampled cases paraphrased by Gemini, cached + manually verified |
| Actionability | `Actionability.py` | Probability-threshold signals vs do-nothing baseline (no edge) |
| UI | `Ui_cache_builder.py` / `UI_precompute.py` | Precompute numpy caches the Flask app serves |
| UI | `app.py` | The web demo itself |
| Quality | `smoke_test_app.py` | End-to-end endpoint tests for the UI |

## Leakage-free protocol (why the numbers are trustworthy)

The headline results use walk-forward cross-validation with expanding
training windows: 20-day test windows, a minimum 60% training share, three
seeds averaged on probabilities, a per-window `StandardScaler` fit only on
the training segment (and only for Logistic Regression), and an explicit
one-day buffer between the last training row and the first test label. An
early version of the pipeline showed an apparent edge at the 20-day horizon;
this was traced to the training window including rows whose labels overlapped
the test period (boundary look-ahead). After correcting the protocol every
model returned to the noise floor.

## Web interface

```
python app.py      # then open http://127.0.0.1:5000
```

Pick a stock (10 pilot stocks), a model, and a day, and the page shows the
prediction with its UP probability and chance baseline, the actual outcome,
the top-5 SHAP contributions, a horizontal SHAP contribution bar chart
(positive contributions green, negative red, value labelled at each bar tip),
and a natural-language explanation. Explanations are drawn from the cached
50-case verification log where available, otherwise generated live via the
Gemini API (key from `GEMINI_API_KEY` env var or `~/.gemini_api_key`; model
`gemini-3.1-flash-lite` by default, override with `GEMINI_MODEL`). Cases can
be deep-linked with query parameters, e.g.
`http://127.0.0.1:5000/?stock=AAPL&model=XGBoost&date=2025-08-06`.

## Setup

Python 3.x (developed on 3.11). There is no `requirements.txt`; install the
dependencies directly:

```
pip install numpy pandas scikit-learn xgboost shap matplotlib flask yfinance
```

Generated artifacts (CSVs, numpy caches, plots, screenshots, data) are
gitignored and must be produced by running the scripts above. The Flask app
expects the precomputed cache under `Results/ui_cache/` (built by
`src/UI_precompute.py`).

## Key results

Pooled over 28 stocks, 22,520 leakage-free out-of-sample predictions:

| Model | Accuracy | Balanced acc. | Base rate | ROC AUC | MCC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 50.3% | 49.6% | 53.2% | 0.498 | -0.008 |
| Random Forest | 50.2% | 50.1% | 53.2% | 0.503 | 0.001 |
| XGBoost | 50.1% | 50.0% | 53.2% | 0.502 | -0.000 |

Pooled McNemar p = 1.00 (no pairwise difference). Cross-sectional extension
(69 features, 22,460 predictions): best daily mean Rank-IC 0.0065, i.e. within
the band of pure chance and far below the ~0.02 threshold at which published
factor libraries consider daily IC tradeable.