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
daily Rank-IC reproduces the same null result, and adding leakage-safe S&P 500
and VIX market-context features moves balanced accuracy by less than one
percentage point. The project is therefore a
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
| Data | `Download_index_data.py` | Pulls ^GSPC (S&P 500) and ^VIX daily OHLCV, 2018-2026 |
| Data | `Backfill_data.py` | Extends history back to 2018-01-01 for pre-COVID/COVID/post-COVID regimes |
| Feature engineering | `Preprocess_data.py` | Cleaning, outlier removal, 23 indicators (RSI, MACD, EMA, ATR, Bollinger, ROC, lags, MA, volatility); optional `--with-market` merge |
| Feature engineering | `Market_features.py` | Leakage-safe ^GSPC/^VIX features (14 market + 5 stock-relative) used by `--with-market` |
| Quality | `Test_market_features.py` | Blocking leakage audit for the market features (truncated builds, AST scan, correlation controls) |
| Feature engineering | `Enhance_features.py` | Market-relative and calendar feature variants |
| Modelling | `LogisticRegression_model.py`, `RandomForest_model.py`, `XGBoost_model.py` | Baseline classifiers, per-stock train/evaluate |
| Modelling | `main.py` | Orchestrates preprocess -> train -> summary CSV (10-stock pilot) |
| Validation | `Full_run.py` | **Headline numbers**: leakage-free walk-forward on all 28 stocks, pooled metrics incl. balanced accuracy and base rate; `--tag`, `--feature-set`, `--market-lag` |
| Validation | `Compare_runs.py` | Paired A/B comparison of two runs: deltas with date-block bootstrap CI and stock-date McNemar |
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

## Market-context experiment (^GSPC / ^VIX)

The 23 technical indicators are stock-local, so the headline experiment cannot
say whether a market-wide risk state carries next-day direction information.
The market-context extension adds 14 S&P 500 / VIX series features (1/5/20-day
returns, realised volatility, trend, drawdown, VIX level / change / z-score /
regime flags) and 5 stock-relative features (excess return over 1/5/20 days,
60-day beta and correlation) — 42 features in total.

**Data.** `Download_index_data.py` pulls ^GSPC and ^VIX from 2018-01-01
(`auto_adjust=True`) so the pre-COVID, COVID and post-COVID regimes are all
present. `Market_features.py` builds the feature block from the index closes
and left-joins it onto each stock's frame;
`Preprocess_data.py --with-market --out-dir Data/cleaned_mkt` writes the
augmented dataset. The 44 technical columns are byte-identical to
`Data/cleaned` and the row counts/dates match exactly, so the A/B isolates the
19 new columns.

**Leakage contract.** Row *t* may only use index information observable at the
close of *t*: every market series is same-day or backward-looking
(`pct_change`, `rolling`, `shift(+k)`), never `shift(-k)`, `bfill` or a centred
window. Index rows where ^GSPC has no close (for example a stale ^VIX quote on
a US holiday) are dropped before returns are computed, so a holiday cannot
create a phantom row or a forward-filled stale return.
`Test_market_features.py` enforces the contract with truncation tests
(rebuilding on a data prefix must not change earlier rows), an AST scan for
look-ahead primitives, coverage checks and a leaky-control correlation test; it
exits non-zero on failure.

**Results.** Identical protocol to the headline run (HORIZON=1, STEP=20,
MIN_TRAIN=0.6, 3 seeds); only the feature matrix changes.

| Feature set | n | LR acc. | RF acc. | XGB acc. |
| --- | --- | --- | --- | --- |
| Technical only (baseline) | 22,520 | 50.28% | 50.24% | 50.07% |
| Technical + market | 22,520 | 50.62% | 50.32% | 50.51% |
| Market only | 22,520 | 51.25% | 51.13% | 50.96% |
| Technical + market, market lagged 1 day | 22,460 | 50.41% | 50.30% | 49.78% |

Paired comparisons (`Compare_runs.py` on identical stock-dates, with a
date-block bootstrap because the market block is shared by all stocks on a
date):

- **Technical + market vs baseline:** +0.34 / +0.07 / +0.44 pp for LR / RF /
  XGB. Every 95% CI crosses zero (e.g. LR accuracy −0.38 to +1.05 pp),
  McNemar p ≥ 0.18, and only 14–16 of 28 stocks improve. **Null.**
- **Market only vs baseline:** +0.97 / +0.89 / +0.89 pp; 18–19 of 28 stocks
  improve; McNemar exact p = 0.045 / 0.097 / 0.104 (χ² 0.022 / 0.048 / 0.052);
  balanced accuracy rises only to ≈50.4–50.5%. Directionally the strongest
  variant, but borderline and still far from a usable edge.
- **Market lagged 1 day vs market:** XGB −0.75 pp (bootstrap CI −1.39 to
  −0.07, McNemar p = 0.043), LR/RF unchanged. Removing same-close market
  information erases the small XGB gain — the expected control, since the weak
  signal lives in the close-to-close overlap rather than in persistent market
  state.

Conclusion: market context is worth at most a fraction of a percentage point
of balanced accuracy under a protocol where the base rate alone is 53.2%. It
does not overturn the null result, and the technical block dilutes rather than
complements the market signal.

**Reproduce.**

```bash
python src/Download_index_data.py
python src/Test_market_features.py
python src/Preprocess_data.py --with-market --out-dir Data/cleaned_mkt
python src/Full_run.py --data-dir Data/cleaned --tag baseline
python src/Full_run.py --data-dir Data/cleaned_mkt --tag market
python src/Full_run.py --data-dir Data/cleaned_mkt --tag market_only --feature-set market
python src/Full_run.py --data-dir Data/cleaned_mkt --tag market_lag1 --market-lag 1
python src/Compare_runs.py --baseline baseline --market market
python src/Compare_runs.py --baseline baseline --market market_only
python src/Compare_runs.py --baseline market --market market_lag1
```

## Diagnostics

- src/Learning_curves.py — train/validation log-loss curves vs training size (per stock/model)
- src/train_test_loss_all.py — generates train/test log-loss learning curves for all 28 stocks (3 models) and saves Results/train_test_loss/train_test_loss_all.csv and Plots/train_test_loss_all/tl_{TICKER}.png
- src/Bias_variance.py, src/Bias_variance_full.py, src/Bias_variance_all_withplots.py — bias–variance decomposition vs model complexity; all-28 run produces Results/bias_variance/bias_variance_all.csv and per-stock plots in Plots/bias_variance_all/bv_{TICKER}.png. Results consistently show high bias (flat near the base rate) and low variance across capacities, indicating a low-signal (high-bias) regime rather than overfitting.
