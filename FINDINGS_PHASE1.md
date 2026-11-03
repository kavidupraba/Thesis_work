# Phase 1 findings — why the market block does not beat the drift

**Scope.** Phase 1 of the market-context experiment: an honest benchmark, proper
scoring, a calibration check, a power calculation and an index-level test of the
S&P 500 / VIX signal. No new features, no hyperparameter search, no threshold
tuning, no extra seeds.

**Status.** Complete on `working_development` (analysis committed as `7b6687f`;
this write-up and the figure script in the follow-up commit). The pre-registered
decision gate (plan §8) was met, so the market-features thread is closed with a
negative result.

**Evidence base.** Every number below is produced by code, not typed by hand:

| Source | Produces |
| --- | --- |
| `src/Full_run.py` | per-stock-date OOS probabilities, `p_prior`, `fold` |
| `src/Compare_runs.py` | benchmarks, Brier/log-loss, calibration, power → `Results/compare_*_{scoring,calibration,deltas,mcnemar}.csv` |
| `src/Market_signal_test.py` | HAC ICs, event study, index timing, ceiling → `Results/market_signal_*.csv` |
| `src/Phase1_figures.py` | all nine figures below → `Plots/phase1/` |

Sample: 28 stocks, 22,520 out-of-sample stock-days, 831 trading dates,
2023-02-06 to 2026-05-29.

---

## 0. The findings at a glance

| # | Claim | Headline number | Figure | Defence |
| --- | --- | --- | --- | --- |
| F1 | Every model is **worse than always-predict-UP**, not merely at chance | −1.94 to −3.12 pp vs 53.19%, all CIs exclude 0 | 1 | §4 Q1, Q2 |
| F2 | No model has any **skill over the expanding-window base rate** | Brier skill −0.011 to −0.094, CIs exclude 0 | 2 | §4 Q3 |
| F3 | The probabilities are **poorly calibrated and nearly flat** | calibration slope −0.03 to +0.11 (perfect = 1), ECE up to 0.12 | 3, 9a | §4 Q4 |
| F4 | The market block carries **a little probability information**, but no decision edge | Brier Δ −0.0018/−0.0051/−0.0093 (CIs exclude 0); accuracy Δ +0.89 to +0.97 pp (CIs include 0) | 4 | §4 Q5 |
| F5 | The index-level signal is **weak and not joint** | IC 0.056–0.113, joint HAC p = 0.52 | 5, 6 | §4 Q6 |
| F6 | At this signal strength even **perfect use of the market block cannot beat the drift by more than ~0.3 pp** | ceiling 53.20–53.52% vs drift 53.19%; market_only 50.96–51.25% | 7 | §4 Q7 |
| F7 | The study is **underpowered** for sub-pp effects | 831 dates; ~0.5 pp needs 3,200–6,800 dates | 8 | §4 Q8 |
| F8 | Half the labels are **economically tiny moves**, and the OOS window is one regime | 30.5% of \|moves\| < 0.5%; 15 dates with VIX > 30 | 9b | §4 Q9 |

Read in one sentence: **the models are 1.9–3.1 pp worse than the drift, the
market block adds a measurable but tiny amount of probability information, and
arithmetic shows that even perfect use of that information cannot produce an edge
over the drift.**

---

## Figure index — where each plot is written

`src/Phase1_figures.py` is the only script that produces Phase-1 figures. Running
`python src/Phase1_figures.py` writes exactly these nine PNGs into
`Plots/phase1/`, relative to the repository root. `Plots/` is gitignored by
design — no plot is committed — so this table is the canonical map from a finding
to the file you cite in the thesis.

| Fig | Path (relative to repo root) | Proves | Built from | Caption for the thesis |
| --- | --- | --- | --- | --- |
| 1 | `Plots/phase1/fig1_accuracy_vs_always_up.png` | F1 | `Results/compare_*_scoring.csv` | Pooled out-of-sample accuracy of the three feature sets and three models versus the 53.19% always-UP benchmark; bars are 95% date-block bootstrap CIs. |
| 2 | `Plots/phase1/fig2_skill_vs_prior.png` | F2 | `Results/compare_*_scoring.csv` | Brier and log-loss skill against the expanding-window base-rate forecast; all nine run–model pairs are negative and every 95% CI excludes zero. |
| 3 | `Plots/phase1/fig3_calibration.png` | F3 | `Results/compare_baseline_vs_market_only_calibration.csv`, `..._scoring.csv` | Reliability curves by decile of predicted probability for the technical baseline and the market-only model; the near-flat curves and slopes far below 1 show that the probabilities barely order outcomes. |
| 4 | `Plots/phase1/fig4_market_only_deltas.png` | F4 | `Results/compare_baseline_vs_market_only_deltas.csv` | market_only minus technical baseline on identical stock-days: accuracy, Brier and log-loss deltas with 95% date-block CIs. The probability scores improve; the accuracy gain's CI includes zero. |
| 5 | `Plots/phase1/fig5_index_ic.png` | F5 | `Results/market_signal_ic.csv` | Spearman information coefficients with the next-day S&P 500 return in 2023–2026 (left) and across evaluation windows (right); only 3 of 14 features reach \|t\| > 2, and the magnitude collapses in 2021–2022. |
| 6 | `Plots/phase1/fig6_event_study.png` | F5 | `Results/market_signal_event_study.csv` | Mean next-day S&P 500 return by 60-day drawdown quintile and VIX regime in 2023–2026; there is no monotone gradient, and none of the nine out-of-sample bootstrap CIs excludes zero. |
| 7 | `Plots/phase1/fig7_ceiling.png` | F6 | `Results/market_signal_ceiling.csv`, `Results/compare_*_scoring.csv` | Accuracy minus the 53.19% drift for all nine models against the range implied by the measured index IC; every model is 1.9–3.1 pp below the drift, while perfect use of the signal is at most 0.3 pp above it. |
| 8 | `Plots/phase1/fig8_power.png` | F7 | SEs from `Results/compare_*_deltas.csv` | Independent dates needed to detect an accuracy effect at 80% power, using the observed date-block bootstrap standard errors; 831 dates resolve ~1 pp, not the sub-pp effects at issue. |
| 9 | `Plots/phase1/fig9_base_rate_and_labels.png` | F3, F8 | `Results/full28_wf_predictions_baseline.csv`, `Data/cleaned/` | Left: the expanding-window prior per fold and its cross-stock range against the pooled 53.19% up-rate. Right: absolute next-day returns on the same stock-days, 30.5% of them below 0.5%. |

Paths are relative to the repository root
(`C:\Users\Admin\Documents\thesis_sending\PythonProject`). If a file is missing,
it has not been generated yet: run `python src/Phase1_figures.py`, which prints
each path as it writes it. The script reads only `Results/` and `Data/cleaned/`,
so the figures can be rebuilt without rerunning the models, and it never touches
the data directories.

---

## 1. What was measured, and why 50% is the wrong yardstick

**Target.** For each stock and date *t*, the label is 1 if
`Close[t+1] > Close[t]`, else 0. Features on row *t* may use only information
observable at the close of day *t*.

**Protocol.** Walk-forward validation: train on the first 60% of each stock's
history, predict the next 20 days, move the window forward 20 days, retrain.
Three seeds are averaged. No test row is ever in a training set.

**Sample.** 28 stocks × about 800 test rows each = 22,520 stock-days, covering
831 distinct dates.

**The benchmark.** In this sample the price went up the next day **53.19%** of
the time. A model that always says "up" therefore scores 53.19% without looking
at a single feature. Comparing a model to 50% (the coin-flip rate) flatters it by
3.19 pp before any analysis begins.

Two facts make the 53% figure the honest reference:

- The training-window up-share (`p_prior`) moves between **48% and 55%** over
  the walk-forward folds (figure 9a). "50%" is not a fixed property of the data.
- **30.5%** of next-day absolute moves are below 0.5% and **54.4%** below 1%
  (figure 9b). Most labels are noise-dominated coin flips, so accuracy near the
  base rate is expected even from a model with no information.

![Base rate and label noise](Plots/phase1/fig9_base_rate_and_labels.png)

*Figure 9. Left: the expanding-window prior per fold (blue line), its range
across stocks (band) and the pooled 53.19% rate. Right: distribution of absolute
next-day returns on the same stock-days.*

---

## 2. Findings

### F1 — The models are worse than always-predict-UP

**Claim.** All three feature sets and all three models score **below** the
always-UP rule. The gap is statistically significant: every 95% confidence
interval excludes zero.

**Evidence.**

![Accuracy vs always-UP](Plots/phase1/fig1_accuracy_vs_always_up.png)

| Run | LR | RF | XGB | Δ vs 53.19% (LR / RF / XGB) |
| --- | --- | --- | --- | --- |
| technical only | 50.28% | 50.24% | 50.07% | −2.91 / −2.95 / −3.12 pp |
| technical + market | 50.62% | 50.32% | 50.51% | −2.57 / −2.88 / −2.68 pp |
| market only | 51.25% | 51.13% | 50.96% | −1.94 / −2.06 / −2.23 pp |

Source: `Results/compare_*_scoring.csv` (`accuracy`, `d_acc_vs_always_up`,
`d_acc_ci_lo`, `d_acc_ci_hi`).

**Terms used here.** *Always-UP benchmark*, *base rate*, *percentage point*,
*confidence interval*, *date-block bootstrap* — see §5.

**How to defend it.**

1. The benchmark is computed on **exactly the same rows** as the model, not on a
   different sample or a different period.
2. The comparison is **paired**: the same 22,520 stock-days are scored by both
   rules, and the CI comes from resampling whole dates (2,000 draws), which
   respects the fact that all stocks share a market on a given day.
3. The result is not a border case: the closest interval is
   −1.94 pp with a 95% CI of [−3.27, −0.58].
4. The conclusion survives the alternative view too: balanced accuracy is
   49.6–50.5% and MCC is ≈0 for every model, so it is not an artefact of the
   class imbalance (see the `pooled` CSVs).

**What would overturn it.** A rerun on a different, larger sample in which the
model's accuracy CI lies wholly above the sample up-rate. Nothing in this data
does that.

---

### F2 — No model has skill over the expanding-window base rate

**Claim.** Scored against a forecast that simply predicts the training-window
up-share for each fold, **every** run and model has a *negative* Brier skill
score and a *negative* log-loss skill score, with CIs excluding zero.

**Evidence.**

![Skill vs prior](Plots/phase1/fig2_skill_vs_prior.png)

| Run | Brier skill (LR / RF / XGB) | Log-loss skill (LR / RF / XGB) |
| --- | --- | --- |
| technical only | −0.018 / −0.036 / −0.094 | −0.014 / −0.028 / −0.079 |
| technical + market | −0.032 / −0.024 / −0.086 | −0.025 / −0.018 / −0.072 |
| market only | −0.011 / −0.015 / −0.056 | −0.008 / −0.012 / −0.046 |

A skill of −0.02 means the model's squared error is 2% **larger** than the
base-rate forecast's. Source: `compare_*_scoring.csv`
(`brier_skill`, `bss_ci_lo`, `bss_ci_hi`, `logloss_skill`, `lls_ci_lo`,
`lls_ci_hi`).

**Terms used here.** *Proper scoring rule*, *Brier score*, *log-loss*, *skill
score*, *expanding-window prior*, *out-of-sample* — see §5.

**How to defend it.**

1. The reference forecast (`p_prior`) is **strictly out of sample**: for each
   fold it is `mean(y[:tr_end])`, computed only from labels strictly before the
   test window. It was independently recomputed from the raw labels for all
   28 stocks × 3 runs and matches the saved values to 5.6×10⁻¹⁷.
2. Brier and log-loss are *proper* scoring rules: they cannot be improved by
   hedging or by forecasting a constant other than the true rate. Negative skill
   is therefore a genuine statement about information, not about a threshold.
3. This is a **stronger** statement than "chance level": a model that outputs the
   base rate always would score 0. The models score below that.
4. Both proper scores agree, and so do all three models and all three feature
   sets: 18 negative estimates, 18 CIs excluding zero.

**What would overturn it.** A positive Brier skill with a CI excluding zero. None
of the nine run/model combinations produces one.

---

### F3 — The probabilities are badly calibrated and nearly flat

**Claim.** The models' probabilities barely order the outcomes. Plotted against
observed frequencies they are close to a horizontal line near the base rate, the
calibration slope of `y ~ logit(p)` is near 0 rather than 1, and the expected
calibration error is up to 0.12.

**Evidence.**

![Calibration](Plots/phase1/fig3_calibration.png)

| Model | Slope, technical only | Slope, market only | ECE, technical / market only |
| --- | --- | --- | --- |
| LR | −0.03 [−0.14, 0.08] | +0.05 [−0.11, 0.22] | 0.048 / 0.043 |
| RF | +0.03 [−0.05, 0.12] | +0.11 [−0.01, 0.23] | 0.075 / 0.053 |
| XGB | +0.01 [−0.03, 0.06] | +0.07 [0.01, 0.14] | 0.124 / 0.100 |

Source: `compare_baseline_vs_market_only_{calibration,scoring}.csv`. A perfectly
calibrated model has slope 1; a model with no information has slope 0. The only
interval excluding 0 is market-only XGB, at +0.07 — still an order of magnitude
below 1.

**Terms used here.** *Calibration*, *reliability diagram*, *expected calibration
error (ECE)*, *calibration slope*, *logit*, *discrimination vs calibration* — see §5.

**How to defend it.**

1. **Discrimination and calibration are separate.** A model can rank well but be
   badly scaled; here both are weak, which is why the plot is almost flat.
2. The slope is estimated by logistic regression of the label on the model's
   log-odds, with a date-block bootstrap CI — the same dependence correction as
   everywhere else.
3. The flatness is visible without any statistics: in every panel the observed
   up-rate stays within a few points of 53% while the predicted probability
   sweeps its whole range.
4. This also explains F2: a flat, over-confident probability produces a larger
   squared error than the constant base rate, hence negative skill.

**What would overturn it.** A reliability curve close to the diagonal with slope
near 1 and ECE near 0. No model/run pair is close.

---

### F4 — The market block carries a little probability information, but no decision edge

**Claim.** `market_only` significantly improves Brier and log-loss relative to
the technical baseline (all CIs exclude zero), but its accuracy gain is not
robustly significant and remains far below the drift. So the features are not
empty; the information is simply too small to change the decision.

**Evidence.**

![market_only vs baseline](Plots/phase1/fig4_market_only_deltas.png)

| Model | Accuracy Δ (pp) | Brier Δ | Log-loss Δ |
| --- | --- | --- | --- |
| LR | +0.97 [−0.03, +1.90], p=0.056 | −0.0018 [−0.0029, −0.0007] | −0.0040 [−0.0062, −0.0017] |
| RF | +0.89 [−0.14, +1.97], p=0.079 | −0.0051 [−0.0069, −0.0034] | −0.0111 [−0.0149, −0.0075] |
| XGB | +0.89 [−0.13, +1.90], p=0.080 | −0.0093 [−0.0123, −0.0064] | −0.0232 [−0.0296, −0.0166] |

Negative Brier/log-loss Δ means the market-only model is better. McNemar exact
p-values for accuracy: 0.045 / 0.097 / 0.104. Source:
`compare_baseline_vs_market_only_{deltas,mcnemar}.csv`.

**Terms used here.** *Paired comparison*, *McNemar test*, *bootstrap delta*,
*probability information vs decision rule* — see §5.

**How to defend it.**

1. Both statements are true simultaneously: a model can improve a *proper score*
   (which rewards good probabilities everywhere) while failing to improve the
   *0.5-threshold decision* (which only cares about the sign of `p − 0.5`).
2. The accuracy gain is ~1 pp with a CI that touches zero, it does not survive a
   multiplicity correction (point 5), and it leaves the model 1.9–2.2 pp below
   the drift, whereas the ceiling (F6) says no market-based model can be more
   than ~0.3 pp *above* it. The proper-score gain, though real, is ~0.5–2% of the
   Brier level.
3. The same block does better on its own (+0.97/+0.89/+0.89 pp) than combined
   with the 23 technical features (+0.34/+0.07/+0.44 pp), so the technical
   columns dilute it — consistent with a small, fragile signal rather than a
   robust one.
4. All comparisons are on identical stock-dates with date-block resampling, so
   the gain is not a sample-composition artefact.
5. The one nominal sub-0.05 result is LR's McNemar exact p = 0.0447. It does not
   survive multiplicity: Holm-adjusting the three model tests gives 0.134, and
   the LR bootstrap CI still includes zero. It is reported rather than hidden,
   because a single borderline test out of many is exactly what noise produces.

**What would overturn it.** An accuracy delta with a CI wholly above zero that
also lifts the model above the drift — not merely above a weak baseline. Neither
holds.

---

### F5 — The index-level signal is weak and not joint

**Claim.** Some individual market features correlate with the next-day S&P
return, but the correlations are small (|IC| ≤ 0.113), only three reach |t| > 2 in
the OOS window, the magnitude collapses in 2021–2022 (where the volatility
features also flip sign), no feature reaches |t| > 2 in more than two of the four
windows, the joint HAC test is not significant (p = 0.52), and an index-level
walk-forward model cannot beat the index's own up-rate.

**Evidence.**

![Index ICs](Plots/phase1/fig5_index_ic.png)

| Feature (2023–2026) | IC | HAC t |
| --- | --- | --- |
| `vix_high` | +0.113 | +2.03 |
| `mkt_drawdown_60` | −0.092 | −2.58 |
| `mkt_vol_60` | +0.075 | +2.36 |
| `mkt_vol_20` | +0.065 | +1.94 |
| `vix_level` / `vix_log` | +0.056 | +1.66 |

Joint multivariate logit with HAC covariance: **p = 0.516** (OOS), max |t| = 1.66;
full sample p = 0.643, max |t| = 1.74. Index-level walk-forward (5 features
selected on 2018–2022 only): accuracy **55.49%** vs always-UP **56.34%**,
balanced accuracy 49.37%, AUC 0.469, Brier skill −0.005, and the model predicts
UP on **98.2%** of days.

![Event study](Plots/phase1/fig6_event_study.png)

Source: `Results/market_signal_{ic,timing,event_study}.csv`.

**Terms used here.** *Information coefficient (IC)*, *Spearman correlation*,
*Newey–West / HAC standard error*, *t-statistic*, *joint Wald test*, *event
study*, *moving-block bootstrap* — see §5.

**How to defend it.**

1. **Why HAC:** overlapping windows (5-, 20-, 60-day volatility and drawdown)
   make consecutive observations dependent; ordinary t-stats would be too
   optimistic. Newey–West corrects the standard error for that serial
   correlation.
2. **Why the joint test matters:** with 14 features, one or two |t| > 2 is
   expected by chance. The joint test asks whether the vector carries signal at
   all; it does not (p = 0.52).
3. **Why the event study matters:** a monotone gradient (deeper drawdown → higher
   next-day return) would be economically meaningful. The quintiles are not
   monotone, and 17 of 18 bootstrap CIs include zero. The single exclusion is the
   full-sample VIX < 15 bin (0.68 to 10.09 bp); with 18 cells, about one
   exclusion is expected at 95% confidence even under the null.
4. The index model predicting UP 98% of the time is the same phenomenon as F1:
   when the base rate is above 50%, the accuracy-maximising rule degenerates to
   "always up"; the model has learned the drift, not timing.

**What would overturn it.** A joint HAC p < 0.05 with stable signs, or a
monotone event-study gradient with CIs excluding zero in several bins.

---

### F6 — The ceiling: a signal this weak cannot produce an edge over the drift

**Claim.** Combine the best credible index IC (0.113) with the average same-day
stock–market correlation (0.386) and you get an implied stock-level correlation
of 0.036–0.069. At the observed 53.19% up-rate, a classifier that knows the true
probability implied by a signal of that strength would score **53.20–53.52%** —
only **+0.01 to +0.32 pp** above always-UP, and still below any realistic cost
threshold. The market-only models score 50.96–51.25%, i.e. **1.94–2.23 pp below
the drift**. So the ceiling does not claim the +0.9–1.0 pp gain over the technical
baseline is impossible — the baseline is itself ~3 pp below the drift, so a model
can improve over it without approaching the ceiling. It claims something narrower
and sufficient: *even perfect use of this signal cannot turn into a decision edge
over the drift.*

**Evidence.**

![Ceiling](Plots/phase1/fig7_ceiling.png)

*Figure 7. Each point is one model's accuracy minus the 53.19% always-UP rate,
with its 95% date-block CI; rows are models, colours are feature sets. The green
band is the accuracy range implied by the strongest credible index IC, mapped to
stock level with the stock–market correlation. All nine points are far below
zero; the band is just above zero, so even perfect use of the signal barely
clears the drift.*

| IC basis | Stock corr basis | Implied stock corr | Implied AUC | Best accuracy | Gain vs 53.19% |
| --- | --- | --- | --- | --- | --- |
| max \|IC\| with \|t\|≥2 (0.113) | mean corr (0.386) | 0.044 | 0.514 | 53.26% | **+0.07 pp** |
| max \|IC\| with \|t\|≥2 (0.113) | max corr (0.609) | 0.069 | 0.522 | 53.52% | **+0.32 pp** |
| mean \|IC\| with \|t\|≥2 (0.094) | mean corr (0.386) | 0.036 | 0.512 | 53.20% | **+0.01 pp** |
| mean \|IC\| with \|t\|≥2 (0.094) | max corr (0.609) | 0.057 | 0.518 | 53.37% | **+0.18 pp** |

Source: `Results/market_signal_ceiling.csv`; mean correlation computed from
`Data/cleaned_mkt` over the same OOS window.

**Terms used here.** *Ceiling*, *implied correlation*, *AUC*, *Bayes rule*,
*arcsin identity* — see §5.

**How to defend it.**

1. **The chain is multiplicative and conservative**: stock signal ≈ (stock–market
   correlation) × (index signal). Both factors are measured, not assumed.
2. **The accuracy bound is a best case.** The calculation assumes the true
   probability is known exactly and the decision rule is optimal. No model can
   do better than the Bayes rule, so 53.20–53.52% is an *upper bound* on anything
   built from this signal at this base rate.
3. **Why the bound is small despite a non-trivial IC:** at a 53.19% up-rate the
   accuracy-maximising rule says "up" almost always. A correlation of 0.05
   reorders outcomes near the 50% boundary, but there is little mass there. At a
   balanced 50% up-rate the same correlation would be worth ~1–2 pp (the
   `arcsin(r)/π` identity) — which is exactly the intuition that makes the
   sub-1-pp result look surprising until the base rate is taken into account.
4. **Be precise about what it rules out.** The ceiling is a bound on *distance
   above the drift*, not on the difference between two sub-optimal models. The
   technical baseline is 2.91 pp below the drift, so a better model could gain up
   to ~3.2 pp over that baseline while still ending at the ceiling. Comparing
   market_only's +0.97 pp (baseline-relative) with the +0.01 to +0.32 pp
   (drift-relative) band would be a category error; figure 7 plots all nine
   models on the drift-relative axis for exactly this reason. The measured +0.97
   pp is consistent with the ceiling, and that is the point: it moves the model
   from −2.91 to −1.94 pp versus the drift, nowhere near an edge.
5. The conclusion does not depend on the test statistic at all: it is arithmetic
   on measured quantities, so it cannot be fixed by a different model,
   hyperparameter or seed.

**What would overturn it.** A measured stock–market correlation or index IC an
order of magnitude larger, or a near-50% up-rate (which is a different target —
see §3).

---

### F7 — The study is underpowered for sub-pp effects

**Claim.** With 831 independent dates, the study can reliably detect effects
around 1 pp, not the 0.1–0.5 pp range the features could plausibly produce.

**Evidence.**

![Power](Plots/phase1/fig8_power.png)

| Effect to detect | Dates needed (SE 0.36 pp) | Dates needed (SE 0.51 pp) | Available |
| --- | --- | --- | --- |
| 1.0 pp | ~800 | ~1,700 | 831 |
| 0.5 pp | ~3,200 | ~6,800 | 831 |
| 0.3 pp | ~8,900 | ~19,000 | 831 |

The observed date-block bootstrap SE of the accuracy delta is 0.33–0.36 pp
(market vs baseline) and 0.49–0.53 pp (market_only vs baseline), giving 95% CI
half-widths of about 0.7–1.0 pp. The exact 80%-power figures printed by
`Compare_runs.py` are 797 / 3,186 / 8,851 dates for a 1.0 / 0.5 / 0.3 pp effect
(market vs baseline) and 1,710 / 6,841 / 19,003 (market_only vs baseline); the
table above rounds them.

**Terms used here.** *Statistical power*, *minimum detectable effect (MDE)*,
*standard error* — see §5.

**How to defend it.**

1. The SE is **measured** from the actual paired deltas by resampling dates, not
   assumed from a formula.
2. Saying "the result is null" without this table invites the reply "so the
   effect is zero". The correct statement is: *the accuracy difference this
   design can resolve is ~1 pp, and the ceiling says any edge over the drift is
   at most 0.32 pp anyway*.
3. The implication is not "collect more data to find the effect" but "the effect
   is too small to matter even if it exists": 0.32 pp is below realistic
   transaction costs and tiny against the 53% drift.

**What would overturn it.** A longer OOS window (thousands of dates) that reduces
the SE enough to resolve a ~0.3 pp effect — but see F6: there is no such effect
to resolve.

---

### F8 — Label noise and regime coverage bound what any model could show

**Claim.** Half the labels are economically tiny moves, and the OOS window is a
single regime: only **15** dates have VIX > 30, while COVID (2020) and the 2022
drawdown are entirely in the training set. Both facts push the measured
performance *up*, so the negative result is an upper bound on what the features
could deliver.

**Evidence.**

![Base rate and label noise](Plots/phase1/fig9_base_rate_and_labels.png)

- 30.5% of next-day absolute moves are < 0.5%; 54.4% are < 1.0% (n = 22,515
  rows where the next close exists).
- OOS VIX bins: VIX > 30 has n = 15; VIX < 15 has n = 261 (figure 6).
- 28 mega-cap survivors, hindsight-selected universe, no transaction costs.

**Terms used here.** *Label noise*, *regime*, *survivorship bias*, *upper bound*
— see §5.

**How to defend it.**

1. **Label noise is not symmetric with model error**: when the true move is
   ±0.1%, the label is effectively random, and no feature can predict it. That
   caps achievable accuracy independently of the market block.
2. **Regime coverage** is the strongest external-validity caveat. State it
   plainly: the test period is a bull market with a 53% up-rate; the model never
   has to time a crisis. Phase 2 (testing 2020/2022) was designed for this but is
   not run, because Phase 1 failed its gate.
3. **Survivorship and hindsight selection** make the task easier than a live
   universe, so if the effect is absent here it is unlikely to appear in a fair
   universe. This is the correct direction of the bias for a negative claim.

**What would overturn it.** A regime-covering OOS with a materially different
base rate, where the market block shows a larger and stable gain. That is the
Phase 2 experiment described in §3.

---

## 3. The decision gate and why Phase 2 is not being run

The gate was fixed **before** the analysis. Phase 2 was conditional on *any* of
three conditions being positive with a CI excluding zero.

| Gate criterion (plan §8) | Observed | Result |
| --- | --- | --- |
| Brier skill ≤ 0 (no positive skill) for all models | negative for 9/9 run–model pairs, CIs exclude 0 | closed |
| Multivariate index HAC \|t\| < 2 | joint p = 0.516, max \|t\| = 1.66 | closed |
| Event-study CIs include zero | 17/18 cells include 0 (≈ chance) | closed |

Therefore, per the plan: **close the market-features thread and report the
negative result with the ceiling arithmetic and the power table.**

Phase 2 remains documented but not started:

- **2a. Regime-covering OOS** — lower `MIN_TRAIN` or use a rolling window so
  2020 and 2022 are *tested*. Changes only the split; same target and features.
- **2b. Relative-to-SPX target** — predict excess return vs the index. Removes
  the drift that makes always-UP unbeatable, but changes the research question.

Why not run them anyway: the gate exists to prevent post-hoc searching; 2a would
still be powered only to ~0.5 pp at ~1,400 dates, while the ceiling at this
signal strength is ~0.3 pp over the drift; 2b answers a different question and
invalidates comparability with every number in this document.

---

## 4. Defence playbook

Short answers to the questions an examiner, supervisor or reviewer is most likely
to ask. Each answer points to the figure or table that settles it.

**Q1. "Isn't ~50% accuracy just a coin flip? Why is that a finding?"**
Because 50% is the wrong benchmark. The always-UP rule scores 53.19% on the same
rows, and every model is significantly *below* it (F1). A coin flip is not the
relevant comparison: the models do beat 50% (by 0.1–1.3 pp), but 50% is not the
drift. The gap that matters is the 1.9–3.1 pp shortfall against always-UP.

**Q2. "A 2 pp gap could be class imbalance. Did you check balanced accuracy?"**
Yes. Balanced accuracy is 49.6–50.5% and MCC is ≈0 for every model (F1,
`pooled` CSVs). The gap is not a threshold artefact: the models genuinely rank
outcomes poorly.

**Q3. "But accuracy is a crude metric. What about proper scores?"**
That is why Brier and log-loss were added, and why the reference is the
expanding-window prior, not 50% (F2). All nine run–model pairs have negative
skill with CIs excluding zero. The models are worse than a forecast that ignores
features entirely.

**Q4. "Are the probabilities at least calibrated?"**
No. The reliability curves are almost flat, the slope of `y ~ logit(p)` is
−0.03 to +0.11 instead of 1, and ECE reaches 0.12 (F3). One slope CI (market-only
XGB, +0.07 [0.01, 0.14]) excludes zero but is nowhere near calibrated.

**Q5. "market_only improves Brier and log-loss with CIs excluding zero. Isn't
that a win?"**
It is a real but tiny gain in *probability* accuracy, measured against a weak
baseline — not a gain over the drift. Skill against the prior is still negative
(F2), accuracy is still 1.9–2.2 pp below always-UP (F1), and those two facts, not
the baseline comparison, decide the question. A ~1 pp improvement between two
models that are both ~3 pp below the drift is not an edge; the ceiling (F6) says
even perfect use of the signal tops out ~0.3 pp above the drift. On accuracy,
LR's McNemar p = 0.045 is the only nominal sub-0.05 result and it does not
survive a Holm correction across the three models (0.134). The honest phrasing
is: *the market block contains a little information, but not enough to change a
decision.*

**Q6. "You found ICs with |t| > 2 at the index level. Contradiction?"**
No. Three of 14 features reach |t| > 2 individually; with 14 tests, 0.7 are
expected by chance. The joint HAC test — the one that controls for that — gives
p = 0.52, and the event study shows no monotone gradient (F5). Per-feature
significance without a joint test is the classic multiple-comparisons trap.

**Q7. "Could a better model (deep learning, tuning, ensembles) extract the
signal?"**
The ceiling in F6 is information-theoretic: it assumes the true probability is
known. No model can beat the Bayes rule, so no architecture can exceed a
best-case accuracy of 53.52% (+0.32 pp over the drift) at this IC and base rate.
Independently, the bias–variance diagnostics show high bias / low variance, i.e.
the limitation is signal, not capacity.

**Q8. "Isn't the null just low power? Maybe the effect is there but you can't see
it?"**
Partly, and that is stated explicitly (F7). The design resolves ~1 pp, not
0.1 pp. But low power does not rescue the hypothesis: the ceiling says the edge
over the drift cannot exceed 0.32 pp, so a larger sample would converge on a
number too small to trade. The correct conclusion is "bounded, and too small",
not "zero".

**Q9. "Why not test the COVID crash or 2022?"**
Because that is Phase 2, and Phase 2 was conditional on a positive Phase 1
(§3). It is a documented limitation, not an oversight: the OOS window has 15
VIX > 30 dates, and 2020/2022 sit in training (F8). Running it now would be a
post-hoc change of the research question.

**Q10. "Could the market features be leaking future information?"**
The timing contract allows only same-day closes (the target is next-day), and
`Test_market_features.py` enforces it with truncated-build tests and an AST scan
for look-ahead primitives. As a second, independent check, `p_prior` was
recomputed from `y[:tr_end]` for all 28 stocks × 3 runs and matches to 5.6×10⁻¹⁷;
the `market_lag1` control (market features lagged one day) *reduces* the XGB
result by 0.75 pp (CI −1.39 to −0.07), the expected direction if the small gain
came from the same-close overlap rather than from persistent state.

**Q11. "Isn't this p-hacking in reverse — you set a gate that was easy to fail?"**
The gate was written before the runs, has three independent criteria, and all
three failed by wide margins (p = 0.52 instead of < 0.05; 9/9 negative skill
instead of any positive). The ceiling result is independent of the gate and of
any test statistic, so the conclusion does not rest on the threshold choice.

**Q12. "What about transaction costs and tradability?"**
Not modelled, and not needed: an edge has to exist before costs. Here the
pre-cost accuracy is below the drift, and the implied ceiling (0.01–0.32 pp) is
below realistic round-trip costs. Adding costs can only widen the gap.

**Q13. "Why resample dates rather than stock-days?"**
All 28 stocks share the same market on a given date, so stock-days are not
independent. Resampling whole dates preserves that dependence and gives the
correct (wider) standard errors. Resampling individual rows would understate the
CI and overstate significance.

**Q14. "Is the result reproducible?"**
Yes. The three runs were rerun from scratch and reproduce the previous
probabilities exactly (LR/XGB bit-identical, RF to 4×10⁻¹⁶ from parallel
summation); pooled accuracies match to four decimals; `p_prior` was independently
recomputed. `src/Phase1_figures.py` regenerates every figure from the saved CSVs.

---

## 5. Glossary

**Accuracy.** Share of stock-days where the predicted direction equals the
realised direction. Sensitive to the class balance: with 53.19% up-days, a
constant "up" prediction already scores 53.19%.

**Always-UP benchmark (base rate).** The rule that predicts "up" for every row.
Its accuracy equals the share of up-days in the evaluation sample, 53.19% here.
It is the correct naive reference when the classes are imbalanced.

**AUC (area under the ROC curve).** Probability that a randomly chosen up-day
gets a higher predicted probability than a randomly chosen down-day. 0.5 = no
ranking ability, 1.0 = perfect. Threshold-free, so it measures discrimination
only, not calibration.

**Balanced accuracy.** Average of sensitivity (up-day hit rate) and specificity
(down-day hit rate). 0.5 = chance regardless of class balance. Used as a check
that the accuracy gap is not an imbalance artefact.

**Base rate / up-rate.** The unconditional frequency of the positive class. Here
53.19% of OOS stock-days are up-days.

**Basis point (bp).** One hundredth of a percentage point (0.01%). Used for
returns in the event study.

**Bayes rule / Bayes classifier.** The decision rule that uses the true
conditional probability of the label. No model can beat it in expected accuracy;
it is the basis of the ceiling calculation.

**Bootstrap.** Resampling the data with replacement many times to approximate the
sampling distribution of a statistic. Used here to build confidence intervals
without distributional assumptions.

**Brier score.** Mean squared error of a probability forecast:
`mean((y − p)²)`. Lower is better; it is minimised by reporting the true
probability, so it rewards calibration and resolution (informative variation)
together, and it penalises both over-confidence and flat, uninformative output.

**Calibration.** Agreement between predicted probabilities and observed
frequencies: among rows predicted at 60%, about 60% should be up-days.

**Calibration slope.** Coefficient of `logit(p)` in a logistic regression of the
label on the model's log-odds. 1 = calibrated, 0 = no information, > 1 =
under-confident, < 1 = over-confident.

**Ceiling (implied accuracy).** The best accuracy attainable by a signal of a
given strength, computed under an explicit generative model. Used to decide
whether an observed gain is even possible.

**Confidence interval (CI).** A range that covers the true value in 95% of
repeated samples. Reported at 95% throughout.

**Date-block bootstrap.** Bootstrap that resamples whole trading dates rather
than individual stock-days, because all stocks share a market on a given date.
Preserves cross-sectional dependence and gives honest standard errors.

**Decile / quantile bin.** A group containing one tenth of the observations,
ordered by predicted probability. Used to build reliability tables when
predictions are not uniformly distributed.

**Discrimination vs calibration.** Discrimination = ability to rank (AUC);
calibration = agreement between probability and frequency. A model can be good
at one and bad at the other; here both are weak.

**ECE (expected calibration error).** Weighted average gap between predicted
probability and observed frequency across bins: `Σ (n_b/n)·|obs_b − pred_b|`.
0 = perfect.

**Event study.** Comparison of average outcomes across states (e.g. drawdown
quintiles, VIX bins), with a confidence interval, to test for an economically
meaningful gradient rather than a linear correlation.

**Expanding-window prior (`p_prior`).** For each fold, the up-share of the
training labels strictly before the test window. A forecast that uses no
features, and the reference for skill scores.

**Fold.** One train/test step of the walk-forward procedure. Here each test block
is 20 trading days.

**HAC / Newey–West standard error.** A standard-error correction for serial
correlation and heteroskedasticity. Needed because volatility and drawdown
features overlap across days. Without it, t-statistics are typically too
large when the residuals are positively autocorrelated, as return and
volatility series usually are.

**IC (information coefficient).** Correlation between a feature and the future
return. Here Spearman (rank) correlation with the next-day S&P return; |IC| ≈
0.05 is weak and 0.10 is modest, and published factor work often treats
~0.02-0.03 as the lower bound of an economically interesting daily IC.

**Implied correlation.** Product of two measured correlations along a causal
chain: (stock–market same-day correlation) × (market feature IC). Used to map an
index-level signal onto a stock-level one.

**Joint Wald test.** Test that several coefficients are simultaneously zero,
accounting for their covariance. Controls the multiple-comparisons problem that
per-feature t-tests create.

**Label noise.** Randomness in the target that no feature can explain. With half
the next-day moves below 1%, much of the label is effectively noise.

**Log-loss.** Mean negative log-likelihood of the observed labels under the
predicted probabilities. Lower is better; punishes confident wrong predictions
heavily. A *proper* scoring rule.

**Logit.** `log(p / (1−p))`, the log-odds scale on which logistic regression is
linear. The calibration slope is estimated on this scale.

**MCC (Matthews correlation coefficient).** Correlation between predictions and
labels in {−1, 0, +1}; 0 = chance, robust to class imbalance.

**McNemar test.** Paired test of whether two classifiers have the same error rate
on the same rows, using the discordant pairs. Used for model-vs-model accuracy
comparisons.

**MDE (minimum detectable effect).** The smallest effect a study can detect with
a given power and sample size. Here computed at 80% power, two-sided α = 0.05.

**Multiple comparisons.** Testing many hypotheses inflates the chance of a false
positive. With 14 tests at the 5% level, about 0.7 false positives are expected
by chance; hence the joint test.

**Out-of-sample (OOS).** Data not used in training or in any choice that led to
the model. All reported accuracy is OOS.

**p-value.** Probability of a result at least as extreme as observed if the null
hypothesis were true. Reported only where the test is pre-specified; interpreted
alongside CIs, not as a decision rule on its own.

**Percentage point (pp).** Arithmetic difference between two percentages.
53.19% − 50.28% = 2.91 pp.

**Pooled.** Metrics computed over all stock-days together, rather than averaged
per stock. Used for the headline numbers.

**Power.** Probability of detecting a real effect of a given size. 80% is the
conventional target; the dates-needed table follows from it.

**Proper scoring rule.** A metric (Brier, log-loss) that a forecaster minimises
only by reporting the true probability. This makes skill comparisons meaningful:
you cannot game them by hedging.

**Regime.** A market state (bull, bear, high/low volatility). The OOS window is a
single predominantly bullish regime, which limits external validity.

**R².** Share of variance explained. IC ≈ 0.1 corresponds to R² ≈ 1%, i.e. the
signal explains about one percent of next-day return variance.

**Seed.** Random-number seed. Three seeds are averaged to reduce Monte Carlo
variation in RF/XGB; the protocol fixes them.

**Skill score.** Improvement over a reference forecast, expressed as a fraction:
`1 − loss(model) / loss(reference)`. 0 = no better than the reference, negative =
worse.

**Spearman correlation.** Correlation of ranks; robust to outliers and monotone
transformations. Used for ICs.

**Standard error (SE).** Standard deviation of an estimator across repeated
samples. The bootstrap SE is its empirical estimate.

**Survivorship bias.** Bias from analysing only assets that survived to the
present. The 28 mega-caps are survivors, which flatters performance; for a
negative claim this makes the result conservative.

**t-statistic.** Estimate divided by its standard error. |t| > 2 is the rough
5% threshold for a single test; with HAC errors it is the correct version for
serially correlated data.

**Walk-forward validation.** Time-ordered train/test: train on the past, test on
the next block, roll forward, retrain. Prevents look-ahead bias.

**Leakage / look-ahead.** Using information not available at prediction time.
The feature contract forbids `shift(-k)`, `bfill` and centred windows, and the
target uses only `Close[t+1]` versus `Close[t]`.

---

## 6. How to reproduce everything

```bash
# data + features (already built; regenerate only if needed)
python src/Download_index_data.py
python src/Test_market_features.py
python src/Preprocess_data.py --with-market --out-dir Data/cleaned_mkt

# the four runs (deterministic; ~60-70 min total)
python src/Full_run.py --data-dir Data/cleaned      --tag baseline
python src/Full_run.py --data-dir Data/cleaned_mkt  --tag market
python src/Full_run.py --data-dir Data/cleaned_mkt  --tag market_only --feature-set market
python src/Full_run.py --data-dir Data/cleaned_mkt  --tag market_lag1 --market-lag 1

# comparisons, index test, figures
python src/Compare_runs.py --baseline baseline --market market
python src/Compare_runs.py --baseline baseline --market market_only
python src/Compare_runs.py --baseline market   --market market_lag1
python src/Market_signal_test.py
python src/Phase1_figures.py
```

Every figure, its exact output path, the claim it supports and the CSV it is
built from are listed in the **figure index** near the top of this document.
`Plots/` is gitignored in this repository, as it is for every other plot in the
project; the nine figures regenerate from the CSVs with one command
(`python src/Phase1_figures.py`), so the numbers do not depend on the images
being committed.

Verification already performed and recorded: the three main runs reproduce the
previous probabilities (LR/XGB bit-identical, RF max difference 4×10⁻¹⁶), pooled
accuracies match to four decimals, and `p_prior` recomputed from `y[:tr_end]`
matches the saved values for all 28 stocks × 3 runs (max difference 5.6×10⁻¹⁷).

---

## 7. Thesis-ready summary paragraph

> Phase 1 evaluated whether leakage-safe S&P 500 and VIX market-context features
> improve next-day direction prediction. On 22,520 out-of-sample stock-days the
> sample up-rate is 53.19%, and the always-predict-UP rule therefore scores
> 53.19%. All three classifiers score 1.9–3.1 percentage points *below* that
> rule (95% CIs exclude zero), and all nine run–model pairs have negative Brier
> and log-loss skill against an expanding-window base-rate forecast, with
> calibration slopes near zero. Restricting the feature set to the market block
> recovers part of the gap (+0.97/+0.89/+0.89 pp versus the technical baseline)
> and significantly improves Brier and log-loss, but its accuracy gain remains
> below the drift and is not significant. At the index level the signal is weak
> (|IC| ≤ 0.113, joint HAC p = 0.52) and shows no monotone event-study gradient.
> Combining the measured index IC with the mean stock–market correlation (0.386)
> implies a stock-level correlation of 0.036–0.069 and, at a 53.19% up-rate, a
> best possible accuracy of 53.20–53.52% — only 0.01–0.32 pp above the drift.
> The market-only model still scores 1.94–2.23 pp *below* the drift, so its
> ~1 pp improvement over the technical baseline is a move between two weak
> models, not a usable edge. With 831 out-of-sample dates the design can
> resolve effects of about 1 pp, not the sub-percentage-point effect the features
> could plausibly provide. The market-features thread is therefore closed as a
> clean, bounded negative rather than a marginal positive.
