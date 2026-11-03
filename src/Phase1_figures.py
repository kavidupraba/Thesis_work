"""Phase1_figures.py
==================
Regenerate every figure used by ``FINDINGS_PHASE1.md`` from the saved Phase-1
CSVs.  Read-only w.r.t. ``Data/`` and ``Results/``; writes PNGs to
``Plots/phase1/``.

The exact output names are listed in the figure index at the top of
``FINDINGS_PHASE1.md``.  ``Plots/`` is gitignored, so the PNGs are never
committed; this script is the single source of truth for where each one lands.

Figures
-------
fig1_accuracy_vs_always_up.png    accuracy vs the always-UP benchmark (CIs)
fig2_skill_vs_prior.png           Brier / log-loss skill vs the prior (9 pairs)
fig3_calibration.png              reliability curves, baseline vs market_only
fig4_market_only_deltas.png       market_only - baseline: accuracy, Brier, log-loss
fig5_index_ic.png                 index-level ICs with HAC intervals, by window
fig6_event_study.png              drawdown quintiles and VIX bins, next-day S&P
fig7_ceiling.png                  model accuracy vs the drift and the ceiling
fig8_power.png                    dates needed to detect a given accuracy effect
fig9_base_rate_and_labels.png     prior drift and label noise

Usage
-----
    python src/Phase1_figures.py
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\Admin\Documents\thesis_sending\PythonProject"
RESULTS = os.path.join(BASE_DIR, "Results")
CLEAN = os.path.join(BASE_DIR, "Data", "cleaned")
OUT = os.path.join(BASE_DIR, "Plots", "phase1")

MODELS = ["LR", "RF", "XGB"]
RUNS = ["baseline", "market", "market_only"]
RUN_LABEL = {"baseline": "technical only", "market": "technical + market",
             "market_only": "market only"}
C_RUN = {"baseline": "#8c8c8c", "market": "#4c72b0", "market_only": "#dd8452"}
C_MODEL = {"LR": "#2a6f97", "RF": "#61a5c2", "XGB": "#a53860"}
MARKER = {"LR": "o", "RF": "s", "XGB": "^"}

ALWAYS_UP = 0.5319
N_ROWS, N_DATES, N_STOCKS = 22520, 831, 28

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, BASE_DIR)}")


def load_scoring() -> pd.DataFrame:
    frames = []
    for tag in ("baseline_vs_market", "baseline_vs_market_only"):
        frames.append(pd.read_csv(os.path.join(RESULTS, f"compare_{tag}_scoring.csv")))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# fig1  accuracy vs always-UP
# ---------------------------------------------------------------------------
def fig1_accuracy(sc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(RUNS))
    w = 0.26
    for j, m in enumerate(MODELS):
        vals, lo, hi = [], [], []
        for r in RUNS:
            row = sc[(sc["run"] == r) & (sc["Model"] == m)].iloc[0]
            vals.append(row["accuracy"] * 100)
            lo.append(row["d_acc_ci_lo"] * 100)
            hi.append(row["d_acc_ci_hi"] * 100)
        vals, lo, hi = np.array(vals), np.array(lo), np.array(hi)
        yerr = np.vstack([vals - (ALWAYS_UP * 100 + lo), (ALWAYS_UP * 100 + hi) - vals])
        ax.bar(x + (j - 1) * w, vals, w, yerr=yerr, capsize=3,
               color=C_MODEL[m], alpha=0.9, label=f"{m}",
               error_kw=dict(lw=1.1, capthick=1.1))
        for xi, v in zip(x + (j - 1) * w, vals):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5,
                        color=C_MODEL[m], fontweight="bold")

    ax.axhline(ALWAYS_UP * 100, color="#b00020", lw=1.6, ls="--",
               label=f"always-UP = {ALWAYS_UP * 100:.2f}%")
    ax.axhline(50.0, color="black", lw=0.9, ls=":", label="coin flip = 50%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{RUN_LABEL[r]}\n(n={N_ROWS:,})" for r in RUNS])
    ax.set_ylabel("pooled accuracy (%)")
    ax.set_ylim(48.6, 54.2)
    ax.set_title("All three feature sets and all three models score below the "
                 "always-predict-UP rule")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
    ax.text(0.995, 0.03, "error bars: 95% date-block bootstrap CI of the "
            "difference vs always-UP", transform=ax.transAxes, ha="right",
            fontsize=7.5, color="#444444")
    save(fig, "fig1_accuracy_vs_always_up.png")


# ---------------------------------------------------------------------------
# fig2  skill vs expanding prior
# ---------------------------------------------------------------------------
def fig2_skill(sc: pd.DataFrame) -> None:
    rows = []
    for r in RUNS:
        for m in MODELS:
            row = sc[(sc["run"] == r) & (sc["Model"] == m)].iloc[0]
            rows.append(row)
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(d))[::-1]
    for yi, (_, row) in zip(y, d.iterrows()):
        col = C_RUN[row["run"]]
        bss, b_lo, b_hi = row["brier_skill"], row["bss_ci_lo"], row["bss_ci_hi"]
        lls, l_lo, l_hi = row["logloss_skill"], row["lls_ci_lo"], row["lls_ci_hi"]
        ax.hlines(yi, b_lo, b_hi, color=col, lw=2.2, alpha=0.55)
        ax.hlines(yi + 0.18, l_lo, l_hi, color=col, lw=2.2, alpha=0.55, ls=":")
        ax.plot(bss, yi, marker="o", color=col, ms=7, mec="white", mew=1.0)
        ax.plot(lls, yi + 0.18, marker="D", color=col, ms=5.5, mec="white", mew=0.8)

    ax.axvline(0.0, color="#b00020", lw=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{RUN_LABEL[r['run']]:<18s} {r['Model']}"
                        for _, r in d.iterrows()], fontsize=8.5)
    ax.set_xlabel("skill score vs the expanding-window prior  "
                  "(> 0 = better than the base-rate forecast)")
    ax.set_xlim(-0.115, 0.02)
    ax.set_title("Every run and model has negative Brier and log-loss skill")
    ax.plot([], [], "o", color="#555555", label="Brier skill (BSS)")
    ax.plot([], [], "D", color="#555555", label="log-loss skill (LLS)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.text(0.01, 0.97, "bars = 95% date-block CI", transform=ax.transAxes,
            fontsize=8, color="#444444", va="top")
    ax.set_ylim(-0.7, len(d) - 0.25)
    save(fig, "fig2_skill_vs_prior.png")


# ---------------------------------------------------------------------------
# fig3  calibration
# ---------------------------------------------------------------------------
def fig3_calibration() -> None:
    cal = pd.read_csv(os.path.join(RESULTS, "compare_baseline_vs_market_only_calibration.csv"))
    sc = pd.read_csv(os.path.join(RESULTS, "compare_baseline_vs_market_only_scoring.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        ax.plot([0, 1], [0, 1], color="black", lw=1.0, ls="--", label="perfect calibration")
        ax.axhline(ALWAYS_UP, color="#b00020", lw=0.9, ls=":", label="always-UP rate")
        for r in ("baseline", "market_only"):
            sub = cal[(cal["run"] == r) & (cal["Model"] == m)].sort_values("bin")
            ax.plot(sub["p_mean"], sub["obs_rate"], marker="o", ms=4.5,
                    color=C_RUN[r], lw=1.8, label=f"{RUN_LABEL[r]}")
        row_b = sc[(sc["run"] == "baseline") & (sc["Model"] == m)].iloc[0]
        row_m = sc[(sc["run"] == "market_only") & (sc["Model"] == m)].iloc[0]
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.3, 0.72)
        ax.set_xlabel("predicted P(up)")
        ax.set_title(f"{m}")
        ax.text(0.03, 0.96,
                f"slope: base {row_b['calib_slope']:+.2f}, "
                f"mkt {row_m['calib_slope']:+.2f}\n"
                f"ECE:  base {row_b['ece']:.3f}, "
                f"mkt {row_m['ece']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))
    axes[0].set_ylabel("observed up-rate")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Probabilities are compressed and nearly flat: the model output "
                 "barely orders outcomes", fontweight="bold", y=1.04)
    fig.text(0.5, -0.04, "Points = deciles of predicted probability (quantile bins, "
             "n=2,252 each). Slope is from y ~ logit(p); 1 = calibrated, 0 = no information.",
             ha="center", fontsize=8, color="#444444")
    save(fig, "fig3_calibration.png")


# ---------------------------------------------------------------------------
# fig4  market_only vs baseline deltas
# ---------------------------------------------------------------------------
def fig4_deltas() -> None:
    d = pd.read_csv(os.path.join(RESULTS, "compare_baseline_vs_market_only_deltas.csv"))

    def get(metric):
        s = d[d["Metric"] == metric].set_index("Model").loc[MODELS]
        return s

    acc = get("Accuracy")
    bri = get("Brier")
    lls = get("LogLoss")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    specs = [
        (axes[0], acc, 100.0, "accuracy delta (pp)", "{:+.2f} pp"),
        (axes[1], bri, 1000.0, "Brier delta x1000 (lower = better)", "{:+.2f}"),
        (axes[2], lls, 1000.0, "log-loss delta x1000 (lower = better)", "{:+.2f}"),
    ]
    for ax, s, scale, xlabel, fmt in specs:
        y = np.arange(len(MODELS))[::-1]
        vals = s["Delta"].to_numpy() * scale
        lo = s["ci_lo"].to_numpy() * scale
        hi = s["ci_hi"].to_numpy() * scale
        for yi, v, l, h, m, p in zip(y, vals, lo, hi, MODELS, s["p_boot"]):
            ax.hlines(yi, l, h, color=C_MODEL[m], lw=2.4, alpha=0.6)
            ax.plot(v, yi, marker=MARKER[m], color=C_MODEL[m], ms=8, mec="white", mew=1.0)
            ax.annotate(fmt.format(v), (v, yi), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=8,
                        color=C_MODEL[m], fontweight="bold")
        ax.axvline(0.0, color="black", lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(MODELS)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.7, len(MODELS) - 0.15)
        ax.margins(x=0.24)
    axes[0].set_title("Accuracy: CIs cross zero", color="#8a4b00")
    axes[1].set_title("Brier: CIs exclude zero", color="#1d5b2f")
    axes[2].set_title("Log-loss: CIs exclude zero", color="#1d5b2f")
    fig.suptitle("Adding the market block helps probability scores a little, "
                 "but not the decision rule", fontweight="bold", y=1.03)
    fig.text(0.5, -0.03, "market_only minus baseline on identical stock-dates; "
             "bars = 95% date-block bootstrap CI (2,000 draws over 831 dates).",
             ha="center", fontsize=8, color="#444444")
    save(fig, "fig4_market_only_deltas.png")


# ---------------------------------------------------------------------------
# fig5  index ICs
# ---------------------------------------------------------------------------
def fig5_ic() -> None:
    ic = pd.read_csv(os.path.join(RESULTS, "market_signal_ic.csv"))
    oos = ic[ic["window"] == "2023-2026 OOS"].sort_values("IC").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    y = np.arange(len(oos))
    colors = ["#2a6f97" if v > 0 else "#a53860" for v in oos["IC"]]
    ax.barh(y, oos["IC"], color=colors, alpha=0.85, height=0.68)
    ax.errorbar(oos["IC"], y, xerr=1.96 * oos["se_hac"], fmt="none",
                ecolor="#333333", elinewidth=1.1, capsize=2.5)
    ax.axvline(0, color="black", lw=1.0)
    for v in (0.05, -0.05):
        ax.axvline(v, color="#b00020", lw=0.8, ls=":")
    ax.set_yticks(y)
    labels = [f"{r.feature}  (t={r.t_hac:+.2f})" for r in oos.itertuples()]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Spearman IC with next-day S&P 500 return")
    ax.set_title("2023-2026: only 3 of 14 features reach |t| > 2")
    ax.set_xlim(-0.16, 0.16)
    ax.text(0.985, 0.02, "bars = 95% Newey-West (HAC) CI\ndotted lines = |IC| 0.05",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#444444")

    ax = axes[1]
    top = ["vix_high", "mkt_drawdown_60", "mkt_vol_60", "vix_level",
           "mkt_above_sma50", "mkt_ret_5"]
    windows = ["2018-2019", "2020", "2021-2022", "2023-2026 OOS"]
    wcols = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    width = 0.2
    for j, (win, col) in enumerate(zip(windows, wcols)):
        sub = ic[ic["window"] == win].set_index("feature").reindex(top)
        ax.barh(np.arange(len(top)) + (j - 1.5) * width, sub["IC"],
                width, color=col, label=win, alpha=0.95)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top, fontsize=8.5)
    ax.set_xlabel("IC, by evaluation window")
    ax.set_title("Sign is persistent but magnitude is small")
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.set_xlim(-0.16, 0.16)

    fig.suptitle("At the index level the signal is weak and not robust across windows",
                 fontweight="bold", y=1.02)
    save(fig, "fig5_index_ic.png")


# ---------------------------------------------------------------------------
# fig6  event study
# ---------------------------------------------------------------------------
def fig6_event_study() -> None:
    ev = pd.read_csv(os.path.join(RESULTS, "market_signal_event_study.csv"))
    oos = ev[ev["window"] == "2023-2026 OOS"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, kind, title in [
        (axes[0], "drawdown_quintile",
         "60-day drawdown quintile  (Q1 = deepest, Q5 = at/near high)"),
        (axes[1], "vix_bin", "VIX regime"),
    ]:
        sub = oos[oos["group_type"] == kind].reset_index(drop=True)
        x = np.arange(len(sub))
        ax.axhline(0, color="black", lw=1.0)
        ax.errorbar(x, sub["mean_ret_bp"],
                    yerr=[sub["mean_ret_bp"] - sub["ci_lo_bp"],
                          sub["ci_hi_bp"] - sub["mean_ret_bp"]],
                    fmt="o", color="#2a6f97", ms=7, capsize=4, lw=1.6)
        for xi, row in zip(x, sub.itertuples()):
            ax.annotate(f"n={row.n}", (xi, row.ci_lo_bp), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7.5, color="#555555")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("mean next-day S&P return (bp)")
        ax.margins(y=0.25)
    axes[1].set_ylim(-35, 130)
    fig.suptitle("Event study: no monotone gradient, and every CI that excludes "
                 "zero is a small-sample cell", fontweight="bold", y=1.03)
    fig.text(0.5, -0.03, "2023-2026 OOS; 95% moving-block bootstrap CI (2,000 draws, "
             "5-day blocks). VIX>30 has only 15 OOS dates.", ha="center",
             fontsize=8, color="#444444")
    save(fig, "fig6_event_study.png")


# ---------------------------------------------------------------------------
# fig7  ceiling
# ---------------------------------------------------------------------------
def fig7_ceiling() -> None:
    sc = load_scoring()
    ceil = pd.read_csv(os.path.join(RESULTS, "market_signal_ceiling.csv"))

    lo = float(ceil["gain_vs_always_up_pp"].min())
    hi = float(ceil["gain_vs_always_up_pp"].max())

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.axvspan(lo, hi, color="#2e7d32", alpha=0.22)
    ax.text(0.06, 2.72, "information ceiling\n+0.01 to +0.32 pp over drift",
            fontsize=8.5, color="#1d5b2f", va="center", ha="left")

    ybase = {m: len(MODELS) - 1 - i for i, m in enumerate(MODELS)}
    dy = {"baseline": -0.24, "market": 0.0, "market_only": 0.24}
    for m in MODELS:
        for r in RUNS:
            row = sc[(sc["run"] == r) & (sc["Model"] == m)].iloc[0]
            v = row["d_acc_vs_always_up"] * 100
            l = row["d_acc_ci_lo"] * 100
            h = row["d_acc_ci_hi"] * 100
            yy = ybase[m] + dy[r]
            ax.hlines(yy, l, h, color=C_RUN[r], lw=2.0, alpha=0.55)
            ax.plot(v, yy, MARKER[m], color=C_RUN[r], ms=7, mec="white", mew=0.9)
            if r == "market_only":
                ax.annotate(f"{v:+.2f}", (v, yy), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8,
                            color=C_RUN[r], fontweight="bold")

    ax.axvline(0.0, color="#b00020", lw=1.4, ls="--")
    ax.set_yticks([ybase[m] for m in MODELS])
    ax.set_yticklabels(MODELS)
    ax.set_ylim(-0.75, 2.95)
    ax.set_xlim(-5.4, 1.15)
    ax.set_xlabel("accuracy minus the always-UP rate on the same 22,520 stock-days (pp)")
    ax.set_title("Every model is 1.9–3.1 pp below the drift; even perfect use of the "
                 "market signal\nis at most 0.3 pp above it", fontsize=10.5)

    handles = [plt.Line2D([], [], color=C_RUN[r], lw=3, label=RUN_LABEL[r]) for r in RUNS]
    handles += [plt.Line2D([], [], color="#b00020", lw=1.4, ls="--", label="always-UP (drift)"),
                plt.Rectangle((0, 0), 1, 1, color="#2e7d32", alpha=0.22,
                              label="implied ceiling (4 cases)")]
    ax.legend(handles=handles, ncol=5, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.20))
    fig.text(0.5, -0.14, "x < 0: worse than always-UP. Green band = best achievable "
             "accuracy implied by the measured index IC (0.036–0.069 implied stock "
             "correlation)\nat the observed 53.19% up-rate. Bars = 95% date-block "
             "bootstrap CI; row = model; colour = feature set.", ha="center",
             fontsize=7.5, color="#444444")
    save(fig, "fig7_ceiling.png")


# ---------------------------------------------------------------------------
# fig8  power
# ---------------------------------------------------------------------------
def fig8_power() -> None:
    eff = np.array([0.2, 0.3, 0.5, 1.0, 2.0])
    ses = {"market vs baseline (SE 0.36 pp)": 0.36, "market_only vs baseline (SE 0.51 pp)": 0.51}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for (label, se), col in zip(ses.items(), ["#4c72b0", "#dd8452"]):
        n_req = N_DATES * (se / (eff / 2.80)) ** 2
        ax.plot(eff, n_req, marker="o", color=col, lw=1.8, label=label)
        for e, n in zip(eff, n_req):
            if e in (0.3, 1.0):
                ax.annotate(f"{n:,.0f}", (e, n), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color=col)
    ax.axhline(N_DATES, color="#b00020", lw=1.3, ls="--",
               label=f"available OOS dates = {N_DATES}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(eff)
    ax.set_xticklabels([f"{e:g}" for e in eff])
    ax.set_xlabel("accuracy effect to detect (percentage points)")
    ax.set_ylabel("independent dates needed (80% power)")
    ax.set_title("With 831 dates the study can only resolve effects near 1 pp")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.text(0.5, -0.05, "Two-sided alpha = 0.05, z = 2.80; SE from the date-block "
             "bootstrap of the observed accuracy delta.", ha="center",
             fontsize=8, color="#444444")
    save(fig, "fig8_power.png")


# ---------------------------------------------------------------------------
# fig9  prior drift + label noise
# ---------------------------------------------------------------------------
def fig9_base_rate_and_labels() -> None:
    pred = pd.read_csv(os.path.join(RESULTS, "full28_wf_predictions_baseline.csv"),
                       parse_dates=["Date"])
    g = pred.groupby("fold")["p_prior"].agg(["mean", "min", "max"])

    # forward return magnitude on exactly the same stock-days
    frames = []
    for path in sorted(glob.glob(os.path.join(CLEAN, "*_cleaned.csv"))):
        tk = os.path.basename(path).replace("_cleaned.csv", "")
        d = pd.read_csv(path, index_col="Date", parse_dates=True)
        fwd = d["Close"].pct_change().shift(-1)
        frames.append(pd.DataFrame({"Stock": tk, "Date": d.index, "fwd": fwd}).dropna())
    ret = pd.concat(frames, ignore_index=True)
    ret = ret.merge(pred[["Stock", "Date"]], on=["Stock", "Date"], how="inner")
    absret = ret["fwd"].abs().to_numpy() * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    ax.fill_between(g.index, g["min"] * 100, g["max"] * 100,
                    color="#9ecae1", alpha=0.55, label="range across stocks")
    ax.plot(g.index, g["mean"] * 100, color="#08519c", lw=1.8, label="mean prior")
    ax.axhline(ALWAYS_UP * 100, color="#b00020", lw=1.2, ls="--",
               label=f"pooled OOS up-rate = {ALWAYS_UP * 100:.2f}%")
    ax.set_xlabel("walk-forward fold (20 trading days each)")
    ax.set_ylabel("training up-share (%)")
    ax.set_title("The prior itself ranges ~48-55%, so 50% is the wrong benchmark")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.hist(absret, bins=np.linspace(0, 6, 61), color="#4c72b0", alpha=0.85)
    for cut, col, frac in ((0.5, "#dd8452", 0.88), (1.0, "#b00020", 0.70)):
        share = 100.0 * float(np.mean(absret < cut))
        ax.axvline(cut, color=col, ls="--", lw=1.4)
        ax.annotate(f"|move| < {cut:g}%: {share:.1f}%",
                    (cut, ax.get_ylim()[1] * frac), textcoords="offset points",
                    xytext=(7, 0), fontsize=8.5, color=col, fontweight="bold")
    ax.set_xlabel("absolute next-day return (%)")
    ax.set_ylabel("stock-days")
    ax.set_title("Half of the labels are economically tiny moves")
    fig.suptitle("Why the base rate, not 50%, is the honest benchmark",
                 fontweight="bold", y=1.03)
    save(fig, "fig9_base_rate_and_labels.png")
    print(f"  label noise: {np.mean(absret < 0.5) * 100:.1f}% < 0.5%, "
          f"{np.mean(absret < 1.0) * 100:.1f}% < 1.0%, n={len(absret):,}")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print("Phase 1 figures ->", os.path.relpath(OUT, BASE_DIR))
    sc = load_scoring()
    fig1_accuracy(sc)
    fig2_skill(sc)
    fig3_calibration()
    fig4_deltas()
    fig5_ic()
    fig6_event_study()
    fig7_ceiling()
    fig8_power()
    fig9_base_rate_and_labels()
    print("done")


if __name__ == "__main__":
    main()
