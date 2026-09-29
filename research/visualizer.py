"""Paper figures comparing real and synthetic patients.

    python research/visualizer.py                 # real vs latest MedTwin output
    python research/visualizer.py --benchmark     # also the 5-seed model comparison

Writes PNGs (300 dpi) to figures/:
  fig_continuous.png          distributions of the numeric vitals
  fig_categorical.png         category proportions, "not measured" included
  fig_missingness_by_site.png % not measured per column and hospital
  fig_correlation.png         correlation matrices and their difference
  fig_benchmark.png           mean ± SD per model over seeds (--benchmark)
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # files only; no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.evaluation import align_real_to_report, normalize_report
from core.ingestion import NOT_MEASURED
from core.medtwin import REPORT_COLS

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_PATH = os.path.join(BASE_DIR, "data", "heart_disease_uci.csv")
SYNTH_PATH = os.path.join(BASE_DIR, "data", "synthetic", "grounded_synthetic_data_latest.csv")
RUNS_PATH = os.path.join(BASE_DIR, "data", "synthetic", "multi_seed", "all_runs.csv")
OUT_DIR = os.path.join(BASE_DIR, "figures")

# Validated palette (colour-blind safe; see the dataviz reference palette)
REAL, SYNTH = "#2a78d6", "#eb6834"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])

CONTINUOUS = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL = ["gender", "chest_pain", "fbs", "restecg", "exang", "slope", "thal", "ca", "num", "origin_study"]
MISSABLE = ["trestbps", "chol", "thalch", "oldpeak", "ca", "fbs", "exang", "slope", "thal"]
LABELS = {
    "age": "Age (years)", "trestbps": "Resting BP (mmHg)", "chol": "Cholesterol (mg/dl)",
    "thalch": "Max heart rate (bpm)", "oldpeak": "ST depression (oldpeak)",
    "gender": "Sex", "chest_pain": "Chest pain type", "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG", "exang": "Exercise angina", "slope": "ST slope",
    "thal": "Thalassemia", "ca": "Major vessels (ca)", "num": "Disease severity (num)",
    "origin_study": "Hospital site",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(synth_path):
    raw = pd.read_csv(REAL_PATH)
    real = align_real_to_report(raw, REPORT_COLS)  # "not measured" zeros become blanks
    synth = normalize_report(pd.read_csv(synth_path)[REPORT_COLS])
    return real, synth


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 {os.path.relpath(path, BASE_DIR)}")


def kde(values, grid):
    values = values.astype(float)
    bw = 1.06 * values.std() * len(values) ** (-1 / 5) or 1.0
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


def fig_continuous(real, synth):
    fig, axes = plt.subplots(1, len(CONTINUOUS), figsize=(15, 3.2))
    for ax, col in zip(axes, CONTINUOUS):
        r, s = real[col].dropna().values, synth[col].dropna().values
        grid = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 300)
        for vals, color, name in [(r, REAL, "Real"), (s, SYNTH, "Synthetic")]:
            y = kde(vals, grid)
            ax.plot(grid, y, color=color, label=name)
            ax.fill_between(grid, y, color=color, alpha=0.15, linewidth=0)
        ax.set_title(LABELS[col])
        ax.set_yticks([])
        ax.set_xlabel(f"n = {len(r)} real / {len(s)} synthetic", fontsize=7)
    shared_legend(fig, axes.flat[0])
    fig.suptitle("Numeric vitals: real vs synthetic (measured values only)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig_continuous.png")


def shared_legend(fig, ax):
    """One legend for the whole figure, top right, clear of every panel."""
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=9)


def proportions(series):
    def label(v):
        if pd.isna(v):
            return NOT_MEASURED
        if isinstance(v, float) and v.is_integer():
            return str(int(v))  # vessel counts read as 0.0, 1.0 when blanks exist
        return str(v)

    return series.map(label).value_counts(normalize=True)


def fig_categorical(real, synth):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4))
    for ax, col in zip(axes.flat, CATEGORICAL):
        r, s = proportions(real[col]), proportions(synth[col])
        cats = sorted(set(r.index) | set(s.index), key=lambda c: (c == NOT_MEASURED, c))
        x = np.arange(len(cats))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, r.reindex(cats, fill_value=0), w, color=REAL, label="Real")
        ax.bar(x + w / 2 + 0.01, s.reindex(cats, fill_value=0), w, color=SYNTH, label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" ", "\n", 1) for c in cats], fontsize=7)
        ax.set_title(LABELS[col])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
    shared_legend(fig, axes.flat[0])
    fig.suptitle("Categorical columns: share of patients, including \"not measured\"", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig_categorical.png")


def fig_missingness(real, synth):
    sites = sorted(real["origin_study"].dropna().unique())
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)
    for ax, col in zip(axes.flat, MISSABLE):
        r = real.groupby("origin_study")[col].apply(lambda c: c.isna().mean()).reindex(sites)
        s = synth.groupby("origin_study")[col].apply(lambda c: c.isna().mean()).reindex(sites)
        x = np.arange(len(sites))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, r, w, color=REAL, label="Real")
        ax.bar(x + w / 2 + 0.01, s, w, color=SYNTH, label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels([site.replace(" ", "\n") for site in sites], fontsize=8)
        ax.set_title(LABELS[col])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("% of patients with the value not measured, by hospital site", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig_missingness_by_site.png")


def fig_correlation(real, synth):
    cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "num"]
    rc = real[cols].astype(float).corr()
    sc = synth[cols].astype(float).corr()
    # Constrained layout reserves room for both colour bars and their labels.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    # The difference panel gets its own, tighter scale: on the -1..1 scale of
    # the correlations themselves, gaps of 0.1-0.2 would all look neutral.
    diff = sc - rc
    diff_max = max(0.1, float(np.ceil(np.abs(diff.values).max() * 10) / 10))
    panels = [(rc, "Real", 1.0), (sc, "Synthetic", 1.0), (diff, "Synthetic − real", diff_max)]
    for ax, (m, title, vmax) in zip(axes, panels):
        im = ax.imshow(m.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        ax.grid(False)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
        ax.set_title(title)
        if ax is axes[1]:
            fig.colorbar(im, ax=axes[:2], shrink=0.85, label="Pearson correlation")
    fig.colorbar(im, ax=axes[2], shrink=0.85, label="Difference")
    fig.suptitle("Correlations between numeric columns (pairwise, measured values)", x=0.01, ha="left")
    save(fig, "fig_correlation.png")


def fig_benchmark(runs_path):
    runs = pd.read_csv(runs_path)
    models = ["MedTwin WGAN-GP"] + sorted(m for m in runs.model.unique() if m != "MedTwin WGAN-GP")
    metrics = [
        ("fidelity_pct", "Fidelity (%) ↑", None),
        ("auc_ratio", "Utility: TSTR / TRTR AUC → 1", 1.0),
        ("discriminator_pct", "Real-vs-synthetic classifier (%) → 50", 50),
        ("mia_auc", "Membership-inference AUC → 0.5", 0.5),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.8))
    for ax, (col, title, target) in zip(axes, metrics):
        stats = runs.groupby("model")[col].agg(["mean", "std"]).reindex(models)
        x = np.arange(len(models))
        ax.bar(x, stats["mean"], 0.62, yerr=stats["std"], color=MODEL_COLORS[: len(models)],
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
        for xi, (mean, std) in zip(x, stats.values):
            ax.text(xi, mean + (std if not np.isnan(std) else 0), f"{mean:.3g}",
                    ha="center", va="bottom", fontsize=7, color=INK)
        if target is not None:
            ax.axhline(target, color=INK_2, linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n", 1) for m in models], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    n = runs.groupby("model").seed.nunique().min()
    fig.suptitle(f"Model comparison, mean ± SD over {n} seeds (dashed line = ideal)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig_benchmark.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=SYNTH_PATH, help="synthetic report CSV")
    parser.add_argument("--benchmark", action="store_true", help="also plot the multi-seed comparison")
    args = parser.parse_args()

    real, synth = load(args.synthetic)
    fig_continuous(real, synth)
    fig_categorical(real, synth)
    fig_missingness(real, synth)
    fig_correlation(real, synth)
    if args.benchmark:
        fig_benchmark(RUNS_PATH)
