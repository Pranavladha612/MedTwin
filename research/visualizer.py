"""Paper figures comparing real and synthetic patients.

    python research/visualizer.py                 # real vs latest MedTwin output
    python research/visualizer.py --benchmark     # also the 5-seed model comparison
    add --config configs/<dataset>.yaml for another dataset

Writes PNGs (300 dpi) to figures/<dataset name>/:
  fig_continuous.png          distributions of the numeric vitals
  fig_categorical.png         category proportions, "not measured" included
  fig_missingness_by_site.png % not measured per column and hospital (if conditioned)
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
from core import config
from core.evaluation import align_real_to_report, normalize_report
from core.ingestion import NOT_MEASURED
from research.multi_seed import LEGACY_NAMES

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = config.ROOT

# Validated palette (colour-blind safe; see the dataviz reference palette)
REAL, SYNTH = "#2a78d6", "#eb6834"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])


def out_dir():
    """figures/<dataset name>/"""
    return os.path.join(BASE_DIR, "figures", config.active().name)


def label(col):
    return config.active().labels.get(col, col)


def column_groups():
    """Which report columns go in which figure, from the settings file.
    Whole-number scores and counts (e.g. vessel count, severity) are drawn as
    bars with the categories rather than as smooth curves."""
    cfg = config.active()
    numeric = [cfg.report_name(c) for c in cfg.numeric]
    discrete = [cfg.report_name(c) for c in cfg.discrete_numeric]
    condition = [cfg.report_name(cfg.condition_on)] if cfg.condition_on else []
    return {
        "continuous": [c for c in numeric if c not in discrete],
        "categorical": [cfg.report_name(c) for c in cfg.categorical] + discrete + condition,
        "correlation": numeric,
    }


def panel_grid(n, ncols, panel_w, panel_h, **kwargs):
    """A grid of n panels, at most ncols wide; spare cells are hidden."""
    rows, cols = int(np.ceil(n / ncols)), min(n, ncols)
    fig, axes = plt.subplots(rows, cols, figsize=(panel_w * cols, panel_h * rows), squeeze=False, **kwargs)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat[:n]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(synth_path):
    cfg = config.active()
    raw = pd.read_csv(cfg.data_file)
    real = align_real_to_report(raw, cfg.report_columns)  # "not measured" zeros become blanks
    synth = normalize_report(pd.read_csv(synth_path)[cfg.report_columns])
    return real, synth


def save(fig, name):
    os.makedirs(out_dir(), exist_ok=True)
    path = os.path.join(out_dir(), name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 {os.path.relpath(path, BASE_DIR)}")


def kde(values, grid):
    values = values.astype(float)
    bw = 1.06 * values.std() * len(values) ** (-1 / 5) or 1.0
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


def fig_continuous(real, synth):
    cols = column_groups()["continuous"]
    fig, axes = panel_grid(len(cols), 5, 3, 3.2)
    for ax, col in zip(axes, cols):
        r, s = real[col].dropna().values, synth[col].dropna().values
        grid = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 300)
        for vals, color, name in [(r, REAL, "Real"), (s, SYNTH, "Synthetic")]:
            y = kde(vals, grid)
            ax.plot(grid, y, color=color, label=name)
            ax.fill_between(grid, y, color=color, alpha=0.15, linewidth=0)
        ax.set_title(label(col))
        ax.set_yticks([])
        ax.set_xlabel(f"n = {len(r)} real / {len(s)} synthetic", fontsize=7)
    shared_legend(fig, axes[0])
    fig.suptitle("Numeric columns: real vs synthetic (measured values only)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig_continuous.png")


def shared_legend(fig, ax):
    """One legend for the whole figure, centred below the panels, where it
    can't collide with the title however narrow the figure is."""
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)


def category_order(value):
    """Numbers in numeric order (2 before 10), then text, "not measured" last."""
    try:
        return (0, float(value), "")
    except ValueError:
        return (1 if value != NOT_MEASURED else 2, 0.0, value)


def proportions(series):
    def label(v):
        if pd.isna(v):
            return NOT_MEASURED
        if isinstance(v, float) and v.is_integer():
            return str(int(v))  # vessel counts read as 0.0, 1.0 when blanks exist
        return str(v)

    return series.map(label).value_counts(normalize=True)


def fig_categorical(real, synth):
    cols = column_groups()["categorical"]
    if not cols:
        return
    fig, axes = panel_grid(len(cols), 5, 3.2, 3.2)
    for ax, col in zip(axes, cols):
        r, s = proportions(real[col]), proportions(synth[col])
        cats = sorted(set(r.index) | set(s.index), key=category_order)
        x = np.arange(len(cats))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, r.reindex(cats, fill_value=0), w, color=REAL, label="Real")
        ax.bar(x + w / 2 + 0.01, s.reindex(cats, fill_value=0), w, color=SYNTH, label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" ", "\n", 1) for c in cats], fontsize=7)
        ax.set_title(label(col))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
    shared_legend(fig, axes[0])
    fig.suptitle("Categories and counts: share of patients, including \"not measured\"", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig_categorical.png")


def fig_missingness(real, synth):
    """Only drawn for data with a condition column (e.g. hospital site)."""
    cfg = config.active()
    if not cfg.condition_on:
        return
    site = cfg.report_name(cfg.condition_on)
    # Columns not measured for at least 1% of real patients: numeric first,
    # then categorical
    categorical = [cfg.report_name(c) for c in cfg.categorical]
    cols = [c for c in cfg.flagged_report + categorical if real[c].isna().mean() >= 0.01]
    if not cols:
        return
    sites = sorted(real[site].dropna().unique())
    fig, axes = panel_grid(len(cols), 3, 13 / 3, 3, sharey=True)
    for ax, col in zip(axes, cols):
        r = real.groupby(site)[col].apply(lambda c: c.isna().mean()).reindex(sites)
        s = synth.groupby(site)[col].apply(lambda c: c.isna().mean()).reindex(sites)
        x = np.arange(len(sites))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, r, w, color=REAL, label="Real")
        ax.bar(x + w / 2 + 0.01, s, w, color=SYNTH, label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels([site.replace(" ", "\n") for site in sites], fontsize=8)
        ax.set_title(label(col))
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"% of patients with the value not measured, by {label(site).lower()}", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig_missingness_by_site.png")


def fig_correlation(real, synth):
    cols = column_groups()["correlation"]
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
    if not os.path.exists(runs_path):
        print(f"⏭️ No multi-seed results yet ({os.path.relpath(runs_path, BASE_DIR)}); "
              "run research/multi_seed.py for this dataset first.")
        return
    runs = pd.read_csv(runs_path).rename(columns=LEGACY_NAMES)
    models = ["MedTwin WGAN-GP"] + sorted(m for m in runs.model.unique() if m != "MedTwin WGAN-GP")
    metrics = [
        ("fidelity_pct", "Fidelity (%) ↑", None),
        ("auc_ratio", "Utility: TSTR / TRTR AUC → 1", 1.0),
        ("detectability_pct", "Detectability (%) → 50", 50),
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
    parser.add_argument("--synthetic", help="synthetic report CSV (default: the dataset's latest)")
    parser.add_argument("--benchmark", action="store_true", help="also plot the multi-seed comparison")
    config.add_argument(parser)
    args = parser.parse_args()

    cfg = config.use(args.config)
    synth_path = args.synthetic or os.path.join(cfg.output_dir, "grounded_synthetic_data_latest.csv")
    real, synth = load(synth_path)
    fig_continuous(real, synth)
    fig_categorical(real, synth)
    fig_missingness(real, synth)
    fig_correlation(real, synth)
    if args.benchmark:
        fig_benchmark(os.path.join(cfg.output_dir, "multi_seed", "all_runs.csv"))
