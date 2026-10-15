"""LaTeX tables for the paper, built straight from the saved multi-seed,
ablation and sensitivity results, so no number in the paper is typed by
hand.

    python research/paper_tables.py      # writes paper/tables/*.tex

Each cell is mean ± SD over seeds. In the benchmark tables the best mean
per column is bold. In the ablation and sensitivity tables a * marks a
variant whose paired difference from the full model has a 95% CI that
excludes 0 (see research/multi_seed.py: paired_table).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import config, medtwin
from research.multi_seed import MEDTWIN, METRICS, load_runs, paired_table

OUT = os.path.join(config.ROOT, "paper", "tables")
DATASETS = {
    "heart_disease": ("configs/heart_disease.yaml", "Heart disease"),
    "diabetes": ("configs/diabetes.yaml", "Diabetes"),
}
MODEL_LABELS = {MEDTWIN: "MedTwin (ours)", "CTGAN": "CTGAN", "TVAE": "TVAE", "TabDDPM": "TabDDPM"}
VARIANT_LABELS = {
    "full": "Full model",
    "no_corr_loss": "w/o correlation loss",
    "no_missing_flags": "w/o missing-value flags",
    "no_zero_flags": "w/o zero flag (oldpeak)",
    "no_condition": "w/o site conditioning",
    "corr_weight_0.5": "Correlation weight 0.5 (default 2)",
    "corr_weight_8": "Correlation weight 8",
    "epochs_300": "300 epochs (default 600)",
    "epochs_1200": "1200 epochs",
}

# (column, header, decimals). Arrows say what is better.
BENCHMARK_COLS = [
    ("fidelity_pct", r"Fidelity (\%) $\uparrow$", 1),
    ("tstr_auc", r"TSTR AUC $\uparrow$", 3),
    ("auc_ratio", r"AUC ratio $\uparrow$", 3),
    ("detectability_pct", r"Detect. (\%) $\to 50$", 1),
    ("dcr_closer_to_train_pct", r"DCR (\%) $\to 50$", 1),
    ("mia_auc", r"MIA AUC $\to 0.5$", 3),
]
ABLATION_COLS = [
    ("fidelity_pct", r"Fid.\ (\%)", 1),
    ("detectability_pct", r"Detect.\ (\%)", 1),
    ("corr_gap", r"Corr.\ gap", 3),
    ("auc_ratio", r"AUC ratio", 3),
    ("dcr_closer_to_train_pct", r"DCR (\%)", 1),
    ("mia_auc", r"MIA AUC", 3),
]

# Missing-value fidelity (gaps, 0 = same as real) and privacy (attacks)
MISSING_COLS = [
    ("miss_rate_gap", r"Rate gap $\downarrow$", 3),
    ("site_miss_gap", r"Site gap $\downarrow$", 3),
    ("pattern_tvd", r"Pattern TVD $\downarrow$", 3),
    ("rare_pattern_dist_mia_auc", r"MIA rare $\to 0.5$", 3),
    ("common_pattern_dist_mia_auc", r"MIA common $\to 0.5$", 3),
    ("unique_pattern_leak_pct", r"Unique leak $\to 0$", 1),
]


def cell(values, decimals, bold=False, star=False):
    values = values.dropna()
    if values.empty:
        return "--"
    text = f"{values.mean():.{decimals}f} $\\pm$ {values.std(ddof=1):.{decimals}f}"
    if bold:
        text = f"\\textbf{{{values.mean():.{decimals}f}}} $\\pm$ {values.std(ddof=1):.{decimals}f}"
    return text + ("$^*$" if star else "")


def best(means, metric):
    rule = METRICS[metric]
    if "target" in rule:
        return (means - rule["target"]).abs().idxmin()
    return means.idxmax()


def benchmark_table(runs):
    models = [MEDTWIN] + sorted(m for m in runs.model.unique() if m != MEDTWIN)
    header = " & ".join(["Model"] + [h for _, h, _ in BENCHMARK_COLS])
    lines = [r"\begin{tabular}{l" + "c" * len(BENCHMARK_COLS) + "}", r"\toprule", header + r" \\", r"\midrule"]
    means = runs.groupby("model")[[c for c, _, _ in BENCHMARK_COLS]].mean()
    winners = {c: best(means[c], c) for c, _, _ in BENCHMARK_COLS}
    for model in models:
        sub = runs[runs.model == model]
        cells = [cell(sub[c], d, bold=winners[c] == model) for c, _, d in BENCHMARK_COLS]
        lines.append(" & ".join([MODEL_LABELS.get(model, model)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def clear_differences(runs, reference, variants):
    """{(variant, metric)} where the paired 95% CI vs the full model excludes 0."""
    table = paired_table(runs, reference, variants, column="variant", label="full")
    return {(r.vs, r.metric) for r in table.itertuples() if r.verdict != "no clear difference"}


def variant_table(blocks):
    """blocks: [(dataset label, runs, variant order)]"""
    header = " & ".join(["Variant"] + [h for _, h, _ in ABLATION_COLS])
    lines = [r"\begin{tabular}{l" + "c" * len(ABLATION_COLS) + "}", r"\toprule", header + r" \\"]
    for label, runs, order in blocks:
        order = [v for v in order if v in set(runs.variant)]
        stars = clear_differences(runs, "full", order[1:])
        lines += [r"\midrule", rf"\multicolumn{{{len(ABLATION_COLS) + 1}}}{{l}}{{\textit{{{label}}}}} \\"]
        for v in order:
            sub = runs[runs.variant == v]
            cells = [cell(sub[c], d, star=(v, c) in stars) if c in sub else "--" for c, _, d in ABLATION_COLS]
            lines.append(" & ".join([VARIANT_LABELS.get(v, v)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def missingness_table(blocks):
    """blocks: [(dataset label, multi-seed runs, ablation runs)]. Per dataset:
    a real-vs-real reference row (held-out vs training patients), every
    model, then the ablations that touch missingness."""
    cols = MISSING_COLS
    header = " & ".join(["Data"] + [h for _, h, _ in cols])
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", header + r" \\"]
    for label, runs, ablation in blocks:
        lines += [r"\midrule", rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{label}}}}} \\"]
        reference = runs[runs.model == MEDTWIN]
        ref_cells = [cell(reference[f"{c}_ref"], d) if f"{c}_ref" in reference else "" for c, _, d in cols]
        lines.append(" & ".join([r"Real vs.\ real (reference)"] + ref_cells) + r" \\")
        models = [MEDTWIN] + sorted(m for m in runs.model.unique() if m != MEDTWIN)
        means = runs.groupby("model")[[c for c, _, _ in cols if c in runs]].mean()
        # Only the fidelity columns get a winner: the privacy scores are too
        # noisy across seeds for a best mean to mean much.
        winners = {c: best(means[c], c) for c in ["miss_rate_gap", "site_miss_gap", "pattern_tvd"]
                   if c in means and means[c].notna().any()}
        for model in models:
            sub = runs[runs.model == model]
            cells = [cell(sub[c], d, bold=winners.get(c) == model) if c in sub else "--" for c, _, d in cols]
            lines.append(" & ".join([MODEL_LABELS.get(model, model)] + cells) + r" \\")
        order = [v for v in ["full", "no_missing_flags", "no_condition"]
                 if v in set(ablation.variant) and ablation[ablation.variant == v]["pattern_tvd"].notna().any()]
        if len(order) > 1:
            stars = clear_differences(ablation, "full", order[1:])
            lines.append(r"\cmidrule(l){1-" + str(len(cols) + 1) + "}")
            for v in order:
                sub = ablation[ablation.variant == v]
                name = "MedTwin, ablation run" if v == "full" else "MedTwin " + VARIANT_LABELS.get(v, v)
                cells = [cell(sub[c], d, star=(v, c) in stars) if c in sub else "--" for c, _, d in cols]
                lines.append(" & ".join([name] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Generated by research/paper_tables.py -- do not edit by hand.\n" + text + "\n")
    print(f"📝 {os.path.relpath(path, config.ROOT)}")


def main():
    ablation_blocks, sensitivity_blocks, missing_blocks = [], [], []
    for key, (settings, label) in DATASETS.items():
        config.use(settings)
        runs = load_runs(os.path.join(config.active().output_dir, "multi_seed"))
        write(f"benchmark_{key}.tex", benchmark_table(runs))

        ablation = load_runs(os.path.join(config.active().output_dir, "ablation"))
        ablation_blocks.append((label, ablation, list(medtwin.applicable_ablations())))
        if "pattern_tvd" in runs:
            missing_blocks.append((label, runs, ablation))
        if any(v in set(ablation.variant) for v in medtwin.SENSITIVITY if v != "full"):
            sensitivity_blocks.append((label, ablation, list(medtwin.SENSITIVITY)))

    write("ablation.tex", variant_table(ablation_blocks))
    write("sensitivity.tex", variant_table(sensitivity_blocks))
    if missing_blocks:
        write("missingness.tex", missingness_table(missing_blocks))


if __name__ == "__main__":
    main()
