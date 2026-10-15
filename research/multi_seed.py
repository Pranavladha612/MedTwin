"""Repeats the MedTwin-vs-baselines comparison over several random seeds.

Each seed changes BOTH which real patients are held out and every model's
random initialisation, so the spread across seeds shows how much of a score
difference is luck. Results are saved per (seed, model), so an interrupted
run resumes where it stopped.

    python research/multi_seed.py --seed 3          # run one seed (all models)
    python research/multi_seed.py --summarize       # mean ± sd, paired tests

Add --config configs/<dataset>.yaml to run on another dataset.
"""

import argparse
import glob
import json
import os
import sys
import warnings

# ctgan's data preparation runs joblib with n_jobs=-1, i.e. one ~250 MB worker
# process per CPU core. With several seeds running in parallel that exhausted
# the machine's RAM, so joblib is limited to one core (it then runs in-process).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import config, medtwin
from core.evaluation import split_real_data
from research.baselines import (
    build_models,
    flatten,
    prepare_train,
    score,
    set_seed,
    train_and_generate,
)

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SEEDS = [0, 1, 2, 3, 4]
MEDTWIN = "MedTwin WGAN-GP"

# How each metric is judged. "target" metrics are best when closest to the
# target value (e.g. MIA AUC 0.5); the others are best when highest.
METRICS = {
    "fidelity_pct": {"higher": True},
    "tstr_auc": {"higher": True},
    "auc_ratio": {"higher": True},
    "discriminator_pct": {"target": 50},
    "dcr_closer_to_train_pct": {"target": 50},
    "mia_auc": {"target": 0.5},
}


def out_dir():
    """Results for the active dataset, e.g. data/synthetic/multi_seed/."""
    return os.path.join(config.active().output_dir, "multi_seed")


def result_path(seed, model):
    return os.path.join(out_dir(), f"seed{seed}_{model.split()[0].lower()}.json")


def run_seed(seed):
    warnings.filterwarnings("ignore")
    os.makedirs(out_dir(), exist_ok=True)
    raw_df = pd.read_csv(config.active().data_file)
    train_raw, holdout_raw = split_real_data(raw_df, seed=seed)

    def save(name, synth, seconds):
        row = flatten(name, len(synth), score(raw_df, train_raw, holdout_raw, synth), seconds)
        row["seed"] = seed
        with open(result_path(seed, name), "w") as f:
            json.dump(row, f, indent=2, default=float)
        print(f"✅ seed {seed} | {name}: saved")

    if not os.path.exists(result_path(seed, MEDTWIN)):
        print(f"\n🏗️ seed {seed} | {MEDTWIN}")
        set_seed(seed)
        ingestor, processed_df = medtwin.prepare(train_raw)
        start = pd.Timestamp.now()
        generator = medtwin.train(ingestor, processed_df)
        seconds = round((pd.Timestamp.now() - start).total_seconds(), 1)
        save(MEDTWIN, medtwin.generate_report(generator, ingestor), seconds)

    train_df, discrete_cols = prepare_train(train_raw)
    for name, make in build_models(seed).items():
        if os.path.exists(result_path(seed, name)):
            continue
        print(f"\n🏗️ seed {seed} | {name}")
        synth, seconds = train_and_generate(make, train_df, discrete_cols, seed=seed)
        save(name, synth, seconds)


def mean_sd_table(runs, models, column="model"):
    """Mean ± SD of every metric per model (or per ablation variant)."""
    summary = []
    for model in models:
        sub = runs[runs[column] == model]
        row = {column: model, "n_seeds": len(sub)}
        for m in METRICS:
            row[m] = f"{sub[m].mean():.3f} ± {sub[m].std(ddof=1):.3f}"
        summary.append(row)
    return pd.DataFrame(summary)


def paired_table(runs, reference, others, column="model", label=None):
    """`reference` vs each of `others` on the same seeds. For target metrics,
    compares distance from the target (smaller = better). Positive
    "advantage" = the reference is better. `label` names the reference in
    the verdicts (default: its full name)."""
    label = label or reference
    comparisons = []
    for other in others:
        both = runs[runs[column].isin([reference, other])].pivot(index="seed", columns=column)
        both = both.dropna()
        for m, rule in METRICS.items():
            a, b = both[m][reference].astype(float), both[m][other].astype(float)
            if "target" in rule:
                adv = (b - rule["target"]).abs() - (a - rule["target"]).abs()
            else:
                adv = a - b
            n = len(adv)
            mean = adv.mean()
            half = stats.t.ppf(0.975, n - 1) * adv.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            p = stats.ttest_1samp(adv, 0).pvalue if n > 1 else np.nan
            comparisons.append({
                "vs": other,
                "metric": m,
                "advantage": round(mean, 4),
                "ci95_low": round(mean - half, 4),
                "ci95_high": round(mean + half, 4),
                "p_value": round(p, 3),
                "wins": f"{int((adv > 0).sum())}/{n}",
                "verdict": (
                    f"{label} better" if mean - half > 0
                    else f"{other} better" if mean + half < 0
                    else "no clear difference"
                ),
            })
    return pd.DataFrame(comparisons)


def summarize():
    rows = []
    for path in glob.glob(os.path.join(out_dir(), "seed*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    runs = pd.DataFrame(rows).sort_values(["model", "seed"])
    runs.to_csv(os.path.join(out_dir(), "all_runs.csv"), index=False)

    models = [MEDTWIN] + sorted(m for m in runs.model.unique() if m != MEDTWIN)
    seeds_per_model = runs.groupby("model").seed.nunique()
    print(f"Seeds completed per model: {seeds_per_model.to_dict()}")

    # 1. Mean ± SD per model
    summary = mean_sd_table(runs, models)
    summary.to_csv(os.path.join(out_dir(), "summary.csv"), index=False)
    print("\n📊 Mean ± SD across seeds")
    print(summary.to_string(index=False))

    # 2. Paired comparison: MedTwin vs each baseline on the same seeds.
    comparisons = paired_table(runs, MEDTWIN, models[1:], label="MedTwin").rename(
        columns={"advantage": "medtwin_advantage", "wins": "medtwin_wins"}
    )
    comparisons.to_csv(os.path.join(out_dir(), "paired_comparisons.csv"), index=False)
    print("\n⚖️ MedTwin vs each baseline, paired by seed "
          "(advantage > 0 = MedTwin better; 'clear' = 95% CI excludes 0)")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="run one seed")
    parser.add_argument("--threads", type=int, default=2, help="torch threads per process")
    parser.add_argument("--summarize", action="store_true")
    config.add_argument(parser)
    args = parser.parse_args()

    config.use(args.config)
    torch.set_num_threads(args.threads)
    if args.summarize:
        summarize()
    elif args.seed is not None:
        run_seed(args.seed)
    else:
        for s in SEEDS:
            run_seed(s)
        summarize()
