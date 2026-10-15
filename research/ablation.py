"""Ablation study: retrains MedTwin with one component switched off at a
time (see core.medtwin.ABLATIONS) and compares each variant with the full
model, paired by seed.

Each seed changes which real patients are held out and every model's random
initialisation, exactly as in research/multi_seed.py; within a seed every
variant starts from the same random state, so the only difference between
paired runs is the component switched off. Results are saved per
(seed, variant), so an interrupted run resumes where it stopped.

    python research/ablation.py --seed 3        # run one seed (all variants)
    python research/ablation.py --summarize     # mean ± sd, paired tests

Add --config configs/<dataset>.yaml to run on another dataset; variants that
change nothing for it (e.g. no_condition without a site column) are skipped.
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

# Same reason as research/multi_seed.py: keeps joblib from spawning a worker
# per core when several seeds run in parallel.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import config, medtwin
from core.evaluation import split_real_data
from research.baselines import flatten, score, set_seed
from research.multi_seed import SEEDS, mean_sd_table, paired_table

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

FULL = "full"


def out_dir():
    """Results for the active dataset, e.g. data/synthetic/ablation/."""
    return os.path.join(config.active().output_dir, "ablation")


def result_path(seed, name):
    return os.path.join(out_dir(), f"seed{seed}_{name}.json")


def run_seed(seed, names=None):
    warnings.filterwarnings("ignore")
    os.makedirs(out_dir(), exist_ok=True)
    raw_df = pd.read_csv(config.active().data_file)
    train_raw, holdout_raw = split_real_data(raw_df, seed=seed)

    for name, variant in medtwin.applicable_ablations().items():
        if names and name not in names:
            continue
        if os.path.exists(result_path(seed, name)):
            continue
        print(f"\n🧪 seed {seed} | {name}")
        set_seed(seed)
        ingestor, processed_df = medtwin.prepare(train_raw, variant)
        start = time.time()
        generator = medtwin.train(ingestor, processed_df)
        seconds = round(time.time() - start, 1)
        synth = medtwin.generate_report(generator, ingestor)

        row = flatten(name, len(synth), score(raw_df, train_raw, holdout_raw, synth), seconds)
        row["variant"] = row.pop("model")
        row["seed"] = seed
        with open(result_path(seed, name), "w") as f:
            json.dump(row, f, indent=2, default=float)
        print(f"✅ seed {seed} | {name}: saved")


def summarize():
    rows = []
    for path in glob.glob(os.path.join(out_dir(), "seed*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        print(f"No ablation results yet in {out_dir()}")
        return
    runs = pd.DataFrame(rows).sort_values(["variant", "seed"])
    runs.to_csv(os.path.join(out_dir(), "all_runs.csv"), index=False)

    order = [n for n in medtwin.ABLATIONS if n in set(runs.variant)]
    print(f"Seeds completed per variant: {runs.groupby('variant').seed.nunique().to_dict()}")

    summary = mean_sd_table(runs, order, column="variant")
    summary.to_csv(os.path.join(out_dir(), "summary.csv"), index=False)
    print("\n📊 Mean ± SD across seeds")
    print(summary.to_string(index=False))

    # Advantage > 0 = the full model is better, i.e. the switched-off
    # component helps on that metric.
    comparisons = paired_table(runs, FULL, order[1:], column="variant", label="full model")
    comparisons.to_csv(os.path.join(out_dir(), "paired_comparisons.csv"), index=False)
    print("\n⚖️ Full model vs each ablation, paired by seed "
          "(advantage > 0 = the removed component helps; 'clear' = 95% CI excludes 0)")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="run one seed")
    parser.add_argument("--variant", nargs="+", help="run only these variants")
    parser.add_argument("--threads", type=int, default=2, help="torch threads per process")
    parser.add_argument("--summarize", action="store_true")
    config.add_argument(parser)
    args = parser.parse_args()

    config.use(args.config)
    torch.set_num_threads(args.threads)
    if args.summarize:
        summarize()
    elif args.seed is not None:
        run_seed(args.seed, args.variant)
    else:
        for s in SEEDS:
            run_seed(s, args.variant)
        summarize()
