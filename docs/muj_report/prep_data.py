"""Collects every number the MUJ-format report needs from the saved
results into data.json, so build_report.js never contains a typed number.

    python docs/muj_report/prep_data.py
"""

import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from core import config, medtwin  # noqa: E402
from core.ingestion import mark_missing  # noqa: E402
from research.multi_seed import MEDTWIN, load_runs, paired_table  # noqa: E402

HEART = os.path.join(ROOT, "configs", "heart_disease.yaml")
DIAB = os.path.join(ROOT, "configs", "diabetes.yaml")
MODELS = [MEDTWIN, "TVAE", "TabDDPM", "CTGAN"]


def stats(runs, kind, group, metric):
    v = runs.loc[runs[kind] == group, metric].dropna() if metric in runs else pd.Series(dtype=float)
    if v.empty:
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def clear(runs, kind, ref, other, metric):
    t = paired_table(runs, ref, [other], column=kind, label="ref")
    row = t[t.metric == metric]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"clear": r.verdict != "no clear difference", "advantage": float(r.advantage),
            "low": float(r.ci95_low), "high": float(r.ci95_high), "wins": r.wins}


METRICS = ["fidelity_pct", "detectability_pct", "tstr_auc", "trtr_auc", "auc_ratio",
           "dcr_closer_to_train_pct", "mia_auc", "corr_gap", "corr_gap_numeric",
           "miss_rate_gap", "site_miss_gap", "pattern_tvd", "miss_rate_gap_ref",
           "site_miss_gap_ref", "pattern_tvd_ref", "rare_pattern_dist_mia_auc",
           "common_pattern_dist_mia_auc", "unique_pattern_leak_pct", "n_rare_patients",
           "n_unique_patterns", "train_seconds", "exact_copies"]


def dataset_block(settings, key):
    cfg = config.use(settings)
    raw = pd.read_csv(cfg.data_file)
    marked = mark_missing(raw)
    out = {"name": cfg.name, "n": int(len(raw)),
           "positive_pct": float((raw[cfg.target] > cfg.positive_above).mean() * 100),
           "numeric": cfg.numeric, "categorical": cfg.categorical, "condition": cfg.condition_on,
           "zero_means_missing": cfg.zero_means_missing, "flagged": cfg.flagged,
           "missing_pct": {c: float(marked[c].isna().mean() * 100) for c in marked.columns if c != cfg.id_column},
           "labels": cfg.labels, "rename": cfg.rename,
           "zero_pct": {c: float((raw[c] == 0).mean() * 100) for c in cfg.exactly_zero_spike}}
    if cfg.condition_on:
        out["sites"] = {k: int(v) for k, v in raw[cfg.condition_on].value_counts().items()}
        cols = ["chol", "trestbps", "ca", "thal", "slope", "fbs"]
        out["missing_by_site"] = {s: {c: float(g[c].isna().mean() * 100) for c in cols}
                                  for s, g in marked.groupby(cfg.condition_on)}
    runs = load_runs(os.path.join(cfg.output_dir, "multi_seed"))
    abl = load_runs(os.path.join(cfg.output_dir, "ablation"))
    out["models"] = {m: {k: stats(runs, "model", m, k) for k in METRICS} for m in MODELS}
    out["vs_baselines"] = {b: {k: clear(runs, "model", MEDTWIN, b, k) for k in METRICS[:16]}
                           for b in MODELS[1:]}
    variants = list(medtwin.applicable_ablations()) + [v for v in medtwin.SENSITIVITY if v != "full"]
    present = [v for v in variants if v in set(abl.variant)]
    out["variants"] = {v: {k: stats(abl, "variant", v, k) for k in METRICS} for v in present}
    out["vs_full"] = {v: {k: clear(abl, "variant", "full", v, k) for k in METRICS[:20]}
                      for v in present if v != "full"}
    sample = pd.read_csv(os.path.join(cfg.output_dir, "multi_seed", "seed0_medtwin.csv.gz"))
    out["sample"] = {"columns": list(sample.columns),
                     "rows": sample.head(8).astype(object).where(sample.head(8).notna(), None).values.tolist()}
    return out


def main():
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True).stdout
    n_tests = sum(1 for line in tests.splitlines() if "::" in line)
    data = {"heart": dataset_block(HEART, "h"), "diabetes": dataset_block(DIAB, "d"),
            "settings": {"epochs": medtwin.EPOCHS, "latent": medtwin.LATENT_DIM, "samples": medtwin.N_SAMPLES},
            "n_tests": n_tests}
    with open(os.path.join(HERE, "data.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    print("wrote data.json; tests:", n_tests)


if __name__ == "__main__":
    main()
