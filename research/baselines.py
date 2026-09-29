"""Benchmarks MedTwin's WGAN-GP against standard tabular generators.

Every baseline is trained on the same real TRAIN split main.py uses (the
held-out patients stay unseen), gets the same clinical filtering and rounding
as MedTwin's output, and is scored with the same metrics.

Run after main.py (it reads MedTwin's own scores from latest_metrics.json):

    python research/baselines.py
"""

import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ctgan import CTGAN, TVAE

from core.evaluation import (
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    align_real_to_report,
    evaluate_synthetic,
    fill_missing,
    split_real_data,
)
from core.ingestion import FLAGGED_COLS, blank_flagged, mark_missing, missing_flag
from core.medical_logic import MedicalLogicFilter, quantize_clinical
from core.metrics import calculate_fidelity_score
from research.adversarial_audit import calculate_privacy_score
from research.tabddpm import TabDDPM

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_PATH = os.path.join(BASE_DIR, "data", "heart_disease_uci.csv")
SYNTH_DIR = os.path.join(BASE_DIR, "data", "synthetic")
OUT_DIR = os.path.join(SYNTH_DIR, "baselines")

# Same columns as MedTwin's grounded report (main.py)
REPORT_COLS = [
    "age", "gender", "chest_pain", "trestbps", "chol",
    "thalch", "oldpeak", "num", "origin_study",
]
DISCRETE_COLS = ["gender", "chest_pain", "origin_study", "num"]
N_SAMPLES = 2000  # same as main.py, before clinical filtering


def build_models(seed=SPLIT_SEED):
    # Batch sizes shrunk from the library defaults (500) because the train
    # split is only ~600 rows; must stay divisible by CTGAN's pac=10.
    return {
        "CTGAN": lambda: CTGAN(epochs=300, batch_size=100, verbose=False),
        "TVAE": lambda: TVAE(epochs=300, batch_size=100),
        "TabDDPM": lambda: TabDDPM(steps=8000, seed=seed),
    }


def set_seed(seed):
    """Seeds every RNG the models draw from (CTGAN/TVAE use the globals)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_train(train_raw):
    """Baselines get the clean report-schema view of the train split. Like
    MedTwin, blank cholesterol becomes a "was it measured?" flag plus a
    placeholder value, since none of these models accept blanks."""
    train_df = align_real_to_report(train_raw, REPORT_COLS)
    (train_df,) = fill_missing([train_df], train_df)
    discrete_cols = DISCRETE_COLS + [missing_flag(c) for c in FLAGGED_COLS]
    return train_df, discrete_cols


def train_and_generate(make, train_df, discrete_cols, seed=SPLIT_SEED):
    """Fits one baseline and returns (synthetic report, training seconds),
    post-processed exactly like MedTwin's output."""
    set_seed(seed)
    start = time.time()
    model = make()
    model.fit(train_df, discrete_columns=discrete_cols)
    seconds = round(time.time() - start, 1)

    synth = blank_flagged(model.sample(N_SAMPLES))
    synth = MedicalLogicFilter().apply_rules(synth)
    return quantize_clinical(synth)[REPORT_COLS], seconds


def score(raw_df, train_raw, holdout_raw, synth):
    evaluation = evaluate_synthetic(train_raw, holdout_raw, synth)
    return {
        "fidelity_pct": calculate_fidelity_score(
            mark_missing(raw_df), synth, ["age", "trestbps", "chol", "thalch"]
        ),
        "discriminator_pct": calculate_privacy_score(raw_df, synth),
        **evaluation,
    }


def flatten(name, n_kept, metrics, seconds):
    u, p = metrics["utility"], metrics["privacy"]
    return {
        "model": name,
        "rows_kept": n_kept,
        "fidelity_pct": metrics["fidelity_pct"],
        "discriminator_pct": metrics["discriminator_pct"],
        "tstr_auc": u["tstr"]["auc"],
        "trtr_auc": u["trtr"]["auc"],
        "auc_ratio": u["auc_ratio"],
        "dcr_closer_to_train_pct": p["dcr_closer_to_train_pct"],
        "mia_auc": p["mia_auc"],
        "exact_copies": p["exact_copies_of_train"],
        "train_seconds": seconds,
    }


def run_baselines():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    raw_df = pd.read_csv(REAL_PATH)
    train_raw, holdout_raw = split_real_data(raw_df)
    train_df, discrete_cols = prepare_train(train_raw)
    print(f"Training baselines on {len(train_df)} clean train rows "
          f"({HOLDOUT_FRACTION:.0%} of patients held out)")

    rows = []

    metrics_path = os.path.join(SYNTH_DIR, "latest_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            medtwin = json.load(f)
        if "utility" in medtwin:
            medtwin_df = pd.read_csv(
                os.path.join(SYNTH_DIR, "grounded_synthetic_data_latest.csv")
            )
            rows.append(flatten("MedTwin WGAN-GP", len(medtwin_df), medtwin, None))
    if not rows:
        print("⚠️ No held-out metrics for MedTwin yet; run main.py first to include it.")

    for name, make in build_models().items():
        print(f"\n🏗️ {name}")
        synth, seconds = train_and_generate(make, train_df, discrete_cols)
        synth.to_csv(os.path.join(OUT_DIR, f"{name.lower()}.csv"), index=False)

        rows.append(flatten(name, len(synth), score(raw_df, train_raw, holdout_raw, synth), seconds))

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(SYNTH_DIR, "baseline_comparison.csv"), index=False)

    print("\n📊 Baseline comparison (targets: auc_ratio → 1.0, "
          "dcr_closer_to_train_pct → 50, mia_auc → 0.5, discriminator_pct → 50)")
    print(table.drop(columns=["train_seconds"]).to_string(index=False))
    return table


if __name__ == "__main__":
    run_baselines()
