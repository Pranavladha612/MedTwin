"""End-to-end checks for every part of the pipeline, with tiny training
budgets so the whole suite runs in a few minutes:

    python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core import medtwin
from core.evaluation import (
    REPORT_CATEGORICAL,
    align_real_to_report,
    evaluate_synthetic,
    split_real_data,
)
from core.ingestion import (
    FLAGGED_COLS,
    NOT_MEASURED,
    ZERO_INFLATED,
    mark_missing,
    missing_flag,
    zero_flag,
)
from core.medical_logic import PLAUSIBLE_RANGES, MedicalLogicFilter
from research.adversarial_audit import calculate_privacy_score
from research.baselines import prepare_train, train_and_generate
from research.tabddpm import TabDDPM

RAW = pd.read_csv(os.path.join(ROOT, "data", "heart_disease_uci.csv"))


@pytest.fixture(scope="module")
def split():
    return split_real_data(RAW, seed=0)


@pytest.fixture(scope="module")
def trained(split):
    """A MedTwin model trained for 3 epochs: enough to exercise every code
    path, not to produce good data."""
    torch.manual_seed(0)
    np.random.seed(0)
    ingestor, processed = medtwin.prepare(split[0])
    old_epochs, medtwin.EPOCHS = medtwin.EPOCHS, 3
    try:
        generator = medtwin.train(ingestor, processed)
    finally:
        medtwin.EPOCHS = old_epochs
    return ingestor, processed, generator


@pytest.fixture(scope="module")
def report(trained):
    ingestor, _, generator = trained
    np.random.seed(0)
    return medtwin.generate_report(generator, ingestor, n=500)


# ---------- data preparation ----------

def test_training_matrix_is_complete_and_matches_output_layout(trained):
    ingestor, processed, _ = trained
    assert not processed.isna().any().any()
    assert processed.values.min() >= 0 and processed.values.max() <= 1
    assert sum(width for _, width in ingestor.output_info) == processed.shape[1]
    for col in FLAGGED_COLS:
        assert missing_flag(col) in processed.columns
    for col in ZERO_INFLATED:
        # the flag must mark real exact zeros (43% of oldpeak)
        assert processed[zero_flag(col)].mean() > 0.3
    assert "dataset" not in " ".join(processed.columns)  # site is the condition, not an output
    assert abs(ingestor.site_probs.sum() - 1) < 1e-9


def test_unmeasured_categories_are_a_category_not_invented(trained):
    ingestor, processed, _ = trained
    # 'thal' is missing for most Hungary/VA patients: it must be learned as
    # "not measured", not filled with the most common value.
    assert f"thal_{NOT_MEASURED}" in ingestor.cat_groups["thal"]
    assert processed[f"thal_{NOT_MEASURED}"].mean() > 0.3


def test_no_real_patient_breaks_the_plausibility_rules():
    kept = MedicalLogicFilter().apply_rules(mark_missing(RAW))
    assert len(kept) == len(RAW)
    assert len(align_real_to_report(RAW, medtwin.REPORT_COLS)) == len(RAW)


# ---------- generation ----------

def test_report_has_all_15_columns(report):
    assert list(report.columns) == medtwin.REPORT_COLS
    assert len(medtwin.REPORT_COLS) == 15


def test_report_values_are_valid(report):
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        values = report[col].dropna().astype(float)
        assert values.between(lo, hi).all(), col
    assert set(report["num"]) <= {0, 1, 2, 3, 4}
    assert set(report["ca"].dropna().astype(int)) <= {0, 1, 2, 3}
    assert set(report["origin_study"]) <= set(RAW["dataset"])
    for col in REPORT_CATEGORICAL:
        assert NOT_MEASURED not in set(report[col].dropna()), col  # blanks, not the label


def test_every_generated_category_is_exactly_one_hot(trained):
    ingestor, _, generator = trained
    cond = torch.zeros(50, len(ingestor.site_names))
    cond[:, 0] = 1
    generator.eval()
    with torch.no_grad():
        out = pd.DataFrame(
            generator(torch.randn(50, medtwin.LATENT_DIM), cond, hard=True).numpy(),
            columns=ingestor.columns,
        )
    for group in ingestor.cat_groups.values():
        assert (out[group].sum(axis=1) == 1).all()


def test_site_conditioning_controls_origin_study(trained):
    ingestor, _, generator = trained
    probs = np.zeros(len(ingestor.site_names))
    probs[ingestor.site_names.index("Switzerland")] = 1.0
    saved, ingestor.site_probs = ingestor.site_probs, probs
    try:
        swiss = medtwin.generate_report(generator, ingestor, n=100)
    finally:
        ingestor.site_probs = saved
    assert set(swiss["origin_study"]) == {"Switzerland"}


# ---------- evaluation ----------

def test_evaluation_runs_on_report_read_back_from_csv(tmp_path, split, report):
    # CSV round trip turns "True"/"False" into booleans; scores must not change.
    path = tmp_path / "report.csv"
    report.to_csv(path, index=False)
    reread = pd.read_csv(path)
    direct = evaluate_synthetic(split[0], split[1], report)
    roundtrip = evaluate_synthetic(split[0], split[1], reread)
    assert direct["privacy"]["mia_auc"] == roundtrip["privacy"]["mia_auc"]
    assert direct["utility"]["tstr"]["auc"] == roundtrip["utility"]["tstr"]["auc"]
    assert 0 <= calculate_privacy_score(RAW, reread) <= 100


def test_privacy_tests_flag_copied_patients(split):
    # Handing in the real training patients themselves must look like a leak.
    copies = align_real_to_report(split[0], medtwin.REPORT_COLS)
    result = evaluate_synthetic(split[0], split[1], copies)["privacy"]
    assert result["mia_auc"] > 0.9
    assert result["exact_copies_of_train"] > 0


# ---------- baselines ----------

@pytest.mark.parametrize(
    "make",
    [lambda: TabDDPM(steps=30, num_timesteps=20, seed=0)],
    ids=["tabddpm"],
)
def test_baseline_produces_valid_15_column_report(split, make):
    train_df, discrete = prepare_train(split[0])
    synth, _ = train_and_generate(make, train_df, discrete, seed=0)
    assert list(synth.columns) == medtwin.REPORT_COLS
    assert len(synth) > 0
