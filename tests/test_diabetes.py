"""The same pipeline on a second dataset, driven only by
configs/diabetes.yaml. It differs from heart disease in ways the settings
file must handle: all columns numeric, no hospital column (unconditioned),
renamed numeric columns, and zeros that mean "not measured" in some columns
but are real values in another (pregnancies)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core import config, medtwin
from core.evaluation import evaluate_synthetic, split_real_data
from core.ingestion import mark_missing
from core.metrics import calculate_fidelity_score
from research.adversarial_audit import calculate_privacy_score
from research.baselines import prepare_train, train_and_generate
from research.tabddpm import TabDDPM

SETTINGS = os.path.join(ROOT, "configs", "diabetes.yaml")


@pytest.fixture(scope="module", autouse=True)
def cfg():
    return config.use(SETTINGS)


@pytest.fixture(scope="module")
def raw(cfg):
    return pd.read_csv(cfg.data_file)


@pytest.fixture(scope="module")
def split(raw):
    return split_real_data(raw, seed=0)


@pytest.fixture(scope="module")
def report(split):
    torch.manual_seed(0)
    np.random.seed(0)
    ingestor, processed = medtwin.prepare(split[0])
    old_epochs, medtwin.EPOCHS = medtwin.EPOCHS, 3
    try:
        generator = medtwin.train(ingestor, processed)
    finally:
        medtwin.EPOCHS = old_epochs
    return medtwin.generate_report(generator, ingestor, n=300)


def test_settings_describe_the_data(cfg):
    # Missing-value columns are found from the data, not listed by hand
    assert cfg.flagged == ["plas", "pres", "skin", "insu", "mass"]
    assert "preg" not in cfg.zero_means_missing  # 0 pregnancies is real
    assert cfg.condition_on is None
    assert cfg.report_columns[:2] == ["pregnancies", "glucose"]


def test_report_uses_report_names_and_blanks(cfg, report):
    assert list(report.columns) == cfg.report_columns
    for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        assert not (report[col] == 0).any(), col  # "not measured" is blank, never 0
    assert set(report["outcome"]) <= {0, 1}
    assert report["pregnancies"].between(0, 17).all()


def test_values_respect_plausible_ranges(cfg, report):
    for col, (lo, hi) in cfg.by_both_names(cfg.plausible_ranges).items():
        if col in report.columns:
            assert report[col].dropna().astype(float).between(lo, hi).all(), col


def test_evaluation_and_fidelity_cover_every_column(cfg, raw, split, report):
    result = evaluate_synthetic(split[0], split[1], report)
    assert 0 <= result["privacy"]["mia_auc"] <= 1
    assert 0 <= calculate_privacy_score(raw, report) <= 100
    # All four fidelity columns must be compared (renamed ones included)
    real = mark_missing(raw).rename(columns=cfg.rename)
    assert calculate_fidelity_score(real, report, cfg.fidelity_report) > 0
    with pytest.raises(KeyError):
        calculate_fidelity_score(mark_missing(raw), report, cfg.fidelity_report)


def test_baseline_runs_from_settings(cfg, split):
    train_df, discrete = prepare_train(split[0])
    synth, _ = train_and_generate(
        lambda: TabDDPM(steps=30, num_timesteps=20, seed=0), train_df, discrete, seed=0
    )
    assert list(synth.columns) == cfg.report_columns


def test_only_applicable_ablations_run(cfg):
    # No site column and no zero-spiked column in this data
    assert set(medtwin.applicable_ablations()) == {
        "full", "no_corr_loss", "no_missing_flags", "no_filter"
    }
