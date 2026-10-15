"""Missing-value fidelity and privacy checks (core/missingness.py) on cases
whose answer is known: real patients handed in as "synthetic", and data with
every gap filled in."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core import config
from core.evaluation import align_real_to_report, split_real_data
from core.missingness import (
    evaluate_missingness,
    evaluate_missingness_privacy,
    missing_columns,
    patterns,
)


@pytest.fixture(scope="module")
def heart():
    cfg = config.use(config.DEFAULT)
    train_raw, hold_raw = split_real_data(pd.read_csv(cfg.data_file), seed=0)
    return (
        align_real_to_report(train_raw, cfg.report_columns),
        align_real_to_report(hold_raw, cfg.report_columns),
    )


def filled(df):
    out = df.copy()
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode()[0])
    return out


def test_patterns_include_site_and_every_gap(heart):
    train, _ = heart
    cols = missing_columns(train)
    assert "origin_study" not in cols and "chol" in cols and "thal" in cols
    p = patterns(train, cols)
    assert p.str.startswith(tuple(train["origin_study"].unique())).all()
    assert p.str.split("|").str[1].str.len().eq(len(cols)).all()


def test_copies_score_like_real_data_and_filled_gaps_score_badly(heart):
    train, hold = heart
    copies = evaluate_missingness(train, hold, train.copy())
    gapless = evaluate_missingness(train, hold, filled(train))
    # Copies are about as close as a second real sample
    assert copies["pattern_tvd"] <= copies["pattern_tvd_ref"] * 1.2
    # Filling every gap is far worse on every score
    for key in ["miss_rate_gap", "site_miss_gap", "pattern_tvd"]:
        assert gapless[key] > 5 * copies[key], key


def test_privacy_attacks_catch_copied_patients(heart):
    train, hold = heart
    leak = evaluate_missingness_privacy(train, hold, train.copy())
    assert leak["rare_pattern_dist_mia_auc"] > 0.9
    assert leak["unique_pattern_leak_pct"] > 50


def test_privacy_attacks_see_no_leak_in_unrelated_real_patients(heart):
    train, hold = heart
    # Real patients the "generator" never saw: half of the held-out set
    # stands in for synthetic data, scored against the other half.
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(hold))
    fake_hold, stand_in = hold.iloc[idx[: len(idx) // 2]], hold.iloc[idx[len(idx) // 2 :]]
    result = evaluate_missingness_privacy(train, fake_hold, stand_in)
    assert abs(result["common_pattern_dist_mia_auc"] - 0.5) < 0.15


def test_unconditioned_data_has_no_site_score():
    cfg = config.use(os.path.join(ROOT, "configs", "diabetes.yaml"))
    try:
        train_raw, hold_raw = split_real_data(pd.read_csv(cfg.data_file), seed=0)
        train = align_real_to_report(train_raw, cfg.report_columns)
        hold = align_real_to_report(hold_raw, cfg.report_columns)
        result = evaluate_missingness(train, hold, train.copy())
        assert "site_miss_gap" not in result and "pattern_tvd" in result
    finally:
        config.use(config.DEFAULT)
