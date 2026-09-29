"""Held-out evaluation harness for synthetic data.

The generator is trained on a TRAIN split of the real data only; a HOLDOUT
split of real patients is never seen during training. Comparing synthetic
records against both splits is what makes utility and privacy claims
testable:

- TSTR utility: a classifier trained on synthetic data, tested on real
  holdout patients, compared with one trained on real train data (TRTR).
- DCR: whether synthetic records sit closer to the training patients than to
  unseen holdout patients (a sign of memorisation).
- Membership inference: whether an attacker holding the synthetic data can
  tell which real patients were in the training set.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from core.ingestion import FLAGGED_COLS, add_missing_flags, mark_missing
from core.medical_logic import MedicalLogicFilter

HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 42

# Raw source columns -> column names used in main.py's synthetic report
REAL_TO_REPORT = {"sex": "gender", "cp": "chest_pain", "dataset": "origin_study"}


def split_real_data(raw_df, seed=SPLIT_SEED):
    """Deterministic train/holdout split, stratified on disease severity.
    Different seeds hold out different patients (see research/multi_seed.py)."""
    return train_test_split(
        raw_df,
        test_size=HOLDOUT_FRACTION,
        stratify=raw_df["num"],
        random_state=seed,
    )


def align_real_to_report(raw_df, report_cols):
    """Puts raw real data on the synthetic report's schema.

    "Not measured" zeros become blanks, then the same MedicalLogicFilter the
    synthetic data went through is applied, so both sides are cleaned the
    same way. Blank flagged columns (e.g. chol) are kept -- they're a real
    property of those patients -- while rows missing anything else are dropped.
    """
    real_df = MedicalLogicFilter().apply_rules(mark_missing(raw_df))
    real_df = real_df.rename(columns=REAL_TO_REPORT)
    shared_cols = [c for c in report_cols if c in real_df.columns]
    required = [c for c in shared_cols if c not in FLAGGED_COLS]
    return real_df[shared_cols].dropna(subset=required)


def fill_missing(frames, reference):
    """Turns blanks in flagged columns into a 0/1 flag plus the reference
    median, for classifiers and distance metrics that can't take NaN."""
    fill_values = {c: reference[c].median() for c in FLAGGED_COLS if c in reference.columns}
    return [add_missing_flags(f, fill_values) for f in frames]


def _encode(frames, reference):
    """One-hot encodes categoricals and MinMax-scales numerics, using
    `reference` (the real train split) for the category set and the scale, so
    every frame lives in the same feature space."""
    reference, *frames = fill_missing([reference, *frames], reference)
    cat_cols = [c for c in reference.columns if reference[c].dtype == object]
    num_cols = [c for c in reference.columns if c not in cat_cols]

    ref_encoded = pd.get_dummies(reference, columns=cat_cols)
    lo = reference[num_cols].min()
    span = (reference[num_cols].max() - lo).replace(0, 1)

    encoded = []
    for frame in frames:
        enc = pd.get_dummies(frame[reference.columns], columns=cat_cols)
        enc = enc.reindex(columns=ref_encoded.columns, fill_value=0)
        enc[num_cols] = (enc[num_cols] - lo) / span
        encoded.append(enc.astype(float).values)
    return encoded


def _nearest_distance(target, queries):
    nn = NearestNeighbors(n_neighbors=1).fit(target)
    return nn.kneighbors(queries)[0][:, 0]


def evaluate_utility(train_real, holdout_real, synth):
    """TSTR vs. TRTR on the binary task 'has heart disease' (num > 0)."""
    features = [c for c in train_real.columns if c != "num"]
    X_train_real, X_hold, X_synth = _encode(
        [train_real, holdout_real, synth], train_real[features]
    )
    y_train_real = (train_real["num"] > 0).astype(int)
    y_hold = (holdout_real["num"] > 0).astype(int)
    y_synth = (synth["num"] > 0).astype(int)

    results = {}
    for name, X, y in [("trtr", X_train_real, y_train_real), ("tstr", X_synth, y_synth)]:
        clf = RandomForestClassifier(n_estimators=200, random_state=SPLIT_SEED)
        clf.fit(X, y)
        if len(clf.classes_) == 1:
            # Training data had only one class (e.g. a collapsed generator
            # producing no sick patients): constant prediction, AUC 0.5.
            proba = np.full(len(X_hold), float(clf.classes_[0]))
        else:
            proba = clf.predict_proba(X_hold)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "accuracy": round(accuracy_score(y_hold, pred) * 100, 2),
            "f1": round(f1_score(y_hold, pred) * 100, 2),
            "auc": round(roc_auc_score(y_hold, proba), 3),
        }
    results["auc_ratio"] = round(results["tstr"]["auc"] / results["trtr"]["auc"], 3)
    return results


def evaluate_privacy(train_real, holdout_real, synth):
    """Distance-based privacy checks against the held-out patients.

    The train split is subsampled to the holdout's size so the two real sets
    are compared on equal footing (a bigger set is closer to everything just
    by being bigger).
    """
    rng = np.random.RandomState(SPLIT_SEED)
    n = min(len(train_real), len(holdout_real))
    train_sub = train_real.iloc[rng.choice(len(train_real), n, replace=False)]
    hold_sub = holdout_real.iloc[rng.choice(len(holdout_real), n, replace=False)]

    X_train, X_hold, X_synth = _encode([train_sub, hold_sub, synth], train_real)

    # DCR: for each synthetic record, is its nearest real neighbour a
    # training patient or an unseen one? ~50% = no memorisation.
    d_train = _nearest_distance(X_train, X_synth)
    d_hold = _nearest_distance(X_hold, X_synth)
    closer_to_train = float(np.mean(d_train < d_hold))

    # Membership inference: attacker scores each real patient by how close
    # the nearest synthetic record is. AUC 0.5 = can't tell members from
    # non-members; 1.0 = full leakage.
    member_dist = _nearest_distance(X_synth, X_train)
    nonmember_dist = _nearest_distance(X_synth, X_hold)
    labels = np.r_[np.ones(n), np.zeros(n)]
    scores = -np.r_[member_dist, nonmember_dist]
    mia_auc = roc_auc_score(labels, scores)

    return {
        "dcr_closer_to_train_pct": round(closer_to_train * 100, 2),
        "median_dcr_train": round(float(np.median(d_train)), 4),
        "median_dcr_holdout": round(float(np.median(d_hold)), 4),
        "exact_copies_of_train": int(np.sum(d_train == 0)),
        "mia_auc": round(float(mia_auc), 3),
        "n_per_real_set": int(n),
    }


def evaluate_synthetic(train_raw, holdout_raw, synth_report):
    """Runs the full held-out evaluation. Takes raw real splits (source
    schema) and the synthetic report (report schema)."""
    train_real = align_real_to_report(train_raw, synth_report.columns)
    holdout_real = align_real_to_report(holdout_raw, synth_report.columns)
    synth = synth_report[train_real.columns]
    return {
        "utility": evaluate_utility(train_real, holdout_real, synth),
        "privacy": evaluate_privacy(train_real, holdout_real, synth),
    }
