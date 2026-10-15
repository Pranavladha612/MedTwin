import argparse
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Path injection: lets this module find 'core' whether it's imported from the
# project root (e.g. by main.py) or run directly as `python
# research/adversarial_audit.py`.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import config
from core.evaluation import align_real_to_report, fill_missing, normalize_report

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def calculate_detectability(raw_real_df, synth_df):
    """Detectability: trains a classifier to tell real patients from
    synthetic ones.

    Returns the held-out accuracy as a percentage (0-100). ~50% means the
    synthetic records are indistinguishable from real ones; closer to 100%
    means they are easy to spot. This measures realism (fidelity of the
    joint distribution), NOT privacy: a generator that copied real patients
    would score a perfect 50%. Privacy is measured by DCR and membership
    inference against held-out patients (core/evaluation.py).

    `raw_real_df` is the raw source dataframe (e.g. heart_disease_uci.csv,
    with columns like 'sex'/'cp'/'dataset'). `synth_df` is the grounded
    synthetic report dataframe (with columns like 'gender'/'chest_pain'/
    'origin_study'). The two use different column names/encodings (see
    core/ingestion.py and main.py's category reconstruction step), so this
    aligns them onto a shared schema before comparing.

    The real data is cleaned exactly as the synthetic data was (see
    core.evaluation.align_real_to_report): "not measured" zeros become blanks
    and the MedicalLogicFilter is applied. Otherwise the classifier could
    "win" purely on that data-cleaning gap rather than on genuine
    similarity/leakage. Blank numeric values are encoded as a 0/1 flag and
    blank categories as a "not measured" category.
    """
    real_df = align_real_to_report(raw_real_df, synth_df.columns)
    shared_cols = list(real_df.columns)
    if not shared_cols:
        raise ValueError("No shared columns between real and synthetic data.")

    synth_df = normalize_report(synth_df[shared_cols])
    real_df, synth_df = fill_missing([real_df, synth_df], real_df)
    shared_cols = list(real_df.columns)

    # Balance the two classes so the classifier can't win on class imbalance alone
    n = min(len(real_df), len(synth_df))
    real_df = real_df.sample(n=n, random_state=42)
    synth_df = synth_df.sample(n=n, random_state=42)

    real_df = real_df.copy()
    synth_df = synth_df.copy()
    real_df["is_real"] = 1
    synth_df["is_real"] = 0

    audit_data = pd.concat([real_df, synth_df]).sample(frac=1, random_state=42)

    # Encode remaining categorical columns (gender, chest_pain, origin_study, ...)
    cat_cols = [c for c in shared_cols if audit_data[c].dtype == object]
    audit_data = pd.get_dummies(audit_data, columns=cat_cols)

    X, y = audit_data.drop(columns=["is_real"]), audit_data["is_real"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(
        X_train, y_train
    )
    return round(accuracy_score(y_test, clf.predict(X_test)) * 100, 2)


def run_detectability_audit():
    """CLI entry point: loads main.py's real/synthetic outputs from disk and
    prints the detectability score."""
    cfg = config.active()
    real_path = cfg.data_file
    synth_path = os.path.join(cfg.output_dir, "grounded_synthetic_data_latest.csv")

    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        print("❌ Files missing. Run main.py first.")
        return

    raw_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)

    score = calculate_detectability(raw_df, synth_df)
    print(f"\n📊 Detectability: {score}%")
    print(
        "   (~50% = synthetic data is indistinguishable from real; "
        "closer to 100% = synthetic records are easy to tell apart from real ones)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.add_argument(parser)
    config.use(parser.parse_args().config)
    run_detectability_audit()
