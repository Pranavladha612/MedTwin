import hashlib
import json
import os
import sys
from datetime import datetime

import pandas as pd
import torch

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from core.evaluation import (
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    evaluate_synthetic,
    split_real_data,
)
from core import medtwin
from core.generator import load_model, save_model
from core.ingestion import mark_missing

# Bump when preprocessing changes, so a model trained on the old
# preprocessing is never reused (the CSV hash alone wouldn't notice).
PREP_VERSION = 2  # v2: zeros in chol/trestbps treated as "not measured"
# Same idea for the training code in core/generator.py.
TRAIN_VERSION = 2  # v2: correlation loss skips constant columns (NaN fix)
from core.metrics import calculate_fidelity_score
from research.adversarial_audit import calculate_privacy_score


def get_data_hash(file_path):
    """Creates a unique fingerprint of the CSV to prevent redundant training."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def run_pipeline():
    # 1. SETUP PATHS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = os.path.join(BASE_DIR, "data", "heart_disease_uci.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "core", "trained_gen.pth")
    HASH_FILE = os.path.join(BASE_DIR, "core", "last_training_hash.txt")
    SYNTH_DIR = os.path.join(BASE_DIR, "data", "synthetic")

    if not os.path.exists(SYNTH_DIR):
        os.makedirs(SYNTH_DIR)

    # 2. DATA INGESTION
    print("--- Phase 1: Ingesting Real Medical Data ---")
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: {FILE_PATH} not found!")
        return

    raw_df = pd.read_csv(FILE_PATH)

    # Hold out real patients the GAN never sees, so utility and privacy can be
    # measured against unseen data (see core/evaluation.py).
    train_raw, holdout_raw = split_real_data(raw_df)
    print(f"Split: {len(train_raw)} train / {len(holdout_raw)} held-out patients")

    ingestor, processed_df = medtwin.prepare(train_raw)

    # 3. TRAINING LOGIC (IDENTITY GATEKEEPER)
    # The split settings and preprocessing version are part of the
    # fingerprint: a model trained differently must not be reused.
    current_hash = (
        f"{get_data_hash(FILE_PATH)}|holdout={HOLDOUT_FRACTION}|seed={SPLIT_SEED}"
        f"|prep=v{PREP_VERSION}|train=v{TRAIN_VERSION}"
    )
    trained_gen = None
    if os.path.exists(HASH_FILE) and os.path.exists(MODEL_PATH):
        with open(HASH_FILE, "r") as f:
            if f.read() == current_hash:
                print("⚠️ DATA UNCHANGED: Loading Boosted AI Model...")
                trained_gen = load_model(medtwin.new_generator(processed_df), MODEL_PATH)

    if trained_gen is None:
        print("🚀 NEW DATA: retraining.")
        trained_gen = medtwin.train(processed_df)
        save_model(trained_gen, MODEL_PATH)
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)

    # 4-7. GENERATION, CLINICAL FILTERING, ROUNDING, READABLE CATEGORIES
    final_report = medtwin.generate_report(trained_gen, ingestor, processed_df.columns)

    # 8. SAVE OUTPUTS (HISTORY + LATEST)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dated_filename = f"synthetic_report_{timestamp}.csv"

    dated_path = os.path.join(SYNTH_DIR, dated_filename)
    latest_path = os.path.join(SYNTH_DIR, "grounded_synthetic_data_latest.csv")

    final_report.to_csv(dated_path, index=False)
    final_report.to_csv(latest_path, index=False)

    # 9. FINAL VALIDATION
    # Compared against real data with "not measured" zeros removed, otherwise
    # the 172 fake zero cholesterol readings distort the real distribution.
    fidelity_score = calculate_fidelity_score(
        mark_missing(raw_df), final_report, ["age", "trestbps", "chol", "thalch"]
    )

    # 10. PRIVACY VALIDATION
    # Trains a classifier to distinguish real patients from synthetic ones.
    # ~50% = indistinguishable (good privacy); closer to 100% = the
    # generator is leaking/overfitting to real records (bad privacy).
    print("🕵️ Running Adversarial Privacy Audit...")
    try:
        privacy_score = calculate_privacy_score(raw_df, final_report)
    except Exception as e:
        print(f"⚠️ Privacy audit failed to run: {e}")
        privacy_score = None

    # 11. HELD-OUT EVALUATION (utility + memorisation checks)
    print("🧪 Running Held-Out Evaluation...")
    evaluation = evaluate_synthetic(train_raw, holdout_raw, final_report)
    metrics = {
        "timestamp": timestamp,
        "fidelity_pct": fidelity_score,
        "discriminator_pct": privacy_score,
        **evaluation,
    }
    with open(os.path.join(SYNTH_DIR, "latest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    utility, privacy = evaluation["utility"], evaluation["privacy"]

    print(f"\n🎉 PIPELINE SUCCESS!")
    print(f"📊 Statistical Fidelity: {fidelity_score}%")
    if privacy_score is not None:
        print(f"🔒 Privacy Audit (real vs. synthetic distinguishability): {privacy_score}%")
        print("   (~50% = strong privacy; closer to 100% = data may be leaking real patients)")
    print("\n🧪 Held-out evaluation (real patients the GAN never saw):")
    print(
        f"   Utility  TRTR AUC {utility['trtr']['auc']} | TSTR AUC {utility['tstr']['auc']}"
        f" | ratio {utility['auc_ratio']} (1.0 = synthetic as useful as real)"
    )
    print(
        f"   DCR      {privacy['dcr_closer_to_train_pct']}% of synthetic records closer to"
        f" train than holdout (~50% = no memorisation)"
    )
    print(
        f"   MIA      AUC {privacy['mia_auc']} (0.5 = attacker can't identify training"
        f" patients) | exact copies of train: {privacy['exact_copies_of_train']}"
    )
    print(f"📄 Saved Archive: {dated_filename}")


if __name__ == "__main__":
    run_pipeline()
