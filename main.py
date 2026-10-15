import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime

import pandas as pd

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
from core import config, medtwin
from core.generator import load_model, save_model
from core.ingestion import mark_missing

# Bump when preprocessing changes, so a model trained on the old
# preprocessing is never reused (the CSV hash alone wouldn't notice).
# v2: zeros in chol/trestbps treated as "not measured"
# v3: all 15 columns, "not measured" for every column, site as condition
# v4: "exactly zero" flag for oldpeak
PREP_VERSION = 4
# Same idea for the training code in core/generator.py.
# v2: correlation loss skips constant columns (NaN fix)
# v3: conditional generator with Gumbel-softmax categorical outputs
TRAIN_VERSION = 3

# Dated reports kept in the dataset's output folder; older ones are deleted
# on each run.
KEEP_REPORTS = 10

from core.metrics import calculate_fidelity_score
from research.adversarial_audit import calculate_detectability


def get_data_hash(file_path):
    """Creates a unique fingerprint of the CSV to prevent redundant training."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def run_pipeline():
    # 1. SETUP PATHS (all from the dataset's settings file)
    cfg = config.active()
    FILE_PATH = cfg.data_file
    MODEL_PATH = cfg.model_file
    HASH_FILE = os.path.splitext(MODEL_PATH)[0] + ".hash.txt"
    SYNTH_DIR = cfg.output_dir

    os.makedirs(SYNTH_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 2. DATA INGESTION
    print(f"--- Phase 1: Ingesting Real Medical Data ({cfg.name}) ---")
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
    # The split, the preprocessing and training versions and the settings
    # file are all part of the fingerprint: a model trained differently must
    # not be reused.
    current_hash = (
        f"{get_data_hash(FILE_PATH)}|holdout={HOLDOUT_FRACTION}|seed={SPLIT_SEED}"
        f"|prep=v{PREP_VERSION}|train=v{TRAIN_VERSION}|settings={get_data_hash(cfg.path)}"
    )
    trained_gen = None
    if os.path.exists(HASH_FILE) and os.path.exists(MODEL_PATH):
        with open(HASH_FILE, "r") as f:
            if f.read() == current_hash:
                print("⚠️ DATA UNCHANGED: Loading Boosted AI Model...")
                trained_gen = load_model(medtwin.new_generator(ingestor), MODEL_PATH)

    if trained_gen is None:
        print("🚀 NEW DATA: retraining.")
        trained_gen = medtwin.train(ingestor, processed_df)
        save_model(trained_gen, MODEL_PATH)
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)

    # 4-7. GENERATION, CLINICAL FILTERING, ROUNDING, READABLE CATEGORIES
    final_report = medtwin.generate_report(trained_gen, ingestor)

    # 8. SAVE OUTPUTS (HISTORY + LATEST)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dated_filename = f"synthetic_report_{timestamp}.csv"

    dated_path = os.path.join(SYNTH_DIR, dated_filename)
    latest_path = os.path.join(SYNTH_DIR, "grounded_synthetic_data_latest.csv")

    final_report.to_csv(dated_path, index=False)
    final_report.to_csv(latest_path, index=False)

    # Timestamped names sort by date, so everything before the newest
    # KEEP_REPORTS is old. The "latest" copy above is always kept.
    dated_reports = sorted(glob.glob(os.path.join(SYNTH_DIR, "synthetic_report_*.csv")))
    for old in dated_reports[:-KEEP_REPORTS]:
        os.remove(old)

    # 9. FINAL VALIDATION
    # Compared against real data with "not measured" zeros removed, otherwise
    # zero placeholders (172 cholesterol readings for heart disease) distort
    # the real distribution.
    fidelity_score = calculate_fidelity_score(
        mark_missing(raw_df).rename(columns=cfg.rename), final_report, cfg.fidelity_report
    )

    # 10. DETECTABILITY
    # Trains a classifier to tell real patients from synthetic ones.
    # ~50% = indistinguishable; closer to 100% = easy to spot. A realism
    # check, not a privacy one (copied patients would score 50%); privacy is
    # the DCR and membership-inference part of the held-out evaluation.
    print("🕵️ Running Real-vs-Synthetic Detectability Test...")
    try:
        detectability = calculate_detectability(raw_df, final_report)
    except Exception as e:
        print(f"⚠️ Detectability test failed to run: {e}")
        detectability = None

    # 11. HELD-OUT EVALUATION (utility + memorisation checks)
    print("🧪 Running Held-Out Evaluation...")
    evaluation = evaluate_synthetic(train_raw, holdout_raw, final_report)
    metrics = {
        "timestamp": timestamp,
        # Patients removed by the plausibility filter: shows whether the
        # generator itself respects the clinical ranges.
        "generated": medtwin.N_SAMPLES,
        "kept_after_filter": len(final_report),
        "fidelity_pct": fidelity_score,
        "detectability_pct": detectability,
        **evaluation,
    }
    with open(os.path.join(SYNTH_DIR, "latest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    utility, privacy = evaluation["utility"], evaluation["privacy"]

    print(f"\n🎉 PIPELINE SUCCESS!")
    print(f"📊 Statistical Fidelity: {fidelity_score}%")
    if detectability is not None:
        print(f"🔍 Detectability (real-vs-synthetic classifier accuracy): {detectability}%")
        print("   (~50% = indistinguishable from real; closer to 100% = easy to spot)")
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
    print(
        f"   Corr gap {evaluation['correlation']['corr_gap']} all features |"
        f" {evaluation['correlation']['corr_gap_numeric']} numeric (0 = same correlations)"
    )
    gaps, leaks = evaluation["missingness"], evaluation["missingness_privacy"]
    if gaps:
        print(
            f"   Missing  pattern TVD {gaps['pattern_tvd']} (real-vs-real {gaps['pattern_tvd_ref']})"
            f" | rare-pattern MIA AUC {leaks['rare_pattern_dist_mia_auc']}"
            f" | unique-pattern leak {leaks['unique_pattern_leak_pct']} pts"
        )
    print(f"📄 Saved Archive: {dated_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedTwin and generate synthetic patients.")
    config.add_argument(parser)
    config.use(parser.parse_args().config)
    run_pipeline()
