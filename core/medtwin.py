"""MedTwin's train-and-generate steps, shared by main.py (which adds model
caching and saves the outputs) and research/multi_seed.py (which retrains
from scratch for each seed)."""

import pandas as pd
import torch

from core.generator import Generator, train_dp_gan
from core.ingestion import DataIngestion, blank_flagged
from core.medical_logic import MedicalLogicFilter, quantize_clinical

LATENT_DIM = 128
EPOCHS = 600
N_SAMPLES = 2000  # generated before filtering, to ensure plenty pass

REPORT_COLS = [
    "age", "gender", "chest_pain", "trestbps", "chol",
    "thalch", "oldpeak", "num", "origin_study",
]

BINARY_PREFIXES = [
    "sex_", "fbs_", "exang_", "cp_", "dataset_", "restecg_", "slope_", "thal_",
]


def prepare(train_raw):
    """Ingests the real train split. Returns the fitted ingestor (needed to
    undo the scaling later) and the processed training frame."""
    ingestor = DataIngestion()
    processed_df = ingestor.process_data(ingestor.de_identify(train_raw.copy()))
    return ingestor, processed_df


def new_generator(processed_df):
    return Generator(LATENT_DIM, processed_df.shape[1])


def train(processed_df):
    print(f"🚀 Training Boosted GAN ({EPOCHS} Epochs)...")
    data_tensor = torch.FloatTensor(processed_df.values)
    return train_dp_gan(data_tensor, epochs=EPOCHS, latent_dim=LATENT_DIM)


def generate_report(generator, ingestor, columns, n=N_SAMPLES):
    """Samples the generator and turns its output into the clinical report:
    real units, rounded binaries, blank unmeasured values, clinical
    filtering, rounding, and readable category columns."""
    print("\n--- Phase 4: Generating Synthetic Population ---")
    generator.eval()
    with torch.no_grad():
        synth_raw = generator(torch.randn(n, LATENT_DIM)).numpy()
    synth_df = pd.DataFrame(synth_raw, columns=columns)

    # Convert decimals back to Real Units
    synth_df[ingestor.num_cols] = ingestor.scaler.inverse_transform(
        synth_df[ingestor.num_cols]
    )

    # Clean Binary columns (0 or 1)
    binary_cols = [c for c in synth_df.columns if any(p in c for p in BINARY_PREFIXES)]
    synth_df[binary_cols] = synth_df[binary_cols].round().astype(int)

    # Patients the model marks as "cholesterol not measured" get a blank
    # value, like the real records, instead of an invented number.
    synth_df = blank_flagged(synth_df)

    # Clinical filtering, then rounding to recorded precision
    grounded_df = MedicalLogicFilter().apply_rules(synth_df)
    print("🎯 Quantizing values for High Statistical Fidelity...")
    final_df = quantize_clinical(grounded_df)

    # Human-friendly category reconstruction
    print("🧹 Cleaning columns for Clinical Report...")
    if "sex_Male" in final_df.columns:
        final_df["gender"] = final_df["sex_Male"].apply(
            lambda x: "Male" if x == 1 else "Female"
        )
    cp_cols = [c for c in final_df.columns if "cp_" in c]
    if cp_cols:
        final_df["chest_pain"] = final_df[cp_cols].idxmax(axis=1).str.replace("cp_", "")
    ds_cols = [c for c in final_df.columns if "dataset_" in c]
    if ds_cols:
        final_df["origin_study"] = (
            final_df[ds_cols].idxmax(axis=1).str.replace("dataset_", "")
        )

    return final_df[[c for c in REPORT_COLS if c in final_df.columns]]
