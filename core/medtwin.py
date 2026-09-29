"""MedTwin's train-and-generate steps, shared by main.py (which adds model
caching and saves the outputs) and research/multi_seed.py (which retrains
from scratch for each seed)."""

import numpy as np
import pandas as pd
import torch

from core import config
from core.generator import Generator, train_wgan_gp
from core.ingestion import NOT_MEASURED, DataIngestion, missing_flag, zero_flag
from core.medical_logic import MedicalLogicFilter, quantize_clinical

LATENT_DIM = 128
EPOCHS = 600
N_SAMPLES = 2000


def prepare(train_raw):
    """Ingests the real train split. Returns the fitted ingestor (needed to
    build, sample and un-scale the model) and the processed training frame."""
    ingestor = DataIngestion()
    processed_df = ingestor.process_data(ingestor.de_identify(train_raw.copy()))
    return ingestor, processed_df


def new_generator(ingestor):
    return Generator(LATENT_DIM, len(ingestor.site_names), ingestor.output_info)


def train(ingestor, processed_df):
    print(f"🚀 Training Boosted GAN ({EPOCHS} Epochs)...")
    return train_wgan_gp(
        torch.FloatTensor(processed_df.values),
        torch.FloatTensor(ingestor.cond_df.values),
        ingestor.output_info,
        epochs=EPOCHS,
        latent_dim=LATENT_DIM,
    )


def generate_report(generator, ingestor, n=N_SAMPLES):
    """Samples the generator and turns its output into the clinical report:
    sites drawn in their real proportions, real units, blanks for anything
    the model marks as not measured, plausibility filtering, rounding, and
    readable category columns. Blank cells mean "not measured"."""
    print("\n--- Phase 4: Generating Synthetic Population ---")
    cfg = config.active()

    # Each synthetic patient is generated for a site, drawn with the same
    # frequencies as the real training patients.
    site_idx = np.random.choice(len(ingestor.site_names), size=n, p=ingestor.site_probs)
    cond = torch.zeros(n, len(ingestor.site_names))
    cond[torch.arange(n), torch.as_tensor(site_idx)] = 1.0

    generator.eval()
    with torch.no_grad():
        synth_raw = generator(torch.randn(n, LATENT_DIM), cond, hard=True).numpy()
    out = pd.DataFrame(synth_raw, columns=ingestor.columns)

    report = pd.DataFrame(
        ingestor.scaler.inverse_transform(out[ingestor.num_cols]),
        columns=ingestor.num_cols,
    )
    for col in cfg.exactly_zero_spike:
        report.loc[out[zero_flag(col)].values >= 0.5, col] = 0.0
    for col in cfg.flagged:  # after zero flags: "not measured" wins
        report.loc[out[missing_flag(col)].values >= 0.5, col] = np.nan
    report = report.rename(columns=cfg.rename)

    for col, group in ingestor.cat_groups.items():
        labels = out[group].idxmax(axis=1).str[len(col) + 1:]
        report[cfg.report_name(col)] = labels.where(labels != NOT_MEASURED).values
    if cfg.condition_on:
        report[cfg.report_name(cfg.condition_on)] = [ingestor.site_names[i] for i in site_idx]

    report = MedicalLogicFilter().apply_rules(report)
    print("🎯 Quantizing values for High Statistical Fidelity...")
    report = quantize_clinical(report)
    print("🧹 Cleaning columns for Clinical Report...")
    return report[cfg.report_columns]
