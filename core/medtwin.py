"""MedTwin's train-and-generate steps, shared by main.py (which adds model
caching and saves the outputs), research/multi_seed.py (which retrains from
scratch for each seed) and research/ablation.py (which retrains with one
component switched off at a time)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from core import config
from core.generator import CORR_WEIGHT, Generator, train_wgan_gp
from core.ingestion import NOT_MEASURED, DataIngestion, missing_flag, zero_flag
from core.medical_logic import MedicalLogicFilter, quantize_clinical

LATENT_DIM = 128
EPOCHS = 600
N_SAMPLES = 2000


@dataclass(frozen=True)
class Variant:
    """Which MedTwin components are switched on. The default is the full
    model; each ablation switches exactly one component off.

    - corr_loss: the correlation penalty in the generator's loss.
    - missing_flags: a "was it measured?" flag per sometimes-missing numeric
      column. Off = blanks filled with the median and never generated (the
      usual imputation approach). Missing categories stay a category.
    - zero_flags: the "exactly zero" flag for spiked columns (oldpeak).
    - condition: generate each patient for a given site. Off = the site is
      generated like any other categorical column.
    - filter: drop generated patients outside the plausible ranges.
    """

    corr_loss: bool = True
    missing_flags: bool = True
    zero_flags: bool = True
    condition: bool = True
    filter: bool = True


FULL = Variant()

ABLATIONS = {
    "full": FULL,
    "no_corr_loss": Variant(corr_loss=False),
    "no_missing_flags": Variant(missing_flags=False),
    "no_zero_flags": Variant(zero_flags=False),
    "no_condition": Variant(condition=False),
    "no_filter": Variant(filter=False),
}


def applicable_ablations():
    """The ablations that change anything for the active dataset: there is no
    zero flag to remove without spiked columns, and no conditioning to remove
    without a condition column."""
    cfg = config.active()
    skip = set()
    if not cfg.exactly_zero_spike:
        skip.add("no_zero_flags")
    if not cfg.condition_on:
        skip.add("no_condition")
    if not cfg.flagged:
        skip.add("no_missing_flags")
    return {name: v for name, v in ABLATIONS.items() if name not in skip}


def prepare(train_raw, variant=FULL):
    """Ingests the real train split. Returns the fitted ingestor (needed to
    build, sample and un-scale the model; it also carries the variant) and the
    processed training frame."""
    ingestor = DataIngestion(variant)
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
        corr_weight=CORR_WEIGHT if ingestor.variant.corr_loss else 0.0,
    )


def generate_report(generator, ingestor, n=N_SAMPLES):
    """Samples the generator and turns its output into the clinical report:
    sites drawn in their real proportions, real units, blanks for anything
    the model marks as not measured, plausibility filtering, rounding, and
    readable category columns. Blank cells mean "not measured"."""
    print("\n--- Phase 4: Generating Synthetic Population ---")
    cfg = config.active()

    # Each synthetic patient is generated for a site, drawn with the same
    # frequencies as the real training patients. (Unconditioned, there is a
    # single "all" site.)
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
    for col in ingestor.zero_cols:
        report.loc[out[zero_flag(col)].values >= 0.5, col] = 0.0
    for col in ingestor.flagged_cols:  # after zero flags: "not measured" wins
        report.loc[out[missing_flag(col)].values >= 0.5, col] = np.nan
    report = report.rename(columns=cfg.rename)

    for col, group in ingestor.cat_groups.items():
        labels = out[group].idxmax(axis=1).str[len(col) + 1:]
        report[cfg.report_name(col)] = labels.where(labels != NOT_MEASURED).values
    if ingestor.conditioned:
        report[cfg.report_name(cfg.condition_on)] = [ingestor.site_names[i] for i in site_idx]

    if ingestor.variant.filter:
        report = MedicalLogicFilter().apply_rules(report)
    print("🎯 Quantizing values for High Statistical Fidelity...")
    report = quantize_clinical(report)
    print("🧹 Cleaning columns for Clinical Report...")
    return report[cfg.report_columns]
