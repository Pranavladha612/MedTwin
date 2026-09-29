import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def compare_distributions(real_path, synthetic_path):
    # 1. Load Data
    real = pd.read_csv(real_path)
    synth = pd.read_csv(synthetic_path)

    # 2. Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Age Distribution
    sns.kdeplot(real["age"], ax=axes[0], label="Real", fill=True)
    sns.kdeplot(synth["age"], ax=axes[0], label="Synthetic", fill=True)
    axes[0].set_title("Age Distribution: Real vs Synthetic")
    axes[0].legend()

    # Plot Blood Pressure (trestbps) Distribution
    sns.kdeplot(real["trestbps"], ax=axes[1], label="Real", fill=True)
    sns.kdeplot(synth["trestbps"], ax=axes[1], label="Synthetic", fill=True)
    axes[1].set_title("Blood Pressure: Real vs Synthetic")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("distribution_comparison.png")
    print("📈 Comparison plot saved as 'distribution_comparison.png'")
    plt.show()


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # main.py's real outputs are the raw source CSV and the "_latest" grounded report.
    REAL = os.path.join(BASE_DIR, "data", "heart_disease_uci.csv")
    SYNTH = os.path.join(
        BASE_DIR, "data", "synthetic", "grounded_synthetic_data_latest.csv"
    )
    compare_distributions(REAL, SYNTH)
