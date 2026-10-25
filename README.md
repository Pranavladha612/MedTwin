# MedTwin: Clinically-Grounded Synthetic EHR Generation

MedTwin is a framework for generating clinically realistic, privacy-preserving synthetic Electronic Health Record (EHR) data. Unlike standard tabular generators that either impute or drop missing values, MedTwin treats **informative missingness**—the clinical reality that diagnostic tests are selectively ordered based on hospital protocols and patient risk—as a core signal to reproduce and preserve.

---

## ✨ Key Features

- **Informative Missingness Modeling**: Accurately learns hospital-specific test ordering patterns and preserves missingness indicators without naive imputation.
- **Clinically-Grounded Architecture**: Conditional WGAN-GP generator with Gumbel-Softmax categorical outputs and correlation-gap loss penalties.
- **Privacy Safeguards**: Rigorously audited against re-identification and membership inference (detectability / adversary classifier metrics).
- **Multi-Dataset Benchmarking**: Evaluated across 5 random seeds against CTGAN, TVAE, and TabDDPM on UCI Heart Disease (multicenter) and Pima Indian Diabetes.
- **Interactive Clinical Dashboard**: Built-in Streamlit app for real-time visualization of distributions, correlation preservation, and held-out evaluation metrics.
- **Full Research Artifacts**: Includes compiled Springer LNCS research paper, project reports, and reproduction scripts.

---

## 📁 Repository Structure

```
├── app/
│   └── dashboard.py               # Streamlit evaluation & inspection dashboard
├── configs/
│   ├── heart_disease.yaml         # Dataset-specific settings for UCI Heart Disease
│   └── diabetes.yaml              # Dataset-specific settings for Pima Diabetes
├── core/
│   ├── generator.py               # MedTwin conditional GAN generator & trainer
│   ├── evaluation.py              # Fidelity, correlation gap, & privacy evaluation
│   ├── ingestion.py               # Preprocessing & informative missingness encoding
│   ├── medtwin.py                 # Pipeline orchestrator
│   └── config.py                  # Configuration loader
├── data/                          # Real & generated datasets, benchmark metrics
├── docs/                          # Comprehensive reports, presentation deck & figures
├── paper/                         # Springer LNCS LaTeX paper source & compiled PDF
├── research/                      # Ablation studies, benchmarks, and table generation
├── main.py                        # Training & generation entry point
├── run.py                         # One-click launch script (train/generate + dashboard)
└── requirements.txt               # Python package dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install the dependencies in a virtual environment:

```bash
git clone https://github.com/Pranavladha612/MedTwin.git
cd MedTwin

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

Run generation and launch the interactive Streamlit dashboard with a single command:

```bash
python run.py
```

Or run individual components:

```bash
# Run training and generation on default config (Heart Disease)
python main.py

# Run on Diabetes config
python main.py --config configs/diabetes.yaml

# Launch Streamlit dashboard directly
streamlit run app/dashboard.py
```

### 3. Run Benchmark and Ablation Experiments

```bash
# Run ablation experiments across 5 seeds
python research/ablation.py

# Generate LaTeX paper tables
python research/paper_tables.py
```

---

## 📊 Benchmark Results Summary

Across 5 seeds on held-out test splits, MedTwin significantly outperforms standard generators in correlation preservation and missingness fidelity while maintaining strict privacy boundaries (detectability ROC-AUC ~ 0.50):

| Generator | Missingness Fidelity | Correlation Gap | Privacy Audit (Detectability) |
|-----------|----------------------|-----------------|-------------------------------|
| **MedTwin** | **High** | **Lowest** | **Private (~0.50)** |
| CTGAN | Low | Moderate | Moderate |
| TVAE | Moderate | High | Moderate |
| TabDDPM | High | High | Private |

*(Detailed tables and exact figures are in `paper/MedTwin_paper.pdf` and `paper/tables/`)*

---

## 📄 Research Paper & Documentation

- **Springer LNCS Paper**: Located in [`paper/MedTwin_paper.pdf`](paper/MedTwin_paper.pdf) (LaTeX sources in `paper/`).
- **Comprehensive Academic Report**: Located in [`docs/MedTwin_Project_Report_MUJ.pdf`](docs/MedTwin_Project_Report_MUJ.pdf).
- **Presentation Deck**: Slide notes and visual layouts in [`docs/presentation/`](docs/presentation/).

---

## 📜 License

This project is licensed under the MIT License.
