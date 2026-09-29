import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# --- PATH INJECTION ---
# Ensures Streamlit can always find the 'core' and 'research' folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.ingestion import mark_missing
from core.metrics import calculate_fidelity_score
from research.adversarial_audit import calculate_privacy_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="MedTwin Portal", page_icon="🏥", layout="wide")

st.title("🏥 MedTwin: AI Clinical Trial Dashboard")
st.markdown("### Privacy-Preserving Synthetic Data Generation via WGAN-GP")
st.divider()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_PATH = os.path.join(BASE_DIR, "data", "heart_disease_uci.csv")
SYNTH_PATH = os.path.join(
    BASE_DIR, "data", "synthetic", "grounded_synthetic_data_latest.csv"
)
METRICS_PATH = os.path.join(BASE_DIR, "data", "synthetic", "latest_metrics.json")


@st.cache_data
def get_heldout_metrics(mtime):
    """Loads the held-out evaluation main.py writes after each run. `mtime`
    is only a cache key, so a new pipeline run is picked up automatically."""
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_data(synth_mtime):
    """`synth_mtime` is only a cache key, so a new pipeline run is picked up
    without restarting the dashboard. Real data has its "not measured" zeros
    (e.g. chol=0) turned into blanks so they don't distort the charts."""
    return mark_missing(pd.read_csv(REAL_PATH)), pd.read_csv(SYNTH_PATH)


@st.cache_data
def get_privacy_score(real_df, synth_df):
    """Runs the adversarial privacy audit (real vs. synthetic distinguishability).

    This is a property of the whole generation run, not of the current
    sidebar filter, so it's computed once against the full datasets and
    cached (keyed on their content) rather than recomputed on every filter
    change.
    """
    try:
        return calculate_privacy_score(real_df, synth_df)
    except Exception:
        return None


if os.path.exists(REAL_PATH) and os.path.exists(SYNTH_PATH):
    real_df, synth_df = get_data(os.path.getmtime(SYNTH_PATH))
else:
    real_df, synth_df = None, None

if synth_df is not None:
    # 1. METRICS & SIDEBAR FILTERS
    cont_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]

    st.sidebar.header("🎛️ Filter Digital Twins")
    gender_list = synth_df["gender"].unique().tolist()
    sel_gender = st.sidebar.multiselect("Gender", gender_list, default=gender_list)

    age_min, age_max = int(synth_df["age"].min()), int(synth_df["age"].max())
    sel_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

    # Apply filters
    display_df = synth_df[
        (synth_df["gender"].isin(sel_gender))
        & (synth_df["age"] >= sel_age[0])
        & (synth_df["age"] <= sel_age[1])
    ]

    # Calculate score based on filtered view
    score = calculate_fidelity_score(real_df, display_df, cont_cols)

    # Privacy score is a property of the full generation run (not the
    # current filter), so it's computed once against the unfiltered data.
    privacy_score = get_privacy_score(real_df, synth_df)

    if privacy_score is None:
        privacy_value, privacy_note = "N/A", "Audit failed to run"
    elif privacy_score <= 60:
        privacy_value, privacy_note = f"✅ {privacy_score}%", "High Privacy (Low Distinguishability)"
    elif privacy_score <= 75:
        privacy_value, privacy_note = f"⚠️ {privacy_score}%", "Moderate Privacy Risk"
    else:
        privacy_value, privacy_note = f"🚨 {privacy_score}%", "Low Privacy (Easily Distinguished)"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Digital Twins", len(display_df))
    m2.metric("Statistical Fidelity", f"{score}%")
    m3.metric("Avg Blood Pressure", f"{display_df['trestbps'].mean():.1f} mmHg")
    m4.metric("Privacy Audit Score", privacy_value, delta=privacy_note, delta_color="off")

    # 1b. HELD-OUT EVALUATION (from the last main.py run)
    st.subheader("🧪 Held-Out Evaluation")
    if os.path.exists(METRICS_PATH):
        heldout = get_heldout_metrics(os.path.getmtime(METRICS_PATH))
        utility, privacy = heldout["utility"], heldout["privacy"]
        st.caption(
            f"Measured against {privacy['n_per_real_set']} real patients the GAN never "
            f"saw during training (run {heldout['timestamp']}). Covers the full "
            "synthetic set, not the sidebar filter."
        )

        h1, h2, h3, h4 = st.columns(4)
        h1.metric(
            "Utility (TSTR AUC)",
            utility["tstr"]["auc"],
            delta=f"{utility['tstr']['auc'] - utility['trtr']['auc']:+.3f} vs real-trained",
            help="A disease classifier trained on synthetic data, tested on held-out "
            "real patients. Compared with the same classifier trained on real data.",
        )
        h2.metric(
            "Utility Ratio",
            f"{utility['auc_ratio']:.0%}",
            help="TSTR AUC / TRTR AUC. 100% = synthetic data is as useful as real data.",
        )
        h3.metric(
            "Closer to Train (DCR)",
            f"{privacy['dcr_closer_to_train_pct']}%",
            delta=f"{privacy['exact_copies_of_train']} exact copies",
            delta_color="off",
            help="Share of synthetic records whose nearest real record is a training "
            "patient rather than a held-out one. ~50% = no memorisation.",
        )
        h4.metric(
            "Membership Inference AUC",
            privacy["mia_auc"],
            delta="0.5 = no leakage",
            delta_color="off",
            help="How well an attacker holding the synthetic data can tell which real "
            "patients were in the training set. 0.5 = no better than a coin flip.",
        )
    else:
        st.info("No held-out evaluation yet. Run `python main.py` to generate it.")

    st.divider()

    # 2. VISUALIZATIONS (Left: Distributions, Right: Correlations)
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Distribution Fidelity")
        feat = st.selectbox("Select Feature to Compare", cont_cols)
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.kdeplot(
            real_df[feat].dropna(),
            label="Real Patients",
            fill=True,
            color="#3498db",
            ax=ax1,
        )
        sns.kdeplot(
            display_df[feat].dropna(),
            label="MedTwin AI",
            fill=True,
            color="#e67e22",
            ax=ax1,
        )
        plt.legend()
        st.pyplot(fig1)

    with col_right:
        st.subheader("🧬 Interactive Correlation Analysis")
        corr_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "num"]

        # Dropdown menu for the user to select the feature
        sel_corr_feat = st.selectbox("Select Feature to Analyze", corr_cols, index=5)

        # Calculate the correlations for just that specific feature
        real_corr = real_df[corr_cols].corr()[sel_corr_feat].drop(sel_corr_feat)
        synth_corr = display_df[corr_cols].corr()[sel_corr_feat].drop(sel_corr_feat)

        # Combine the data into a clean format for the chart
        corr_compare_df = pd.DataFrame(
            {
                "Feature": real_corr.index,
                "Real Patients": real_corr.values,
                "MedTwin AI": synth_corr.values,
            }
        ).melt(id_vars="Feature", var_name="Dataset", value_name="Correlation")

        # Plot the side-by-side bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=corr_compare_df,
            x="Feature",
            y="Correlation",
            hue="Dataset",
            palette=["#3498db", "#e67e22"],
            ax=ax2,
        )

        # Clean up the graph limits and add a zero-line
        ax2.set_ylabel(f"Correlation with '{sel_corr_feat}'")
        ax2.set_ylim(-1.0, 1.0)
        ax2.axhline(0, color="black", linewidth=1)
        plt.legend(title="Source")

        st.pyplot(fig2)

    # 3. DATA EXPLORER
    st.divider()
    st.subheader("📋 Raw Data Explorer")
    preview = display_df.head(50).copy()

    def show(v, col):
        # Blank cells mean "not measured". Columns holding blanks are read
        # as decimals, so whole numbers are shown without a trailing ".0".
        if pd.isna(v):
            return "not measured"
        if col == "oldpeak":
            return f"{v:.1f}"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    for col in preview.columns[preview.isna().any()]:
        preview[col] = preview[col].map(lambda v, c=col: show(v, c))
    st.dataframe(preview, use_container_width=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Filtered Batch",
        data=csv,
        file_name="medtwin_batch.csv",
        mime="text/csv",
    )

else:
    st.error("🚨 System Error: Please run 'python run.py' first to generate the data.")
