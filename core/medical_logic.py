import pandas as pd

# Physiologically plausible ranges. Each covers every measured value in the
# real UCI data (e.g. a resting BP of 200, cholesterol 603, max heart rate 60,
# oldpeak -2.6), so no real patient counts as "impossible" -- the filter only
# removes generated values no real patient could have.
#
# Earlier, narrower limits (BP 95-190, chol 110-500, HR 70-210) excluded 20
# real patients, and a rule "severity >= 2 implies oldpeak >= 0.5" was dropped
# because 55 of the 244 real patients with severity >= 2 (23%) break it.
PLAUSIBLE_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (80, 650),
    "thalch": (50, 220),
    # Negative oldpeak (ST elevation rather than depression) occurs in 12 real
    # patients across severity levels, so it is kept as a genuine reading.
    "oldpeak": (-3.0, 7.0),
}


class MedicalLogicFilter:
    def apply_rules(self, df):
        print("🛡️ Applying Clinical Reality Checks...")
        initial_count = len(df)

        # A blank means "not measured" (see core/ingestion.py), which is
        # valid, not an error.
        mask = pd.Series(True, index=df.index)
        for col, (lo, hi) in PLAUSIBLE_RANGES.items():
            if col in df.columns:
                mask &= df[col].isna() | df[col].between(lo, hi)

        df = df[mask].copy()
        print(f"✅ Refined data: {len(df)}/{initial_count} records kept.")
        return df


def quantize_clinical(df):
    """Rounds generated values to the precision real records are recorded at
    (whole-number vitals and vessel counts, oldpeak to 0.1, integer disease
    severity). Nullable integers keep "not measured" values blank."""
    df = df.copy()
    df["age"] = df["age"].round().astype(int)
    for col in ["trestbps", "chol", "thalch"]:
        df[col] = df[col].round().astype("Int64")
    if "ca" in df.columns:
        df["ca"] = df["ca"].round().clip(0, 3).astype("Int64")
    df["num"] = df["num"].round().clip(0, 4).astype(int)
    df["oldpeak"] = (df["oldpeak"] * 10).round() / 10
    return df
