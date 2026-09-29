import numpy as np
import pandas as pd


class MedicalLogicFilter:
    def apply_rules(self, df):
        print("🛡️ Applying Clinical Reality Checks...")
        initial_count = len(df)

        # 1. Biological Limits
        mask_age = (df["age"] >= 25) & (df["age"] <= 85)
        # Blood Pressure: 60 is too low for this study, 95-190 is realistic
        mask_bp = (df["trestbps"] >= 95) & (df["trestbps"] <= 190)
        # Cholesterol: Must be above 100 to be realistic. A blank means "not
        # measured" (see core/ingestion.py), which is valid, not an error.
        mask_chol = df["chol"].isna() | ((df["chol"] >= 110) & (df["chol"] <= 500))
        # Heart Rate: Max heart rate should be at least 70
        mask_hr = (df["thalch"] >= 70) & (df["thalch"] <= 210)

        # 2. Logic Consistency
        # If disease severity (num) is high, oldpeak shouldn't be 0
        mask_consistency = ~((df["num"] >= 2) & (df["oldpeak"] < 0.5))

        df = df[mask_age & mask_bp & mask_chol & mask_hr & mask_consistency].copy()

        # 3. Rounding Age to whole years for final output
        df["age"] = df["age"].round().astype(int)

        print(f"✅ Refined data: {len(df)}/{initial_count} records kept.")
        return df


def quantize_clinical(df):
    """Rounds generated values to the precision real records are recorded at
    (whole-number vitals, oldpeak to 0.1, integer disease severity)."""
    df = df.copy()
    for col in ["age", "trestbps", "thalch"]:
        df[col] = df[col].round().astype(int)
    # Nullable integer: keeps "not measured" cholesterol blank
    df["chol"] = df["chol"].round().astype("Int64")
    df["num"] = df["num"].round().clip(0, 4).astype(int)
    df["oldpeak"] = df["oldpeak"].apply(lambda x: round(x * 10) / 10).clip(0.0, 6.0)
    return df
