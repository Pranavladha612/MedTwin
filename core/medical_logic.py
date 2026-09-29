import pandas as pd

from core import config

# The plausible ranges, whole-number columns and decimal places all come from
# the active settings file (core/config.py). For heart disease the ranges
# cover every measured value in the real data (e.g. a resting BP of 200,
# cholesterol 603, max heart rate 60, oldpeak -2.6), so no real patient counts
# as "impossible" -- the filter only removes generated values no real
# patient could have.
#
# Earlier, narrower limits (BP 95-190, chol 110-500, HR 70-210) excluded 20
# real patients, and a rule "severity >= 2 implies oldpeak >= 0.5" was dropped
# because 55 of the 244 real patients with severity >= 2 (23%) break it.


class MedicalLogicFilter:
    def apply_rules(self, df):
        print("🛡️ Applying Clinical Reality Checks...")
        initial_count = len(df)

        # A blank means "not measured" (see core/ingestion.py), which is
        # valid, not an error.
        ranges = config.active().by_both_names(config.active().plausible_ranges)
        mask = pd.Series(True, index=df.index)
        for col, (lo, hi) in ranges.items():
            if col in df.columns:
                mask &= df[col].isna() | df[col].between(lo, hi)

        df = df[mask].copy()
        print(f"✅ Refined data: {len(df)}/{initial_count} records kept.")
        return df


def quantize_clinical(df):
    """Rounds generated values to the precision real records are recorded at
    (whole numbers unless the settings give decimal places) and clips
    whole-number scores and counts into their range. Nullable integers keep
    "not measured" values blank."""
    cfg = config.active()
    decimals = cfg.by_both_names(cfg.decimals)
    discrete = cfg.by_both_names(cfg.discrete_numeric)
    df = df.copy()
    for col in [cfg.report_name(c) for c in cfg.numeric]:
        if col not in df.columns:
            continue
        places = decimals.get(col, 0)
        if places:
            scale = 10 ** places
            df[col] = (df[col] * scale).round() / scale
            continue
        values = df[col].round()
        if col in discrete:
            values = values.clip(*discrete[col])
        if col in cfg.flagged_report or values.isna().any():
            df[col] = values.astype("Int64")  # nullable: keeps blanks blank
        else:
            df[col] = values.astype(int)
    return df
