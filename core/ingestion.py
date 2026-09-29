import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Label used for a categorical value that was never recorded. In the model it
# is an ordinary category; in the output report it becomes a blank cell.
NOT_MEASURED = "not measured"

# The source hospitals recorded "not measured" as 0 in these columns (e.g.
# Switzerland never measured cholesterol: all 123 of its patients have chol=0).
# A 0 here is never a real reading, so it is treated as missing.
ZERO_MEANS_MISSING = ["trestbps", "chol"]

# Numeric columns that some sites never recorded (e.g. VA Long Beach lacks
# blood pressure, max heart rate and oldpeak for ~28% of patients; 'ca' is
# missing for 66% of all patients). Each gets a 0/1 "<col>_missing" flag so
# the model learns *whether* the value was measured, and generated records with
# the flag set get a blank instead of an invented number.
FLAGGED_COLS = ["trestbps", "chol", "thalch", "oldpeak", "ca"]

# Numeric columns where a large share of real values are exactly 0 (43% of
# measured oldpeak). A smooth Sigmoid output can't hit a single value, so it
# smears the spike across roughly -0.9..0.9 instead. A 0/1 "<col>_zero" flag
# lets the model choose "exactly 0" as a yes/no decision.
ZERO_INFLATED = ["oldpeak"]

# Every numeric column gets MinMax-scaled to [0,1], because the Generator's
# numeric outputs are Sigmoids (core/generator.py) and can only produce values
# in [0,1]. A numeric column missing from this list would be silently capped
# at 1 ('num', severity 0-4, once was -- the model could never output num >= 2).
NUM_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "num"]

# Categorical columns the model generates, each as one softmax group.
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# The hospital site is not generated: the model is *conditioned* on it, so each
# synthetic patient is generated for a given site and site proportions match
# the real data exactly.
CONDITION_COL = "dataset"


def missing_flag(col):
    return f"{col}_missing"


def zero_flag(col):
    return f"{col}_zero"


def mark_missing(df):
    """Replaces 'not measured' zeros with NaN. Safe to call more than once."""
    df = df.copy()
    for col in ZERO_MEANS_MISSING:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
    return df


def as_category(series):
    """Categorical values as strings (so True and "True" match), with blanks
    turned into the NOT_MEASURED category."""
    return series.map(lambda v: NOT_MEASURED if pd.isna(v) else str(v))


def add_missing_flags(df, fill_values):
    """For models and metrics that can't take blanks: each flagged numeric
    column gets a 0/1 missing flag and its blanks filled with
    `fill_values[col]`; blank categorical (object) values become NOT_MEASURED.
    The flag keeps "not measured" distinguishable from the filled value."""
    df = df.copy()
    for col in FLAGGED_COLS:
        if col in df.columns:
            df[missing_flag(col)] = df[col].isna().astype(int)
            df[col] = df[col].astype(float).fillna(fill_values[col])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = as_category(df[col])
    return df


def blank_flagged(df):
    """Inverse of add_missing_flags: blanks numeric values whose flag is set
    (dropping the flag columns) and NOT_MEASURED categories."""
    df = df.copy()
    for col in FLAGGED_COLS:
        flag = missing_flag(col)
        if flag in df.columns:
            df.loc[df[flag].astype(float).round() == 1, col] = np.nan
            df = df.drop(columns=[flag])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].where(df[col] != NOT_MEASURED)
    return df


class DataIngestion:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_cols = NUM_COLS
        self.cat_cols = CAT_COLS

    def de_identify(self, df):
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        return df

    def process_data(self, df):
        """Turns real records into the GAN's training matrix.

        Column order: scaled numerics, then missing flags (together the
        Sigmoid outputs), then one one-hot block per categorical column (each
        a softmax output). Also records what the Generator needs to be built
        and sampled: `output_info`, `cat_groups`, the site condition matrix
        `cond_df` and the training site frequencies `site_probs`.
        """
        df = mark_missing(df)

        flags = pd.DataFrame(
            {missing_flag(c): df[c].isna().astype(float) for c in FLAGGED_COLS},
            index=df.index,
        )
        for col in ZERO_INFLATED:
            flags[zero_flag(col)] = (df[col] == 0).astype(float)  # blanks count as not zero

        # A flagged-missing value still needs *some* number in the matrix; the
        # median is a placeholder the flag marks as meaningless.
        nums = pd.DataFrame(
            self.num_imputer.fit_transform(df[self.num_cols]),
            columns=self.num_cols,
            index=df.index,
        )
        nums[self.num_cols] = self.scaler.fit_transform(nums[self.num_cols])

        blocks = [nums, flags]
        self.cat_groups = {}
        for col in self.cat_cols:
            dummies = pd.get_dummies(as_category(df[col]), prefix=col, prefix_sep="_")
            self.cat_groups[col] = list(dummies.columns)
            blocks.append(dummies)
        processed = pd.concat(blocks, axis=1).astype("float32")

        self.output_info = [("sigmoid", nums.shape[1] + flags.shape[1])] + [
            ("softmax", len(cols)) for cols in self.cat_groups.values()
        ]

        sites = df[CONDITION_COL].astype(str)
        self.cond_df = pd.get_dummies(sites).astype("float32")
        self.site_names = list(self.cond_df.columns)
        self.site_probs = sites.value_counts(normalize=True).reindex(self.site_names).values

        self.columns = list(processed.columns)
        return processed
