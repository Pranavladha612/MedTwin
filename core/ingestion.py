import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from core import config

# Label used for a categorical value that was never recorded. In the model it
# is an ordinary category; in the output report it becomes a blank cell.
NOT_MEASURED = "not measured"

# Which columns are numeric, categorical, sometimes unmeasured, zero-coded or
# zero-spiked all comes from the active settings file (core/config.py). Why
# each treatment exists:
#
# - "zero means missing": some hospitals wrote 0 for a test they didn't do
#   (e.g. every Swiss patient has chol = 0). Those zeros become blanks.
# - "not measured" flags: a numeric column that is sometimes missing gets a
#   0/1 flag so the model learns *whether* it was measured, and generated
#   records with the flag set get a blank instead of an invented number.
# - "exactly zero" flags: a smooth Sigmoid output can't hit one exact value,
#   so a column with a big spike at 0 (43% of oldpeak) gets a yes/no flag.
# - Every numeric column is MinMax-scaled to [0,1], because the Generator's
#   numeric outputs are Sigmoids and can only produce values in [0,1]. An
#   unscaled column would be silently capped at 1 ('num', severity 0-4, once
#   was -- the model could never output num >= 2).
# - The condition column (e.g. hospital) is not generated: the model is
#   conditioned on it, so each patient is generated for a given site and
#   site proportions match the real data exactly.


def missing_flag(col):
    return f"{col}_missing"


def zero_flag(col):
    return f"{col}_zero"


def mark_missing(df):
    """Replaces 'not measured' zeros with NaN. Safe to call more than once."""
    df = df.copy()
    for col in config.active().zero_means_missing:
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
    for col in config.active().flagged_report:
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
    for col in config.active().flagged_report:
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
        cfg = config.active()
        self.scaler = MinMaxScaler()
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_cols = cfg.numeric
        self.cat_cols = cfg.categorical

    def de_identify(self, df):
        id_column = config.active().id_column
        if id_column and id_column in df.columns:
            df = df.drop(columns=[id_column])
        return df

    def process_data(self, df):
        """Turns real records into the GAN's training matrix.

        Column order: scaled numerics, then missing and zero flags (together
        the Sigmoid outputs), then one one-hot block per categorical column
        (each a softmax output). Also records what the Generator needs to be
        built and sampled: `output_info`, `cat_groups`, the condition matrix
        `cond_df` and the training condition frequencies `site_probs`.
        """
        cfg = config.active()
        df = mark_missing(df)

        flags = pd.DataFrame(
            {missing_flag(c): df[c].isna().astype(float) for c in cfg.flagged},
            index=df.index,
        )
        for col in cfg.exactly_zero_spike:
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

        # Without a condition column every patient gets the same constant
        # condition, which makes the model an ordinary unconditional GAN.
        if cfg.condition_on:
            sites = df[cfg.condition_on].astype(str)
        else:
            sites = pd.Series("all", index=df.index)
        self.cond_df = pd.get_dummies(sites).astype("float32")
        self.site_names = list(self.cond_df.columns)
        self.site_probs = sites.value_counts(normalize=True).reindex(self.site_names).values

        self.columns = list(processed.columns)
        return processed
