import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# The source hospitals recorded "not measured" as 0 in these columns (e.g.
# Switzerland never measured cholesterol: all 123 of its patients have chol=0).
# A 0 here is never a real reading, so it is treated as missing.
ZERO_MEANS_MISSING = ["trestbps", "chol"]

# Columns missing often enough, and in a site-specific way, that the model
# should learn *whether* they were measured, not just their value. Each gets a
# 0/1 "<col>_missing" flag; generated records with the flag set get a blank
# value instead of an invented one.
FLAGGED_COLS = ["chol"]


def missing_flag(col):
    return f"{col}_missing"


def mark_missing(df):
    """Replaces 'not measured' zeros with NaN. Safe to call more than once."""
    df = df.copy()
    for col in ZERO_MEANS_MISSING:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
    return df


def add_missing_flags(df, fill_values):
    """Adds a 0/1 missing flag for each flagged column and fills its blanks
    with `fill_values[col]`, for models and metrics that can't take NaN.
    The flag keeps "not measured" distinguishable from the filled value."""
    df = df.copy()
    for col in FLAGGED_COLS:
        if col in df.columns:
            df[missing_flag(col)] = df[col].isna().astype(int)
            df[col] = df[col].astype(float).fillna(fill_values[col])
    return df


def blank_flagged(df):
    """Inverse of add_missing_flags: blanks values whose flag is set and
    drops the flag columns."""
    df = df.copy()
    for col in FLAGGED_COLS:
        flag = missing_flag(col)
        if flag in df.columns:
            df.loc[df[flag].round() == 1, col] = np.nan
            df = df.drop(columns=[flag])
    return df


class DataIngestion:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        # NOTE: every one of these gets MinMax-scaled to [0,1], which matters
        # because the Generator's output layer is a Sigmoid (see
        # core/generator.py) and can therefore only ever produce values in
        # [0,1]. Any numeric column *not* listed here will be fed to the GAN
        # in its raw real-world range but the model will still only be able
        # to output [0,1] for it -- silently truncating its real range and
        # making synthetic records trivially distinguishable from real ones.
        # ('num', disease severity 0-4, was previously missing from this list
        # for exactly that reason -- the generator could never output above 1.)
        self.num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "num"]
        self.cat_cols = [
            "sex",
            "dataset",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "thal",
        ]

    def de_identify(self, df):
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        return df

    def process_data(self, df):
        # Zeros that mean "not measured" become missing, and flagged columns
        # get a 0/1 flag so the GAN learns which patients lack them. The flag
        # is already in [0,1], so it needs no scaling for the Sigmoid output.
        df = mark_missing(df)
        for col in FLAGGED_COLS:
            df[missing_flag(col)] = df[col].isna().astype(float)

        # Handle Missing Values
        df[self.num_cols] = self.num_imputer.fit_transform(df[self.num_cols])
        df[self.cat_cols] = self.cat_imputer.fit_transform(df[self.cat_cols])

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=self.cat_cols)

        # Scaling
        df_encoded[self.num_cols] = self.scaler.fit_transform(df_encoded[self.num_cols])
        return df_encoded.astype("float32")
