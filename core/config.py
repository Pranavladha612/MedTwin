"""The active dataset's settings, loaded from a YAML file in configs/.

Every part of the pipeline asks this module for column names, rules and
paths, so switching dataset means switching settings file, not editing code.

    from core import config
    cfg = config.active()        # heart_disease.yaml unless told otherwise
    config.use("configs/diabetes.yaml")

Scripts take `--config <file>`; the dashboard reads the MEDTWIN_CONFIG
environment variable.
"""

import os
from functools import cached_property

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT = os.path.join(ROOT, "configs", "heart_disease.yaml")


class DatasetConfig:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(self.path) as f:
            s = yaml.safe_load(f)

        self.name = s["name"]
        self.data_file = os.path.join(ROOT, s["data_file"])
        self.id_column = s.get("id_column")
        self.output_dir = os.path.join(ROOT, s["output_dir"])
        self.model_file = os.path.join(ROOT, s["model_file"])

        self.numeric = s["numeric"]
        self.categorical = s.get("categorical", [])
        self.condition_on = s.get("condition_on")
        self.rename = s.get("rename", {})

        self.zero_means_missing = s.get("zero_means_missing", [])
        self.exactly_zero_spike = s.get("exactly_zero_spike", [])
        self.plausible_ranges = s.get("plausible_ranges", {})
        self.discrete_numeric = s.get("discrete_numeric", {})
        self.decimals = s.get("decimals", {})

        self.target = s["predict"]["column"]
        self.positive_above = s["predict"]["positive_above"]
        self.fidelity_columns = s["fidelity_columns"]
        self.labels = s.get("labels", {})
        self.dashboard = s.get("dashboard", {})

    # ---- names -----------------------------------------------------------
    def report_name(self, col):
        """A source column's name in the output report."""
        return self.rename.get(col, col)

    def by_both_names(self, mapping):
        """A {source column: value} setting, keyed by source *and* report
        name, so it applies to data before and after renaming."""
        out = dict(mapping)
        out.update({self.report_name(c): v for c, v in mapping.items()})
        return out

    @cached_property
    def report_columns(self):
        """Output columns: every configured column in the data file's order,
        with the condition column last, under report names."""
        header = list(pd.read_csv(self.data_file, nrows=0).columns)
        wanted = set(self.numeric) | set(self.categorical)
        cols = [self.report_name(c) for c in header if c in wanted]
        if self.condition_on:
            cols.append(self.report_name(self.condition_on))
        return cols

    @property
    def report_categorical(self):
        cats = self.categorical + ([self.condition_on] if self.condition_on else [])
        return [self.report_name(c) for c in cats]

    @property
    def report_target(self):
        return self.report_name(self.target)

    @property
    def fidelity_report(self):
        return [self.report_name(c) for c in self.fidelity_columns]

    # ---- missing values ----------------------------------------------------
    @cached_property
    def flagged(self):
        """Numeric columns that are sometimes not measured (blank, or 0 where
        0 means missing). Each gets a "was it measured?" flag. Read from the
        data itself, so the settings file doesn't have to list them."""
        raw = pd.read_csv(self.data_file, usecols=self.numeric)
        for col in self.zero_means_missing:
            raw.loc[raw[col] == 0, col] = np.nan
        return [c for c in self.numeric if raw[c].isna().any()]

    @property
    def flagged_report(self):
        return [self.report_name(c) for c in self.flagged]


_active = None


def use(path):
    """Makes the settings file at `path` the active one."""
    global _active
    _active = DatasetConfig(path)
    return _active


def active():
    """The active settings: set by use(), else MEDTWIN_CONFIG, else heart disease."""
    if _active is None:
        use(os.environ.get("MEDTWIN_CONFIG", DEFAULT))
    return _active


def add_argument(parser):
    """Adds the standard --config option to a script's argument parser."""
    parser.add_argument(
        "--config",
        default=os.environ.get("MEDTWIN_CONFIG", DEFAULT),
        help="dataset settings file (default: configs/heart_disease.yaml)",
    )
