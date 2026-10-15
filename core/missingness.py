"""How well synthetic data reproduces which values are missing, and whether
those missing-value patterns leak who was in the training set.

A record's *missingness pattern* is which of its values are blank ("not
measured"), together with its site when the data has one. Both checks work
on the report schema, where a blank cell means not measured.

Fidelity (are the gaps realistic?), each 0 = identical to the real data:
- miss_rate_gap: per column, |real - synthetic| share of blanks, averaged.
- site_miss_gap: the same per site and column, weighted by site size.
- pattern_tvd: total variation distance between the distributions of whole
  patterns, i.e. which values go missing *together* (0 = same, 1 = disjoint).
Scores depend on sample size, so synthetic data is scored on random
subsamples the size of the held-out set, and the held-out real patients are
scored the same way as a reference: the gap between two real samples.

Privacy (do the gaps give away training patients?). Members are training
patients, non-members held-out ones, in equal numbers; results are averaged
over N_DRAWS random member samples. AUC 0.5 = no leak.
- pattern_mia_auc: an attacker who sees only the synthetic data scores each
  real patient by how often their pattern occurs in it.
- rare_pattern_dist_mia_auc / common_pattern_dist_mia_auc: the
  distance-based membership attack of core/evaluation.py, on patients whose
  pattern is rare (held by at most 2% of real patients) versus the rest.
  A higher AUC for rare patterns means unusual gaps make patients easier to
  single out.
- unique_pattern_leak_pct: of patterns held by exactly one patient, the share
  reproduced in the synthetic data when that patient was in training, minus
  the share when they were held out (pooled over draws; 0 = no leak).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core import config

N_DRAWS = 20  # subsamples averaged for the size-matched fidelity scores
RARE_SHARE = 0.02  # a pattern held by at most 2% of real patients is rare
SEED = 42


def site_column(df):
    cfg = config.active()
    col = cfg.report_name(cfg.condition_on) if cfg.condition_on else None
    return col if col in df.columns else None


def missing_columns(real):
    """Columns that are blank for at least one real patient (site excluded)."""
    site = site_column(real)
    return [c for c in real.columns if c != site and real[c].isna().any()]


def patterns(df, cols):
    """Each record's missingness pattern as a string, e.g. 'Hungary|0010'."""
    bits = df[cols].isna().astype(int).astype(str).agg("".join, axis=1)
    site = site_column(df)
    return df[site].astype(str) + "|" + bits if site else bits


def _fidelity(reference, other, cols):
    ref_blank, oth_blank = reference[cols].isna(), other[cols].isna()
    scores = {"miss_rate_gap": float((ref_blank.mean() - oth_blank.mean()).abs().mean())}

    site = site_column(reference)
    if site:
        weights = reference[site].value_counts(normalize=True)
        ref_rates = ref_blank.groupby(reference[site]).mean()
        oth_rates = oth_blank.groupby(other[site]).mean().reindex(ref_rates.index)
        per_site = (ref_rates - oth_rates).abs().mean(axis=1)  # NaN if a site is absent
        scores["site_miss_gap"] = float((per_site * weights).sum() / weights[per_site.notna()].sum())

    p_ref = patterns(reference, cols).value_counts(normalize=True)
    p_oth = patterns(other, cols).value_counts(normalize=True)
    union = p_ref.index.union(p_oth.index)
    scores["pattern_tvd"] = float(
        0.5 * (p_ref.reindex(union, fill_value=0) - p_oth.reindex(union, fill_value=0)).abs().sum()
    )
    return scores


def evaluate_missingness(train_real, holdout_real, synth):
    """Missing-value fidelity of `synth` against the real training patients,
    with the held-out patients as a real-vs-real reference."""
    cols = missing_columns(pd.concat([train_real, holdout_real]))
    if not cols:
        return {}
    n = len(holdout_real)
    rng = np.random.RandomState(SEED)
    draws = [
        _fidelity(train_real, synth.iloc[rng.choice(len(synth), n, replace=False)], cols)
        for _ in range(N_DRAWS)
    ]
    result = {k: round(float(np.mean([d[k] for d in draws])), 4) for k in draws[0]}
    reference = _fidelity(train_real, holdout_real, cols)
    result.update({f"{k}_ref": round(v, 4) for k, v in reference.items()})
    return result


def _auc(labels, scores):
    if len(set(labels)) < 2:
        return float("nan")
    return round(float(roc_auc_score(labels, scores)), 3)


def evaluate_missingness_privacy(train_real, holdout_real, synth):
    """Membership attacks that use missing-value patterns (see module doc)."""
    # Imported here: core.evaluation imports this module.
    from core.evaluation import _encode, _nearest_distance

    cols = missing_columns(pd.concat([train_real, holdout_real]))
    if not cols:
        return {}
    p_train, p_hold = patterns(train_real, cols).values, patterns(holdout_real, cols).values
    synth_share = patterns(synth, cols).value_counts(normalize=True)
    in_synth = set(synth_share.index)
    real_share = pd.Series(np.r_[p_train, p_hold]).value_counts(normalize=True)

    # Distance from each real patient to its nearest synthetic record
    X_train, X_hold, X_synth = _encode([train_real, holdout_real, synth], train_real)
    d_train, d_hold = _nearest_distance(X_synth, X_train), _nearest_distance(X_synth, X_hold)

    n = min(len(train_real), len(holdout_real))
    hold_idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    aucs = {"pattern": [], "rare": [], "common": []}
    n_rare, unique = [], {"member": [0, 0], "nonmember": [0, 0]}  # [reproduced, total]
    for _ in range(N_DRAWS):
        mem_idx = rng.choice(len(train_real), n, replace=False)
        p_m, p_o = p_train[mem_idx], p_hold[hold_idx]
        labels = np.r_[np.ones(n), np.zeros(n)]
        freq = pd.Series(np.r_[p_m, p_o]).map(synth_share).fillna(0).values
        dist = -np.r_[d_train[mem_idx], d_hold[hold_idx]]
        rare = pd.Series(np.r_[p_m, p_o]).map(real_share).values <= RARE_SHARE
        aucs["pattern"].append(_auc(labels, freq))
        aucs["rare"].append(_auc(labels[rare], dist[rare]))
        aucs["common"].append(_auc(labels[~rare], dist[~rare]))
        n_rare.append(rare.sum())

        # Patterns held by exactly one patient in one group and nobody in the other
        c_m, c_o = pd.Series(p_m).value_counts(), pd.Series(p_o).value_counts()
        for group, mine, theirs in [("member", c_m, c_o), ("nonmember", c_o, c_m)]:
            only = [p for p, c in mine.items() if c == 1 and p not in theirs]
            unique[group][0] += sum(p in in_synth for p in only)
            unique[group][1] += len(only)

    def pct(group):
        hit, total = unique[group]
        return 100 * hit / total if total else float("nan")

    return {
        "pattern_mia_auc": round(float(np.nanmean(aucs["pattern"])), 3),
        "rare_pattern_dist_mia_auc": round(float(np.nanmean(aucs["rare"])), 3),
        "common_pattern_dist_mia_auc": round(float(np.nanmean(aucs["common"])), 3),
        "n_rare_patients": round(float(np.mean(n_rare)), 1),
        "unique_reproduced_member_pct": round(pct("member"), 2),
        "unique_reproduced_nonmember_pct": round(pct("nonmember"), 2),
        "unique_pattern_leak_pct": round(pct("member") - pct("nonmember"), 2),
        "n_unique_patterns": round((unique["member"][1] + unique["nonmember"][1]) / N_DRAWS, 1),
    }
