import numpy as np


def calculate_fidelity_score(real_df, synth_df, continuous_cols):
    """Calculates the Overlap (Intersection) of distributions."""
    # Fail loudly: silently skipping a missing column once made this score
    # cover only 'age' when the other columns had been renamed.
    missing = [c for c in continuous_cols if c not in real_df.columns or c not in synth_df.columns]
    if missing:
        raise KeyError(f"Fidelity columns missing from real or synthetic data: {missing}")

    scores = []
    for col in continuous_cols:
        # Create a histogram for both
        r_counts, bin_edges = np.histogram(real_df[col].dropna(), bins=20, density=True)
        s_counts, _ = np.histogram(synth_df[col].dropna(), bins=bin_edges, density=True)

        # Calculate the intersection (overlap)
        # This is a much better metric for GANs
        overlap = np.sum(np.minimum(r_counts, s_counts) * np.diff(bin_edges))
        scores.append(overlap)

    if not scores:
        return 0.0
    return round((sum(scores) / len(scores)) * 100, 2)
