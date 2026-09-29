import numpy as np


def calculate_fidelity_score(real_df, synth_df, continuous_cols):
    """Calculates the Overlap (Intersection) of distributions."""
    scores = []
    for col in continuous_cols:
        if col in real_df.columns and col in synth_df.columns:
            # Create a histogram for both
            r_counts, bin_edges = np.histogram(
                real_df[col].dropna(), bins=20, density=True
            )
            s_counts, _ = np.histogram(
                synth_df[col].dropna(), bins=bin_edges, density=True
            )

            # Calculate the intersection (overlap)
            # This is a much better metric for GANs
            overlap = np.sum(np.minimum(r_counts, s_counts) * np.diff(bin_edges))
            scores.append(overlap)

    if not scores:
        return 0.0
    return round((sum(scores) / len(scores)) * 100, 2)
