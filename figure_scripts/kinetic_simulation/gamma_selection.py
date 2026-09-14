"""Local kinetic-benchmark selection; does not change ValDX's selector."""

import numpy as np
import pandas as pd


def select_group(group):
    """Minimise independently min-max normalised training MSE + HDXer work.

    Constant objectives contribute zero; ties favour the lower gamma. Invalid
    candidates are excluded. The returned row retains its source information.
    """
    candidates = group.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["gamma", "mse", "work"])
    candidates = candidates[candidates.gamma > 0].sort_values(
        ["gamma", "mse", "work"]).drop_duplicates("gamma").reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No finite positive-gamma candidates")
    score = np.zeros(len(candidates))
    for metric in ("mse", "work"):
        values = candidates[metric].to_numpy()
        span = np.ptp(values)
        if span > 0:
            score += (values - values.min()) / span
    candidates["selection_score"] = score
    index = np.flatnonzero(np.isclose(score, score.min(), rtol=1e-12, atol=1e-12))[0]
    return candidates.iloc[index]
