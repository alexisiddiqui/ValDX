"""Distribution and structural recovery metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_rel


def _probability(x):
    x = np.asarray(x, dtype=float)
    if np.any(x < 0) or not np.isfinite(x).all() or x.sum() <= 0:
        raise ValueError("probabilities must be finite, non-negative, and nonzero")
    return x / x.sum()


def compute_kl_divergence_uniform(weights):
    p = _probability(weights)
    return float(np.sum(np.where(p > 0, p * np.log(p * len(p)), 0.0)))


def compute_js_divergence(observed, target):
    return float(jensenshannon(_probability(observed), _probability(target), base=2.0) ** 2)


def calculate_marginal_likelihood(log_likelihood, weights=None):
    values = np.asarray(log_likelihood, dtype=float)
    p = np.full(len(values), 1 / len(values)) if weights is None else _probability(weights)
    peak = values.max()
    return float(peak + np.log(np.sum(p * np.exp(values - peak))))


def recovery_percent(observed, target):
    return float(np.clip(1.0 - np.sqrt(compute_js_divergence(observed, target)), 0, 1) * 100)


def compute_rmsd_to_references(universe, references, selection="protein and name CA"):
    """Return frame-by-reference aligned RMSDs using MDAnalysis."""
    from MDAnalysis.analysis import rms
    mobile = universe.select_atoms(selection)
    result = np.empty((len(universe.trajectory), len(references)))
    for frame, _ in enumerate(universe.trajectory):
        for ref_index, reference in enumerate(references):
            target = reference.select_atoms(selection)
            result[frame, ref_index] = rms.rmsd(mobile.positions, target.positions, superposition=True)
    return result


def cluster_frames_by_rmsd(rmsd_values, state_names=None):
    values = np.asarray(rmsd_values)
    labels = np.argmin(values, axis=1)
    if state_names is None:
        return labels
    names = np.asarray(state_names)
    return names[labels]


def perform_ttest(before, after):
    return ttest_rel(np.asarray(before), np.asarray(after), nan_policy="omit")
