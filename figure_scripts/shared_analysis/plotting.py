"""Small plotting/statistics helpers shared by benchmark figures."""

from __future__ import annotations
import numpy as np

FIGURE_WIDTH = 7.2
FIGURE_DPI = 300
STATE_COLOURS = {"Open": "#D55E00", "Closed": "#0072B2", "Intermediate": "#999999"}
ENSEMBLE_COLOURS = {"ISO-Bimodal": "#0072B2", "ISO-Trimodal": "#D55E00"}


def p_to_stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "ns"


def plot_mean_std_by_group(data, x, y, group, ax=None, colours=None, **kwargs):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca(); colours = colours or {}
    for name, frame in data.groupby(group, sort=False):
        summary = frame.groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.errorbar(summary[x], summary["mean"], yerr=summary["std"], label=name,
                    color=colours.get(name), marker="o", **kwargs)
    return ax


def paired_bootstrap_interval(values, baseline, n_boot=10000, seed=240513):
    delta = np.asarray(values, float) - np.asarray(baseline, float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(1)
    return tuple(np.quantile(draws, [0.025, 0.5, 0.975]))
