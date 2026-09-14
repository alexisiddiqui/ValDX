#!/usr/bin/env python3
"""Summarize HDXer convergence steps versus the tested gamma value."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


GAMMA_RE = re.compile(
    r"(?:reweighting_|mixed_)?gamma_([0-9]+)x10\^(-?[0-9]+)_?per_iteration_output\.dat$"
)
FINAL_RE = re.compile(r"^\s*(\d+)\s+.*# FINAL values")


def read_steps(path: Path) -> tuple[int | None, bool]:
    last = None
    completed = False
    # The iteration count is on the final data line; avoid scanning multi-GB
    # historical logs from the beginning.
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 16384))
        text = handle.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        first = line.split(maxsplit=1)
        if first and first[0].isdigit():
            last = int(first[0])
        match = FINAL_RE.match(line)
        if match:
            completed = True
    return last, completed


def collect(root: Path) -> pd.DataFrame:
    rows = []
    skipped = 0
    for path in root.rglob("*per_iteration_output.dat"):
        match = GAMMA_RE.search(path.name)
        if not match:
            continue
        coefficient, exponent = map(int, match.groups())
        n_steps, completed = read_steps(path)
        if n_steps is None:
            skipped += 1
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "experiment": relative.parts[0] if relative.parts else "root",
                "level": "peptide" if "_peptide" in path.parts else "residue",
                "gamma_coefficient": coefficient,
                "gamma_exponent": exponent,
                "gamma": coefficient * 10.0**exponent,
                "n_steps": n_steps,
                "completed": completed,
                "path": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No production per-iteration logs found")
    print(f"skipped_logs_without_iteration_data={skipped}")
    return frame.sort_values(["level", "gamma", "path"]).reset_index(drop=True)


def correlation(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "pearson_r": float(pearsonr(frame["gamma"], frame["n_steps"]).statistic),
        "pearson_p": float(pearsonr(frame["gamma"], frame["n_steps"]).pvalue),
        "spearman_rho": float(spearmanr(frame["gamma"], frame["n_steps"]).statistic),
        "spearman_p": float(spearmanr(frame["gamma"], frame["n_steps"]).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or Path(__file__).resolve().parent / "_analysis_output" / "n_steps_vs_gamma"
    output.mkdir(parents=True, exist_ok=True)
    frame = collect(args.root)
    frame.to_csv(output / "n_steps_vs_gamma.csv", index=False)

    stats = {
        "all": correlation(frame),
        "completed_only": correlation(frame[frame["completed"]]),
    }
    for level, group in frame.groupby("level"):
        stats[level] = correlation(group)
        stats[f"{level}_completed_only"] = correlation(group[group["completed"]])
    for experiment, group in frame.groupby("experiment"):
        if len(group) >= 2:
            stats[f"experiment:{experiment}"] = correlation(group)
    (output / "correlations.json").write_text(pd.Series(stats, dtype=object).to_json(indent=2))

    groups = [("all", frame), *list(frame.groupby("level"))]
    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, group) in zip(axes, groups):
        for gamma, subset in group.groupby("gamma"):
            ax.scatter(
                subset["gamma"], subset["n_steps"],
                s=18, alpha=0.45, label=f"gamma={gamma:g}",
            )
        stats_group = stats[label]
        completed_stats = stats["completed_only" if label == "all" else f"{label}_completed_only"]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("gamma = coefficient x 10^exponent")
        ax.set_title(
            f"{label.capitalize()}\n"
            f"Pearson r={stats_group['pearson_r']:.3f}; "
            f"Spearman rho={stats_group['spearman_rho']:.3f}\n"
            f"completed only: Pearson r={completed_stats['pearson_r']:.3f}; "
            f"Spearman rho={completed_stats['spearman_rho']:.3f}"
        )
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("HDXer n_steps to convergence")
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("HDXer convergence steps across kinetic benchmark runs")
    fig.tight_layout()
    fig.savefig(output / "n_steps_vs_gamma.png", dpi=180)
    plt.close(fig)

    print(f"runs={len(frame)}")
    print(frame.groupby("level")["n_steps"].agg(["count", "min", "median", "max"]).to_string())
    print("completion status:")
    print(frame.groupby(["level", "completed"]).size().to_string())
    for label, values in stats.items():
        print(f"{label}: {values}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
