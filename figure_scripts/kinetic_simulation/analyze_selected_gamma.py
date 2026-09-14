#!/usr/bin/env python3
"""Recover ValDX's selected L-curve gamma for kinetic target fits."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORK_RE = re.compile(r"reweighting_gamma_([0-9]+)x10\^(-?[0-9]+)_?work\.dat$")


def read_work(path: Path) -> tuple[float, float, float, float] | None:
    with path.open() as handle:
        for line in handle:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            gamma, mse, rmse, work = map(float, line.split()[:4])
            return gamma, mse, rmse, work
    return None


def select_gamma(directory: Path) -> tuple[float, int]:
    records = []
    for path in directory.glob("reweighting_gamma_*work.dat"):
        match = WORK_RE.match(path.name)
        if not match:
            continue
        coefficient, exponent = map(int, match.groups())
        row = read_work(path)
        if row is None:
            continue
        gamma, mse, rmse, work = row
        if np.isfinite(mse) and np.isfinite(work):
            records.append((coefficient * 10.0**exponent, mse, work))
    records.sort(key=lambda row: row[0])
    if not records:
        raise ValueError(f"No valid work rows in {directory}")
    if len(records) == 1:
        return records[0][0], len(records)
    angles = [
        math.atan2(records[i + 1][2] - records[i][2], records[i + 1][1] - records[i][1])
        for i in range(len(records) - 1)
    ]
    index = min(range(len(angles)), key=lambda i: abs(angles[i] - math.pi / 4))
    return records[index][0], len(records)


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for fit_root in (root / "_fits", root / "_fits_peptide"):
        level = "peptide" if fit_root.name.endswith("peptide") else "residue"
        for directory in fit_root.glob("**/Benchmark/RW_bench/**/train_*"):
            work_files = list(directory.glob("reweighting_gamma_*work.dat"))
            if not work_files:
                continue
            try:
                selected, n_candidates = select_gamma(directory)
            except ValueError:
                continue
            if n_candidates != 9:
                continue
            rows.append(
                {
                    "level": level,
                    "selected_gamma": selected,
                    "n_candidates": n_candidates,
                    "rho": next((p[3:] for p in directory.parts if p.startswith("rho")), None),
                    "arm": next((x for x in ("RW-Only", "BV-RW", "RW-BV") if x in directory.name), "unknown"),
                    "path": str(directory),
                }
            )
    return pd.DataFrame(rows).sort_values(["level", "path"]).reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "figure_scripts/kinetic_simulation"
    output = root / "_analysis_output/selected_gamma"
    output.mkdir(parents=True, exist_ok=True)
    frame = collect(root)
    frame.to_csv(output / "selected_gamma.csv", index=False)

    counts = frame.groupby(["level", "selected_gamma"]).size().unstack(fill_value=0)
    counts.to_csv(output / "selected_gamma_counts.csv")
    print(frame.groupby("level")["selected_gamma"].agg(["count", "median", "mean", "min", "max"]).to_string())
    print(counts.to_string())

    ax = counts.T.plot(kind="bar", figsize=(9, 4.8), width=0.8)
    ax.set_xlabel("Selected gamma")
    ax.set_ylabel("Number of fit replicates")
    ax.set_title("Selected L-curve gamma for kinetic targets")
    ax.legend(title="Target")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "selected_gamma_distribution.png", dpi=180)
    plt.close()
    print(f"output={output}")


if __name__ == "__main__":
    main()
