#!/usr/bin/env python3
"""Combine peptide gamma sweeps and select models across both exponents."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .gamma_selection import select_group
except ImportError:
    from gamma_selection import select_group


WORK_RE = re.compile(r"reweighting_gamma_([0-9]+)x10\^(-?[0-9]+)_?work\.dat$")
ARM_RE = re.compile(r"_(RW-Only|BV-RW|RW-BV)[01]_fit_")


def read_work(path: Path):
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            values = list(map(float, line.split()[:4]))
            return values[0], values[1], values[2], values[3]
    return None


def collect(root: Path, label: str):
    rows = []
    for path in root.glob("**/train_*/reweighting_gamma_*work.dat"):
        match = WORK_RE.match(path.name)
        row = read_work(path)
        if not match or row is None:
            continue
        coefficient, exponent = map(int, match.groups())
        rho = next((part[3:] for part in path.parts if part.startswith("rho")), None)
        arm_match = ARM_RE.search(str(path))
        rows.append({
            "fit_key": str(path.parent.relative_to(root)),
            "root": str(root),
            "root_label": label,
            "rho": rho,
            "arm": arm_match.group(1) if arm_match else "unknown",
            "gamma_coefficient": coefficient,
            "gamma_exponent": exponent,
            "gamma": coefficient * 10.0 ** exponent,
            "mse": row[1],
            "rmse": row[2],
            "work": row[3],
            "path": str(path.parent),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-m1", type=Path, required=True)
    parser.add_argument("--root-0", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    candidates = pd.concat([
        collect(args.root_m1, "gamma=-1 sweep"),
        collect(args.root_0, "gamma=0 sweep"),
    ], ignore_index=True)
    if candidates.empty:
        raise SystemExit("No gamma work files found")

    selected = pd.DataFrame([
        select_group(group)
        for _, group in candidates.groupby("fit_key", sort=False)
    ]).reset_index(drop=True)
    selected.to_csv(args.out / "peptide_selected_gamma_combined.csv", index=False)
    candidates.to_csv(args.out / "peptide_gamma_candidates.csv", index=False)

    counts = selected.groupby(["rho", "arm", "root_label", "gamma"]).size().reset_index(name="count")
    counts.to_csv(args.out / "peptide_selected_gamma_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, subset in selected.groupby("root_label"):
        axes[0].scatter(subset["gamma"], subset["rho"].astype(float), s=28, alpha=.7, label=label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Selected gamma")
    axes[0].set_ylabel("rho")
    axes[0].set_title("Peptide models selected across both exponents")
    axes[0].legend(frameon=False)

    plot = selected.copy()
    plot["label"] = plot["gamma"].map(lambda value: f"{value:g}")
    order = sorted(plot["gamma"].unique())
    table = plot.groupby(["root_label", "gamma"]).size().unstack(fill_value=0)
    table = table.reindex(columns=order, fill_value=0)
    table.T.plot(kind="bar", ax=axes[1], width=.8)
    axes[1].set_xlabel("Selected gamma")
    axes[1].set_ylabel("Number of fit replicates")
    axes[1].set_title("Selection counts")
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(args.out / "peptide_selected_gamma_combined.png", dpi=220)
    plt.close(fig)

    print(f"candidates={len(candidates)} selected={len(selected)} output={args.out}")
    print(selected.groupby(["rho", "arm", "root_label", "gamma"]).size().to_string())


if __name__ == "__main__":
    main()
