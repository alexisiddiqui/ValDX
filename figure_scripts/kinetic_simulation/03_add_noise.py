#!/usr/bin/env python3
"""Add matched Gaussian, clipped noise to generated kinetic datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .kinetics import RHO_VALUES
except ImportError:
    from kinetics import RHO_VALUES

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20250801)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--dataset-root", type=Path, default=HERE / "_datasets")
    args = p.parse_args()
    for rep in range(args.replicates):
        # Reinitialising from the same replicate seed for each rho gives matched pointwise noise.
        seed = args.seed + rep
        for rho in RHO_VALUES:
            label = format(rho, "g")
            folder = args.dataset_root / f"rho_{label}"
            source = folder / f"kinetic_rho{label}_expt_resfracs.dat"
            segments = np.loadtxt(source, usecols=(0, 1), dtype=int, ndmin=2)
            values = np.loadtxt(source, usecols=tuple(range(2, 7)), ndmin=2)
            raw = values + np.random.default_rng(seed).normal(0, args.sigma, values.shape)
            noisy = np.clip(raw, 0, 1)
            clipped = float(np.mean(raw != noisy))
            target = folder / f"kinetic_rho{label}_n{rep}_expt_resfracs.dat"
            with target.open("w") as handle:
                handle.write("# ResID Deuterated fraction, Times /\n")
                for segment, row in zip(segments, noisy):
                    handle.write(f"{segment[0]} {segment[1]}\t" + " ".join(f"{x:8.5f}" for x in row) + "\n")
            sidecar = {"rho": rho, "noise_replicate": rep, "noise_seed": seed,
                       "sigma": args.sigma, "noise_mode": "gaussian_clipped",
                       "clipped_point_fraction": clipped, "truth_file": str(folder/f"truth_rho{label}.npz")}
            (folder / f"noise_rho{label}_n{rep}.json").write_text(json.dumps(sidecar, indent=2)+"\n")
            print(f"rho={label} n={rep}: clipped={clipped:.3%}")

    # Diagnostic time course is truth-only and deliberately not in a ValDX input file.
    truth = args.dataset_root / "rho_0.01/truth_rho0.01.npz"
    if truth.exists():
        try:
            from .kinetics import kinetic_uptake
        except ImportError:
            from kinetics import kinetic_uptake
        with np.load(truth) as data:
            diagnostic_times = np.logspace(-3, 4, 100)
            diagnostic = kinetic_uptake(data["PF"], data["k_int"], .01, diagnostic_times)
        np.savez_compressed(args.dataset_root / "rho_0.01/diagnostic_timecourse_rho0.01.npz",
                            times=diagnostic_times, residue_uptake=diagnostic)


if __name__ == "__main__":
    main()
