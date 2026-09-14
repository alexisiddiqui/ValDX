#!/usr/bin/env python3
"""Numerical gates for the exact kinetic simulator; imports no ValDX code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .create_kinetic_target_data import HERE, PACK, _read_detailed
    from .kinetics import RHO_VALUES, ex2_uptake, kinetic_state_probabilities, kinetic_uptake
except ImportError:
    from create_kinetic_target_data import HERE, PACK, _read_detailed
    from kinetics import RHO_VALUES, ex2_uptake, kinetic_state_probabilities, kinetic_uptake


def _actual_arrays(folder, ratefile, frames=(16552, 731), weights=(0.6, 0.4)):
    contacts, hbonds, resids = _read_detailed([folder])
    if contacts.shape[1] != sum(frames):
        raise ValueError(f"expected {sum(frames)} frames, found {contacts.shape[1]}")
    start = 0
    for count, weight in zip(frames, weights):
        contacts[:, start:start+count] *= weight / count
        hbonds[:, start:start+count] *= weight / count
        start += count
    pf = np.exp((0.35 * contacts + 2.0 * hbonds).sum(1))
    rate = np.loadtxt(ratefile, ndmin=2)
    by_resid = dict(zip(rate[:, 0].astype(int), rate[:, 1]))
    return pf, np.array([by_resid[r] for r in resids]), resids


def validate(pf, kint, times):
    # Actual TeaA values are swept over both experimental and diagnostic times.
    # Pathological rate corners use rate-appropriate times; crossing k=1e6 with
    # t=1e8 only tests overflow/conditioning in SciPy's reference implementation.
    test_pf = pf
    test_k = kint
    test_times = np.unique(np.r_[0.0, times, np.logspace(-3, 4, 100)])
    max_backend = 0.0
    min_state, max_conservation = np.inf, 0.0
    for rho in RHO_VALUES:
        analytic = kinetic_uptake(test_pf, test_k, rho, test_times, clip=False)
        reference = kinetic_uptake(test_pf, test_k, rho, test_times, backend="expm", clip=False)
        max_backend = max(max_backend, float(np.max(np.abs(analytic-reference))))
        states = kinetic_state_probabilities(test_pf, test_k, rho, test_times)
        min_state = min(min_state, float(states.min()))
        total = states.sum(-1) + analytic
        max_conservation = max(max_conservation, float(np.max(np.abs(total-1))))
        assert np.all(np.diff(analytic, axis=1) >= -1e-9), f"non-monotone uptake at rho={rho}"
        assert np.all((analytic >= -1e-12) & (analytic <= 1+1e-12)), f"bounds failed at rho={rho}"
        assert np.all(analytic[:, test_times == 0] == 0.0), "t=0 is not exactly zero"
    corner_errors = {}
    corners = {
        "pf_near_one": (np.array([1.000001]), np.array([1.0]), np.array([0, .167, 120.])),
        "very_large_pf": (np.array([1e12]), np.array([1.0]), np.array([0, .167, 120.])),
        "extreme_fast_rate": (np.array([2.0]), np.array([1e6]), np.array([0, 1e-12, 1e-8, 1e-4])),
        "extreme_slow_rate": (np.array([2.0]), np.array([1e-12]), np.array([0, 1., 1e4, 1e8])),
    }
    for name, (corner_pf, corner_k, corner_t) in corners.items():
        error = 0.0
        for rho in RHO_VALUES:
            analytic = kinetic_uptake(corner_pf, corner_k, rho, corner_t, clip=False)
            reference = kinetic_uptake(corner_pf, corner_k, rho, corner_t, backend="expm", clip=False)
            error = max(error, float(np.max(np.abs(analytic-reference))))
            assert np.all(np.diff(analytic, axis=1) >= -1e-9)
            assert analytic[0, 0] == 0.0
        corner_errors[name] = error
    ex2_error = float(np.max(np.abs(kinetic_uptake(pf, kint, 1000, times) - ex2_uptake(pf, kint, times))))
    assert ex2_error < 1e-3, ex2_error
    # The analytic result is stable; scipy.linalg.expm loses several digits for
    # TeaA's largest k_int (~1.6e5 min^-1). Keep this explicit in the artifact.
    assert max_backend < 1e-6, max_backend
    assert max(corner_errors.values()) < 1e-8, corner_errors
    assert min_state >= -1e-10, min_state
    assert max_conservation < 1e-14, max_conservation
    return {"passed": True, "rho1000_max_abs_ex2_error": ex2_error,
            "analytic_expm_max_abs_error": max_backend,
            "minimum_unexchanged_state_probability": min_state,
            "probability_conservation_max_abs_error": max_conservation,
            "pathological_corner_errors": corner_errors,
            "backend_nominal_target": 1e-12, "teaA_scipy_reference_tolerance": 1e-6,
            "backend_tolerance_note": "SciPy expm loses precision for TeaA k_int up to ~1.6e5 min^-1",
            "n_actual_residues": len(pf), "n_crosscheck_timepoints": len(test_times)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=HERE / "_output/calchdx_latest")
    parser.add_argument("--ratefile", type=Path, default=PACK / "code/calc_hdx/data/TeaA_mixed_Intrinsic_rates.dat")
    args = parser.parse_args()
    folder = args.folder.resolve()
    times = np.array([0.167, 1.0, 10.0, 60.0, 120.0])
    pf, kint, resids = _actual_arrays(folder, args.ratefile)
    result = validate(pf, kint, times)
    result.update({"tmp_source_directory": str(folder), "times": times.tolist(), "rho_values": list(RHO_VALUES)})
    out = HERE / "_analysis_output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "simulator_gates.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    order = np.argsort(pf)
    indexes = order[np.linspace(0, len(order) - 1, 10).astype(int)]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharex=True, sharey=True)
    for ax, idx in zip(axes.flat, indexes):
        for rho in RHO_VALUES:
            ax.plot(times, kinetic_uptake(pf[idx:idx+1], kint[idx:idx+1], rho, times)[0], label=f"rho={rho:g}")
        ax.plot(times, ex2_uptake(pf[idx:idx+1], kint[idx:idx+1], times)[0], "k--", label="EX2")
        ax.set_xscale("log"); ax.set_title(f"res {resids[idx]}, PF={pf[idx]:.2g}")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "simulator_debug.png", dpi=180); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
