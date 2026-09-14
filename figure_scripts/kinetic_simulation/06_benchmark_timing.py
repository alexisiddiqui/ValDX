#!/usr/bin/env python3
"""Record measured timing probes and extrapolate the benchmark wall time.

The unit probes are intentionally supplied as measurements: this avoids silently running
hundreds of MaxEnt jobs when a user only intended to inspect the cost model.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maxent-seconds", nargs=3, required=True, type=float,
                   metavar=("RUN1", "RUN2", "RUN3"))
    p.add_argument("--bv-seconds", required=True, type=float)
    p.add_argument("--rw-stage-seconds", required=True, type=float)
    p.add_argument("--cluster-bi-seconds", required=True, type=float)
    p.add_argument("--cluster-tri-seconds", default=0.0, type=float)
    p.add_argument("--rho001-rw-stage-seconds", required=True, type=float)
    args = p.parse_args()
    t_maxent = float(np.mean(args.maxent_seconds))
    # With RW_exponent=[-1], one n_reps=1 RW probe has nine MaxEnt runs and
    # fits into a single 12-worker scheduling wave. An invocation has one RW stage in each of
    # three arms, each crossed with three split replicates: nine probe units.
    # BV-RW and RW-BV each contribute one BV stage across three replicates.
    per_invocation = 9 * args.rw_stage_seconds + 6 * args.bv_seconds
    result = {
        "measurements": vars(args), "mean_single_maxent_seconds": t_maxent,
        "ideal_one_wave_seconds": t_maxent,
        "rw_pool_efficiency": t_maxent/args.rw_stage_seconds,
        "rho001_slowdown": args.rho001_rw_stage_seconds/args.rw_stage_seconds,
        "extrapolated_per_invocation_seconds": per_invocation,
        "projected_minimal_calibration_seconds": 3*per_invocation + args.cluster_bi_seconds,
        "projected_expanded_calibration_seconds": 15*per_invocation + args.cluster_bi_seconds,
        "projected_optional_trimodal_seconds": 6*per_invocation + args.cluster_tri_seconds,
        "validity_caveat": "rho changes convergence, not problem size; use rho=0.01 slowdown as the stress bound",
        "maxent_runs_per_invocation": 81,
        "RW_exponent": [-1],
    }
    out = HERE / "_analysis_output"; out.mkdir(exist_ok=True)
    (out / "timing_model.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"unit MaxEnt: {t_maxent:.1f}s")
    print(f"projected minimal calibration: {result['projected_minimal_calibration_seconds']/3600:.2f}h")
    print(f"projected expanded calibration: {result['projected_expanded_calibration_seconds']/3600:.2f}h")
    print(f"projected optional trimodal: {result['projected_optional_trimodal_seconds']/3600:.2f}h")


if __name__ == "__main__": main()
