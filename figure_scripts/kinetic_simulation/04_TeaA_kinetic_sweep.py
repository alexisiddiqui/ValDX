#!/usr/bin/env python3
"""ValDX driver for the scoped TeaA kinetic benchmark.

The default is the minimal ISO-Bimodal calibration. Larger and optional designs
must be selected explicitly with ``--preset`` or individual CLI overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings

PACK = ROOT / "figure_scripts/jaxent_autovalidation/_Bradshaw/Reproducibility_pack_v2"
TEAA = ROOT / "figure_scripts/jaxent_autovalidation/_TeaA/trajectories"
RHO_VALUES = (1000.0, 100.0, 10.0, 1.0, 0.1, 0.01)
PRESETS = {
    # Cheapest interpretable rho trend: EX2 control, transition, stress condition.
    "calibration": {
        "rho": (1000.0, 1.0, 0.01),
        "ensembles": ("ISO-Bimodal",),
        "noise_replicates": (0,),
        "split_types": ("R3",),
    },
    # Add two intermediate kinetic conditions after the minimal calibration works.
    "calibration-expanded": {
        "rho": (1000.0, 100.0, 10.0, 1.0, 0.01),
        "ensembles": ("ISO-Bimodal",),
        "noise_replicates": (0, 1, 2),
        "split_types": ("R3",),
    },
    # Optional decoy-recruitment check, paired to control and strongest stress.
    "trimodal-followup": {
        "rho": (1000.0, 0.01),
        "ensembles": ("ISO-Trimodal",),
        "noise_replicates": (0, 1, 2),
        "split_types": ("R3",),
    },
}
METHODS = {"RW-Only": (False, True), "BV-RW": [(True, False), (False, True)],
           "RW-BV": [(False, True), (True, False)]}
ENSEMBLES = {"ISO-Bimodal": TEAA / "TeaA_filtered.xtc",
             "ISO-Trimodal": TEAA / "TeaA_initial_sliced.xtc"}


DEFAULT_SEGS = PACK / "data/artificial_HDX_data/segs_teaa_noPro.dat"


def run_invocation(rho, ensemble, noise_replicate, split_type="R3", n_reps=3,
                   rw_exponents=(-1,), dataset_root=None, segs=DEFAULT_SEGS,
                   fits_root=None, gate_name="integration_gate.json", splits_dir=None,
                   n_clusters=500):
    dataset_root = dataset_root or HERE / "_datasets"
    fits_root = fits_root or HERE / "_fits"
    splits_dir = splits_dir or HERE / "_splits"
    if not (HERE / "_analysis_output" / gate_name).exists():
        raise RuntimeError(f"integration gate absent ({gate_name}); run 05_validate_integration.py first")
    label = format(rho, "g")
    invocation = fits_root / f"{ensemble}/split-{split_type}/rho{label}/n{noise_replicate}"
    invocation.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(invocation)
    try:
        settings = Settings(name=f"TeaA_{ensemble}_rho{label}_n{noise_replicate}")
        settings.gamma_range = (1, 10); settings.train_frac = 0.5
        settings.RW_exponent = list(rw_exponents); settings.split_mode = split_type
        settings.random_seed = 240513
        settings.split_provenance_dir = str(splits_dir)
        vdx = ValDXer(settings)
        hdx = dataset_root / f"rho_{label}/kinetic_rho{label}_n{noise_replicate}_expt_resfracs.dat"
        started = time.perf_counter()
        vdx.run_sweep_methods(
            system=f"TeaA_{ensemble}_rho{label}_n{noise_replicate}",
            times=[0.167, 1, 10, 60, 120], expt_name="Kinetic", n_reps=n_reps,
            method=METHODS, split_modes=[split_type], n_clusters=n_clusters, skip_bench=True,
            cluster_cache_dir=str(HERE / f"_output/clusters/{ensemble}_k{n_clusters}"),
            hdx_path=str(hdx), segs_path=str(segs),
            traj_paths=[str(ENSEMBLES[ensemble])], top_path=str(TEAA / "TeaA_ref_open_state.pdb"))
        elapsed = time.perf_counter() - started
    finally:
        os.chdir(previous_cwd)
    (invocation / "invocation.json").write_text(json.dumps({
        "rho": rho, "ensemble": ensemble, "noise_replicate": noise_replicate,
        "split_type": split_type, "split_replicates": n_reps, "study_seed": 240513,
        "gamma_range": [1, 10], "RW_exponent": list(rw_exponents),
        "n_clusters": n_clusters,
        "methods": {k: str(v) for k, v in METHODS.items()}, "wall_seconds": elapsed,
        "hdx_path": str(hdx), "segs_path": str(segs)}, indent=2)+"\n")
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="calibration")
    p.add_argument("--rho", nargs="+", type=float, default=None)
    p.add_argument("--ensembles", nargs="+", choices=ENSEMBLES, default=None)
    p.add_argument("--noise-replicates", nargs="+", type=int, default=None)
    p.add_argument("--split-types", nargs="+", default=None)
    p.add_argument("--split-replicates", type=int, default=3)
    p.add_argument("--n-clusters", type=int, default=500)
    p.add_argument("--rw-exponents", nargs="+", type=int, default=[-1])
    p.add_argument("--dataset-root", type=Path, default=HERE / "_datasets")
    p.add_argument("--segs", type=Path, default=DEFAULT_SEGS)
    p.add_argument("--fits-root", type=Path, default=HERE / "_fits")
    p.add_argument("--gate-name", default="integration_gate.json")
    p.add_argument("--splits-dir", type=Path, default=HERE / "_splits")
    p.add_argument("--timing-tag", default=None, help="suffix for the timing summary file")
    p.add_argument("--dry-run", action="store_true", help="print resolved invocations and exit")
    args = p.parse_args()
    preset = PRESETS[args.preset]
    rhos = tuple(args.rho) if args.rho is not None else preset["rho"]
    ensembles = tuple(args.ensembles) if args.ensembles is not None else preset["ensembles"]
    noise = tuple(args.noise_replicates) if args.noise_replicates is not None else preset["noise_replicates"]
    split_types = tuple(args.split_types) if args.split_types is not None else preset["split_types"]
    unknown_splits = set(split_types) - {"r", "s", "R3", "Sp", "SR", "xR"}
    if unknown_splits:
        p.error(f"unsupported split types: {sorted(unknown_splits)}")
    planned = [
        {"rho": rho, "ensemble": ensemble, "noise_replicate": rep, "split_type": split_type}
        for rep in noise for ensemble in ensembles for rho in rhos for split_type in split_types
    ]
    print(json.dumps({"preset": args.preset, "RW_exponent": args.rw_exponents,
                      "split_replicates": args.split_replicates,
                      "n_clusters": args.n_clusters,
                      "invocation_count": len(planned), "invocations": planned}, indent=2))
    if args.dry_run:
        return
    timing = []
    for item in planned:
        record = dict(item)
        record.update({"preset": args.preset, "wall_seconds": run_invocation(
            item["rho"], item["ensemble"], item["noise_replicate"],
            split_type=item["split_type"], n_reps=args.split_replicates,
            rw_exponents=tuple(args.rw_exponents), dataset_root=args.dataset_root,
            segs=args.segs, fits_root=args.fits_root, gate_name=args.gate_name,
            splits_dir=args.splits_dir, n_clusters=args.n_clusters)})
        timing.append(record)
    out = HERE / "_analysis_output"; out.mkdir(exist_ok=True)
    tag = args.timing_tag or args.preset
    (out / f"timing_{tag}.json").write_text(json.dumps(timing, indent=2)+"\n")


if __name__ == "__main__":
    main()
