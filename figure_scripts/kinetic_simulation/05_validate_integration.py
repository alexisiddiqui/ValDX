#!/usr/bin/env python3
"""Recover generating BV constants from the clean rho=1000 control."""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings

PACK = ROOT / "figure_scripts/jaxent_autovalidation/_Bradshaw/Reproducibility_pack_v2"
TEAA = ROOT / "figure_scripts/jaxent_autovalidation/_TeaA/trajectories"


DEFAULT_SEGS = PACK / "data/artificial_HDX_data/segs_teaa_noPro.dat"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=HERE / "_datasets")
    p.add_argument("--segs", type=Path, default=DEFAULT_SEGS)
    p.add_argument("--fits-root", type=Path, default=HERE / "_fits")
    p.add_argument("--gate-name", default="integration_gate.json")
    p.add_argument("--splits-dir", type=Path, default=HERE / "_splits")
    p.add_argument("--gate-mode", choices=("absolute", "baseline"), default="absolute",
                   help="absolute: require 5%% recovery of the generating betas. "
                        "baseline: record the rho=1000 betas as the paired internal "
                        "control without requiring absolute recovery.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    hdx = args.dataset_root / "rho_1000/kinetic_rho1000_expt_resfracs.dat"
    if not hdx.exists():
        raise FileNotFoundError(f"generate the clean rho=1000 dataset first: {hdx}")
    work = args.fits_root / "integration_gate"; work.mkdir(parents=True, exist_ok=True)
    splits_dir = args.splits_dir
    old = Path.cwd(); os.chdir(work)
    try:
        settings = Settings(name="TeaA_integration_gate")
        settings.train_frac = .5; settings.random_seed = 240513
        settings.split_mode = "R3"; settings.split_provenance_dir = str(splits_dir)
        outputs = ValDXer(settings).run_sweep_methods(
            system="TeaA_integration_gate", n_clusters=500,
            times=[.167, 1, 10, 60, 120], expt_name="Kinetic", method={"BV-Only": (True, False)},
            n_reps=3, split_modes=["R3"], hdx_path=str(hdx), skip_bench=True,
            segs_path=str(args.segs),
            traj_paths=[str(TEAA / "TeaA_filtered.xtc")], top_path=str(TEAA / "TeaA_ref_open_state.pdb"),
            cluster_cache_dir=str(HERE / "_output/clusters/ISO-Bimodal_k500"))
    finally:
        os.chdir(old)
    analysis = outputs[("BV-Only", 0)]["analysis"]
    bv = analysis["BV_constants"]
    bc, bh = np.asarray(bv["Bc"], float), np.asarray(bv["Bh"], float)
    result = {"beta_C": float(np.median(bc)), "beta_H": float(np.median(bh)),
              "beta_C_std": float(np.std(bc, ddof=1)), "beta_H_std": float(np.std(bh, ddof=1)),
              "beta_C_true": .35, "beta_H_true": 2.0,
              "hdx_path": str(hdx), "segs_path": str(args.segs),
              "within_five_percent": bool(abs(np.median(bc)/.35-1) <= .05 and abs(np.median(bh)/2-1) <= .05),
              "beta_C_relative_deviation": float(np.median(bc)/.35-1),
              "beta_H_relative_deviation": float(np.median(bh)/2.0-1),
              "gate_mode": args.gate_mode}
    mse = analysis.get("analysis_df")
    if mse is not None:
        result["forward_prediction_columns"] = [str(x) for x in mse.columns]
        if "mse" in mse.columns:
            result["forward_prediction_mse"] = {
                str(kind): {
                    "values": group["mse"].astype(float).tolist(),
                    "mean": float(group["mse"].astype(float).mean()),
                    "std": float(group["mse"].astype(float).std(ddof=1)),
                }
                for kind, group in mse.groupby("Type", dropna=False)
            }
    if args.gate_mode == "absolute":
        if not result["within_five_percent"]:
            raise AssertionError(json.dumps(result))
    else:
        # Peptide-resolution data does not pin the absolute betas on a prior
        # ensemble that differs from the generating mixture. The rho=1000 fit is
        # then the paired internal control, not an absolute recovery check, so
        # Arm 2 must be read as drift relative to this baseline.
        result["interpretation"] = (
            "rho=1000 betas recorded as the paired internal control; Arm 2 concealment "
            "must be read as drift relative to this baseline, not as absolute recovery")
    out = HERE / "_analysis_output"; out.mkdir(exist_ok=True)
    (out / args.gate_name).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
