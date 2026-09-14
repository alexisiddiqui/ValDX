#!/usr/bin/env python3
"""Build TeaA fractional-uptake data from Bradshaw detailed contact/H-bond files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .kinetics import ex2_uptake, kinetic_uptake
except ImportError:  # direct script execution
    from kinetics import ex2_uptake, kinetic_uptake

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "jaxent_autovalidation" / "_Bradshaw" / "Reproducibility_pack_v2"
DEFAULT_SEGS = PACK / "data/artificial_HDX_data/segs_teaa_noPro.dat"
TIMES = (0.167, 1.0, 10.0, 60.0, 120.0)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--folders", nargs="+", required=True, type=Path)
    p.add_argument("-n", "--numframes", nargs="+", required=True, type=int)
    p.add_argument("-w", "--weights", nargs="+", required=True, type=float)
    p.add_argument("-r", "--ratefile", required=True, type=Path)
    p.add_argument("-exp", "--expfile", default=DEFAULT_SEGS, type=Path)
    p.add_argument("-dt", "--times", nargs="+", default=TIMES, type=float)
    p.add_argument("--rho", default=1000.0, type=float)
    p.add_argument("--ex2", action="store_true")
    p.add_argument("--output-root", type=Path, default=HERE / "_datasets")
    p.add_argument("--scratch", action="store_true", help="write under _scratch without gate")
    return p.parse_args(argv)


def _resid(path):
    return int(path.stem.rsplit("_res_", 1)[1])


def _read_detailed(folders):
    blocks, residue_order = [], None
    for folder in folders:
        contacts = sorted(folder.glob("Contacts_chain_0_res_*.tmp"), key=_resid)
        hbonds = sorted(folder.glob("Hbonds_chain_0_res_*.tmp"), key=_resid)
        cres, hres = [_resid(x) for x in contacts], [_resid(x) for x in hbonds]
        if not contacts or cres != hres:
            raise ValueError(f"unmatched or absent detailed arrays in {folder}")
        common = cres if residue_order is None else sorted(set(residue_order) & set(cres))
        residue_order = common
        blocks.append((dict(zip(cres, contacts)), dict(zip(hres, hbonds))))
    contacts = [np.stack([np.loadtxt(c[r]) for r in residue_order]) for c, _ in blocks]
    hbonds = [np.stack([np.loadtxt(h[r]) for r in residue_order]) for _, h in blocks]
    return np.concatenate(contacts, axis=1), np.concatenate(hbonds, axis=1), np.array(residue_order)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args):
    if len(args.numframes) != len(args.weights):
        raise ValueError("numframes and weights must have equal lengths")
    weights = np.asarray(args.weights, float)
    weights /= weights.sum()
    contacts, hbonds, resids = _read_detailed(args.folders)
    if contacts.shape != hbonds.shape or contacts.shape[1] != sum(args.numframes):
        raise ValueError("detailed-array dimensions do not match --numframes")
    start = 0
    for count, weight in zip(args.numframes, weights):
        stop = start + count
        contacts[:, start:stop] *= weight / count
        hbonds[:, start:stop] *= weight / count
        start = stop
    ave_contacts, ave_hbonds = contacts.sum(1), hbonds.sum(1)
    log_pf = 0.35 * ave_contacts + 2.0 * ave_hbonds
    pf = np.exp(log_pf)

    rate_data = np.loadtxt(args.ratefile, ndmin=2)
    rate_by_resid = dict(zip(rate_data[:, 0].astype(int), rate_data[:, 1]))
    kint = np.array([rate_by_resid[r] for r in resids])
    times = np.asarray(args.times)
    exact = ex2_uptake(pf, kint, times) if args.ex2 else kinetic_uptake(pf, kint, args.rho, times)
    ex2 = ex2_uptake(pf, kint, times)

    segments = np.loadtxt(args.expfile, usecols=(0, 1), dtype=int, ndmin=2)
    filters = [np.isin(resids, np.arange(a, b + 1)[1:]) for a, b in segments]
    counts = np.array([f.sum() for f in filters])
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise AssertionError(
            f"segments with no exchangeable residue: {segments[empty].tolist()}")
    fractions = np.stack([np.nanmean(exact[f], axis=0) for f in filters])
    ex2_segments = np.stack([np.nanmean(ex2[f], axis=0) for f in filters])

    if args.scratch:
        root = HERE / "_scratch"
    else:
        gate = HERE / "_analysis_output/simulator_gates.json"
        if not gate.exists() and not args.ex2:
            raise RuntimeError(f"simulator gate is absent: run 02_validate_simulator.py first ({gate})")
        root = args.output_root
    label = format(args.rho, "g")
    outdir = root / f"rho_{label}"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "ex2_regression" if args.ex2 else f"kinetic_rho{label}"
    frac_path = outdir / f"{stem}_deuterated_fracs.dat"
    data_path = outdir / f"{stem}_expt_resfracs.dat"
    np.savetxt(frac_path, fractions, fmt="%8.5f")
    with data_path.open("w") as handle:
        handle.write("# ResID Deuterated fraction, Times /\n")
        for segment, row in zip(segments, fractions):
            handle.write(f"{segment[0]} {segment[1]}\t" + " ".join(f"{x:8.5f}" for x in row) + "\n")

    p_open = 1.0 / np.maximum(pf, 1.000001)
    truth_path = outdir / f"truth_rho{label}.npz"
    np.savez_compressed(truth_path, PF=pf, k_int=kint, K_open=p_open/(1-p_open),
                        p_open=p_open, frame_weights=weights, beta_C=0.35, beta_H=2.0,
                        residue_ids=resids, segment_ids=segments,
                        ex1_distortion=fractions-ex2_segments, times=times, rho=args.rho)
    trajectories = [PACK / "data/trajectories/TeaA_closed_reimaged.xtc",
                    PACK / "data/trajectories/TeaA_open_reimaged.xtc"]
    topology = PACK / "data/trajectories/TeaA_ref_closed_state.pdb"
    provenance = {
        "truth_npz": str(truth_path.resolve()), "rho": args.rho, "times": times.tolist(),
        "beta_C": 0.35, "beta_H": 2.0, "frame_weights": weights.tolist(),
        "frame_counts": args.numframes, "concatenation_order": [str(x) for x in trajectories],
        "input_sha256": {str(x): _sha256(x) for x in trajectories + [topology, args.ratefile]},
        "tmp_source_directories": [str(x.resolve()) for x in args.folders],
        "calc_hdx_options": {"trajectories": [str(x) for x in trajectories],
                             "topology": str(topology), "method": "Radou",
                             "times": times.tolist(), "output_prefix": "TeaA_kin_",
                             "save_detailed": True, "contact_method": "cutoff"},
        "generator_options": dict(vars(args), weights=args.weights, numframes=args.numframes),
        "aggregation": "mean_fraction",
        "expfile": str(args.expfile),
        "residues_per_segment": {"min": int(counts.min()), "max": int(counts.max()),
                                 "mean": float(counts.mean())},
        "d2o_fraction_scaling": False, "noise_provenance": "per-noisy-dataset sidecar",
    }
    provenance["generator_options"] = {k: ([str(y) for y in v] if isinstance(v, list) and v and isinstance(v[0], Path) else str(v) if isinstance(v, Path) else v) for k, v in provenance["generator_options"].items()}
    (outdir / f"truth_rho{label}.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return data_path


if __name__ == "__main__":
    print(build(parse_args()))
