#!/usr/bin/env python3
"""Build a synthetic pepsin peptide map for the TeaA kinetic benchmark.

Pepsin has no strict recognition motif, so cleavage is modelled as a per-bond
propensity dominated by P1/P1' hydrophobicity, with the usual proline block.
Peptides are drawn as independent stochastic digests and pooled until the
requested per-residue coverage redundancy is reached. The map depends only on
sequence, never on the kinetics, so peptide selection cannot correlate with rho.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEAA = ROOT / "figure_scripts/jaxent_autovalidation/_TeaA/trajectories"
DEFAULT_TOP = TEAA / "TeaA_ref_open_state.pdb"

# CHARMM/protonation-state variants map onto their parent residue.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H",
    "HIE": "H", "HIP": "H", "GLH": "E", "ASH": "D", "LYN": "K", "CYX": "C",
    "CYM": "C", "MSE": "M",
}

# Relative pepsin preference at the two positions flanking the scissile bond.
# Bulky hydrophobics dominate; charged and small polar residues are disfavoured.
W_P1 = {"F": 1.00, "L": 1.00, "W": 0.90, "Y": 0.80, "M": 0.60, "A": 0.40,
        "V": 0.35, "I": 0.35, "E": 0.30, "Q": 0.25, "C": 0.20, "T": 0.15,
        "S": 0.12, "N": 0.12, "G": 0.10, "H": 0.12, "D": 0.08, "K": 0.06,
        "R": 0.05, "P": 0.00}
W_P1P = {"F": 0.90, "L": 0.90, "W": 0.80, "Y": 0.70, "M": 0.50, "A": 0.40,
         "V": 0.35, "I": 0.35, "E": 0.30, "Q": 0.25, "C": 0.20, "T": 0.18,
         "S": 0.15, "N": 0.15, "G": 0.15, "H": 0.15, "D": 0.10, "K": 0.08,
         "R": 0.08, "P": 0.00}
PRO_NEIGHBOUR_FACTOR = 0.15  # proline at P2 or P2' suppresses but does not forbid


def sequence(topology):
    universe = mda.Universe(str(topology))
    alpha = universe.select_atoms("protein and name CA")
    letters = [THREE_TO_ONE.get(name, "X") for name in alpha.resnames]
    return "".join(letters), alpha.resids.astype(int)


def bond_propensity(seq):
    """Relative cleavage propensity of each bond i|i+1, indexed by i (0-based)."""
    weights = np.zeros(len(seq) - 1)
    for i in range(len(seq) - 1):
        p1, p1p = seq[i], seq[i + 1]
        value = W_P1.get(p1, 0.10) * W_P1P.get(p1p, 0.15)
        if p1 == "P" or p1p == "P":
            value = 0.0
        else:
            if i - 1 >= 0 and seq[i - 1] == "P":
                value *= PRO_NEIGHBOUR_FACTOR
            if i + 2 < len(seq) and seq[i + 2] == "P":
                value *= PRO_NEIGHBOUR_FACTOR
        weights[i] = value
    return weights


def calibrate(weights, target_length, tol=1e-4, iterations=200):
    """Scale propensities so the expected inter-cut spacing is target_length."""
    lo, hi = 1e-6, 1e6
    for _ in range(iterations):
        mid = np.sqrt(lo * hi)
        expected_cuts = np.clip(weights * mid, 0.0, 1.0).sum()
        spacing = len(weights) / max(expected_cuts, 1e-9)
        if abs(spacing - target_length) < tol:
            break
        if spacing > target_length:
            lo = mid
        else:
            hi = mid
    return mid


def digest(probabilities, rng, min_length, max_length):
    """One stochastic digest; returns surviving (start_index, stop_index) pairs."""
    cuts = np.flatnonzero(rng.random(len(probabilities)) < probabilities)
    boundaries = np.concatenate(([-1], cuts, [len(probabilities)]))
    fragments = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        length = stop - start
        if min_length <= length <= max_length:
            fragments.append((start + 1, stop))  # inclusive 0-based residue span
    return fragments


def build_map(seq, resids, args):
    weights = bond_propensity(seq)
    scale = calibrate(weights, args.mean_length)
    probabilities = np.clip(weights * scale, 0.0, 1.0)
    rng = np.random.default_rng(args.seed)
    coverage = np.zeros(len(seq))
    peptides, digests = [], 0
    seen = set()
    while coverage.mean() < args.coverage and digests < args.max_digests:
        digests += 1
        for start, stop in digest(probabilities, rng, args.min_length, args.max_length):
            if (start, stop) in seen:
                continue
            seen.add((start, stop))
            peptides.append((start, stop))
            coverage[start:stop + 1] += 1
    peptides.sort()
    return peptides, coverage, probabilities, scale, digests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", type=Path, default=DEFAULT_TOP)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--min-length", type=int, default=5)
    p.add_argument("--max-length", type=int, default=15)
    p.add_argument("--mean-length", type=float, default=10.0)
    p.add_argument("--coverage", type=float, default=2.5, help="target mean per-residue redundancy")
    p.add_argument("--max-digests", type=int, default=500)
    p.add_argument("--out", type=Path, default=HERE / "_peptides")
    args = p.parse_args()

    seq, resids = sequence(args.topology)
    peptides, coverage, probabilities, scale, digests = build_map(seq, resids, args)

    # ValDX segment convention: "start stop" resids, N-terminal amide excluded,
    # so the observable residues of a peptide are start+1 .. stop.
    rows = [(int(resids[a]), int(resids[b])) for a, b in peptides]
    observed = [sum(1 for r in range(a + 1, b + 1)
                    if seq[list(resids).index(r)] != "P") for a, b in rows]
    keep = [row for row, count in zip(rows, observed) if count > 0]

    args.out.mkdir(parents=True, exist_ok=True)
    segs = args.out / "segs_teaa_pepsin.dat"
    with segs.open("w") as handle:
        for a, b in keep:
            handle.write(f"{a} {b}\n")

    lengths = np.array([b - a for a, b in keep])
    provenance = {
        "topology": str(args.topology), "sequence_length": len(seq),
        "seed": args.seed, "digests_drawn": digests, "propensity_scale": scale,
        "length_bounds": [args.min_length, args.max_length],
        "target_mean_length": args.mean_length, "target_coverage": args.coverage,
        "n_peptides": len(keep),
        "peptide_length_mean": float(lengths.mean()),
        "peptide_length_min": int(lengths.min()), "peptide_length_max": int(lengths.max()),
        "coverage_mean": float(coverage.mean()),
        "residues_uncovered": int((coverage == 0).sum()),
        "model": {
            "form": "p_cut(i) = clip(scale * W_P1[P1] * W_P1'[P1'] * proline_mask, 0, 1)",
            "W_P1": W_P1, "W_P1_prime": W_P1P,
            "proline_rule": "zero if P1 or P1' is Pro; x0.15 per Pro at P2 or P2'",
            "pooling": "independent Bernoulli digests, deduplicated, pooled to target coverage",
            "caveats": ["propensity caricature, not a fitted cleavage predictor",
                        "no structure, accessibility, charge-state or MS-detectability model",
                        "sequence-only: independent of the kinetic regime"]},
    }
    (args.out / "pepsin_map_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    np.savez_compressed(args.out / "pepsin_map.npz", segments=np.array(keep),
                        coverage=coverage, bond_probability=probabilities,
                        resids=resids)
    print(json.dumps({k: v for k, v in provenance.items() if k != "model"}, indent=2))
    print(segs)


if __name__ == "__main__":
    main()
