#!/usr/bin/env python3
"""Descriptive analysis for the reduced ISO-Bimodal kinetic calibration."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis import rms

from figure_scripts.shared_analysis.thermodynamic_metrics import calc_logpf_pmf, calc_pmf

sns.set_style("ticks")
sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    },
)

RHOS = (1000.0, 1.0, 0.01)
ARMS = ("RW-Only", "BV-RW", "RW-BV")
# Manuscript display names: the final fitting stage comes before "after".
# Keep the internal arm keys unchanged for compatibility with saved results.
ARM_LABELS = {"RW-Only": "RW-Only", "BV-RW": "RWafterBV", "RW-BV": "BVafterRW"}
FINAL_STAGE = {"RW-Only": ("RW_bench", 0), "BV-RW": ("RW_bench", 1),
               "RW-BV": ("BV_bench", 1)}
BV_STAGE = {"RW-Only": ("RW_bench", 0), "BV-RW": ("BV_bench", 0),
            "RW-BV": ("BV_bench", 1)}
TARGET = np.array([0.4, 0.6])  # Open, Closed

# Rebound by main() so the same analysis serves the residue-level and the
# synthetic-pepsin-peptide experiments.
FITS_ROOT = HERE / "_fits"
DATASET_ROOT = HERE / "_datasets"


def _rho_label(rho):
    return format(rho, "g")


def _find_pickle(rho, arm, stage):
    bench, index = stage
    root = FITS_ROOT / f"ISO-Bimodal/split-R3/rho{_rho_label(rho)}/n0/results"
    matches = list(root.glob(f"**/{bench}/**/*_{arm}{index}_fit_*_analysis.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {arm} stage {index} pickle at rho={rho}; got {matches}")
    return matches[0]


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _replicate(calc_name):
    match = re.search(r"_(\d+)$", str(calc_name))
    if not match:
        raise ValueError(f"cannot identify split replicate in {calc_name!r}")
    return int(match.group(1)) - 1


def classify_cached_frames():
    teaa = ROOT / "figure_scripts/jaxent_autovalidation/_TeaA/trajectories"
    bradshaw = ROOT / "figure_scripts/jaxent_autovalidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    cache = np.load(HERE / "_output/clusters/ISO-Bimodal_k500/cluster_data.npz")
    indices = cache["frame_indices"]
    universe = mda.Universe(str(teaa / "TeaA_ref_open_state.pdb"), str(teaa / "TeaA_filtered.xtc"))
    open_ref = mda.Universe(str(teaa / "TeaA_ref_open_state.pdb"))
    closed_ref = mda.Universe(str(bradshaw / "TeaA_ref_closed_state.pdb"))
    selection = "protein and name CA"
    mobile = universe.select_atoms(selection)
    refs = [open_ref.select_atoms(selection), closed_ref.select_atoms(selection)]
    distances = np.empty((len(indices), 2))
    for row, frame in enumerate(indices):
        universe.trajectory[int(frame)]
        for column, reference in enumerate(refs):
            distances[row, column] = rms.rmsd(
                mobile.positions, reference.positions, superposition=True)
    labels = distances.argmin(axis=1)  # 0 Open; 1 Closed
    return labels, distances, indices


def _train_rows(frame, value_column):
    rows = frame[frame["calc_name"].str.contains("train", na=False)].dropna(subset=[value_column]).copy()
    rows["split_replicate"] = rows["calc_name"].map(_replicate)
    return rows.sort_values("split_replicate")


def reported_hdxer_work(final, rho):
    """Read final-stage HDXer work at each saved selected training gamma."""
    if hasattr(final, "hdxer_work_records"):
        return final.hdxer_work_records
    root = FITS_ROOT / f"ISO-Bimodal/split-R3/rho{_rho_label(rho)}/n0/data"
    result = {}
    for name, gamma in zip(final.train_rep_names, final.train_gammas):
        directories = list(root.glob(f"**/{name}"))
        if len(directories) != 1:
            raise ValueError(f"Expected one training directory for {name}")
        matches = []
        for path in directories[0].glob("reweighting_gamma_*work.dat"):
            match = re.fullmatch(r"reweighting_gamma_([0-9]+)x10\^(-?[0-9]+)_?work.dat", path.name)
            if match and np.isclose(int(match[1]) * 10.0 ** int(match[2]), gamma):
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(f"Expected one work file for {name}, gamma={gamma}")
        values = np.loadtxt(matches[0], comments="#", ndmin=2)
        work = float(values[-1, 3])
        if not np.isfinite(work):
            raise ValueError(f"Non-finite HDXer work in {matches[0]}")
        result[_replicate(name)] = {
            "hdxer_work_kj": work, "hdxer_work_gamma": float(gamma),
            "hdxer_work_source": str(matches[0].resolve()),
        }
    return result


def stage_analysis(rho, arm, stage, overrides=None):
    key = (rho, arm, stage)
    if overrides is not None and key in overrides:
        return overrides[key]
    return _load(_find_pickle(rho, arm, stage))


def analyse(overrides=None):
    labels, distances, indices = classify_cached_frames()
    rows = []
    for rho in RHOS:
        for arm in ARMS:
            final = stage_analysis(rho, arm, FINAL_STAGE[arm], overrides)
            bv_data = stage_analysis(rho, arm, BV_STAGE[arm], overrides)
            hdxer_work = reported_hdxer_work(final, rho)
            weights = _train_rows(final.weights, "weights")
            betas = _train_rows(bv_data.BV_constants, "Bc")
            mse = final.analysis_df[final.analysis_df["Type"].isin(["Train", "Val"]) & final.analysis_df["mse"].notna()].copy()
            mse["split_replicate"] = mse["calc_name"].map(_replicate)
            mse_summary = mse.groupby(["split_replicate", "Type"])["mse"].mean().unstack()
            beta_lookup = betas.set_index("split_replicate")
            pmf_lookup = calc_logpf_pmf(final.LogPfs).set_index("split_replicate")
            for _, weight_row in weights.iterrows():
                rep = int(weight_row["split_replicate"])
                weight = np.asarray(weight_row["weights"], float)
                weight /= weight.sum()
                populations = np.array([weight[labels == state].sum() for state in (0, 1)])
                kl = calc_pmf(weight)["kl_divergence_nats"]
                train_mse = float(mse_summary.loc[rep, "Train"])
                val_mse = float(mse_summary.loc[rep, "Val"])
                rows.append({
                    "ensemble": "ISO-Bimodal", "arm": arm, "rho": rho,
                    "noise_replicate": 0, "split_replicate": rep, "split_type": "R3",
                    "open_population": populations[0], "closed_population": populations[1],
                    "recovery": 100.0 * populations[0] / TARGET[0],
                    "train_mse": train_mse, "val_mse": val_mse,
                    "train_val_gap": val_mse - train_mse,
                    "delta_G_opt_kj": float(pmf_lookup.loc[rep, "delta_G_opt_kj"]),
                    "delta_H_opt_kj": float(pmf_lookup.loc[rep, "delta_H_opt_kj"]),
                    "minus_T_delta_S_opt_kj": float(pmf_lookup.loc[rep, "minus_T_delta_S_opt_kj"]),
                    "delta_H_abs_kj": float(pmf_lookup.loc[rep, "delta_H_abs_kj"]),
                    "kl_divergence_nats": kl,
                    "beta_C": float(beta_lookup.loc[rep, "Bc"]),
                    "beta_H": float(beta_lookup.loc[rep, "Bh"]),
                    **hdxer_work[rep],
                })
    frame = pd.DataFrame(rows)
    baseline = frame[frame.rho == 1000].set_index(["arm", "split_replicate"])
    for metric in ["recovery", "train_mse", "val_mse", "train_val_gap", "delta_G_opt_kj",
                   "open_population", "beta_C", "beta_H"]:
        frame[f"delta_{metric}"] = [
            value - baseline.loc[(arm, rep), metric]
            for value, arm, rep in zip(frame[metric], frame.arm, frame.split_replicate)
        ]
    return frame, distances, indices


def plot(frame, out):
    rho_order = [1000.0, 1.0, .01]
    x = np.arange(len(rho_order))
    colours = {"RW-Only": "#0072B2", "BV-RW": "#D55E00", "RW-BV": "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    panels = [
        ("recovery", "Open state Recovery (%)"), ("open_population", "Open population"),
        ("hdxer_work_kj", r"Apparent $\mathrm{Work}_{\mathrm{HDXer}}$ [kJ/mol]"),
        ("train_mse", "Train MSE"), ("val_mse", "Validation MSE"),
        ("train_val_gap", "Validation − Train MSE"),
    ]
    for ax, (metric, ylabel) in zip(axes.flat, panels):
        for arm in ARMS:
            values = frame[frame.arm == arm].pivot(index="split_replicate", columns="rho", values=metric)
            mean = np.array([values[rho].mean() for rho in rho_order])
            std = np.array([values[rho].std(ddof=1) for rho in rho_order])
            ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=ARM_LABELS[arm], color=colours[arm])
        ax.set_xticks(x); ax.set_xticklabels(["1000", "1", "0.01"])
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=True, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out / "kinetic_calibration_overview.png", dpi=250); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for ax, metric, truth, ylabel in zip(axes, ["beta_C", "beta_H"], [.35, 2.0], [r"$\beta_C$", r"$\beta_H$"]):
        for arm in ["BV-RW", "RW-BV"]:
            values = frame[frame.arm == arm].pivot(index="split_replicate", columns="rho", values=metric)
            ax.errorbar(x, [values[r].mean() for r in rho_order],
                        yerr=[values[r].std(ddof=1) for r in rho_order], marker="o", capsize=3,
                        label=ARM_LABELS[arm], color=colours[arm])
        ax.axhline(truth, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(["1000", "1", "0.01"])
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(ylabel)
    axes[0].legend(frameon=True, framealpha=0.9); fig.tight_layout()
    fig.savefig(out / "kinetic_calibration_beta_drift.png", dpi=250); plt.close(fig)


def plot_work_components(frame, out):
    rho_order = [1000.0, 1.0, .01]
    rho_labels = ["1000", "1", "0.01"]
    positions = np.arange(len(rho_order))
    shape_scale_components = [
        ("delta_H_opt_kj", r"$\mathrm{Work}_{\mathrm{shape}}$", "#D55E00"),
        ("delta_H_abs_kj", r"$\mathrm{Work}_{\mathrm{scale}}$", "#E69F00"),
    ]
    opt_density_components = [
        ("delta_G_opt_kj", r"$\mathrm{Work}_{\mathrm{opt}}$", "#F0E442"),
        ("minus_T_delta_S_opt_kj", r"$\mathrm{Work}_{\mathrm{density}}$", "#0072B2"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex="col")
    width = .34
    for row, arm in enumerate(ARMS):
        arm_frame = frame[frame.arm == arm]
        for offset, (metric, label, colour) in zip([-width/2, width/2], shape_scale_components):
            values = arm_frame.pivot(index="split_replicate", columns="rho", values=metric)
            means = [values[rho].mean() for rho in rho_order]
            errors = [values[rho].std(ddof=1) for rho in rho_order]
            axes[row, 0].bar(positions + offset, means, width=width, yerr=errors,
                             capsize=3, label=label, color=colour, alpha=.9)
        for offset, (metric, label, colour) in zip([-width/2, width/2], opt_density_components):
            values = arm_frame.pivot(index="split_replicate", columns="rho", values=metric)
            means = [values[rho].mean() for rho in rho_order]
            errors = [values[rho].std(ddof=1) for rho in rho_order]
            axes[row, 1].bar(positions + offset, means, width=width, yerr=errors,
                             capsize=3, label=label, color=colour, alpha=.9)
        kl = arm_frame.pivot(index="split_replicate", columns="rho", values="kl_divergence_nats")
        axes[row, 2].errorbar(positions, [kl[r].mean() for r in rho_order],
                              yerr=[kl[r].std(ddof=1) for r in rho_order], marker="o",
                              capsize=3, color="#009E73")
        axes[row, 0].axhline(0, color="black", linewidth=.7)
        axes[row, 1].axhline(0, color="black", linewidth=.7)
        axes[row, 0].set_ylabel(f"{ARM_LABELS[arm]}\nWork [kJ/mol]")
        axes[row, 1].set_ylabel(f"{ARM_LABELS[arm]}\nWork [kJ/mol]")
        axes[row, 2].set_ylabel(f"{ARM_LABELS[arm]}\nKL divergence (nats)")
    axes[0, 0].legend(
        handles=[Patch(color=colour, label=label) for _, label, colour in shape_scale_components],
        frameon=True, framealpha=0.9, ncol=2, loc="upper left")
    axes[0, 1].legend(
        handles=[Patch(color=colour, label=label) for _, label, colour in opt_density_components],
        frameon=True, framealpha=0.9, ncol=2, loc="upper left")
    axes[0, 0].set_title("Shape and scale work")
    axes[0, 1].set_title("Total and density work")
    axes[0, 2].set_title("Frame-weight divergence")
    for ax in axes[-1]:
        ax.set_xticks(positions); ax.set_xticklabels(rho_labels); ax.set_xlabel(r"$\rho$")
    fig.suptitle("Work Done metrics and ensemble-weight KL divergence",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "kinetic_calibration_work_components.png", dpi=250)
    plt.close(fig)


def mechanism_analysis(overrides=None):
    records = []
    times = [.167, 1.0, 10.0, 60.0, 120.0]
    for rho in RHOS:
        truth_path = DATASET_ROOT / f"rho_{_rho_label(rho)}/truth_rho{_rho_label(rho)}.npz"
        with np.load(truth_path) as truth:
            distortion = truth["ex1_distortion"]
        for arm in ARMS:
            final = stage_analysis(rho, arm, FINAL_STAGE[arm], overrides)
            observed = final.merge_df[final.merge_df["calc_name"] == "Kinetic"].set_index("peptide")
            for rep in range(3):
                predicted = pd.concat([final.train_dfs[rep], final.val_dfs[rep]], ignore_index=True)
                predicted = predicted.set_index("peptide")
                residual_parts, distortion_parts = [], []
                mean_by_time, n_dropped = {}, 0
                for column, time in enumerate(times):
                    residual = predicted[time].astype(float) - observed[time].astype(float)
                    # predicted's train/val union can miss a few peptides that
                    # observed always has (splitter drops), which would
                    # otherwise poison the whole-array lstsq/corrcoef below.
                    residual = residual.sort_index().dropna()
                    n_dropped += len(observed) - len(residual)
                    current_distortion = distortion[residual.index.to_numpy(), column]
                    residual_parts.append(residual.to_numpy())
                    distortion_parts.append(current_distortion)
                    mean_by_time[f"residual_mean_t{time:g}"] = float(residual.mean())
                y = np.concatenate(residual_parts); x = np.concatenate(distortion_parts)
                design = np.column_stack([np.ones(len(x)), x])
                intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
                fitted = intercept + slope * x
                ss_res = np.sum((y - fitted) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
                records.append({"arm": arm, "rho": rho, "noise_replicate": 0, "n_dropped": n_dropped,
                                "split_replicate": rep, "ols_intercept": intercept,
                                "ols_distortion_slope": slope,
                                "ols_r_squared": 1 - ss_res / ss_tot if ss_tot else np.nan,
                                "residual_distortion_correlation": np.corrcoef(x, y)[0, 1],
                                **mean_by_time})
    return pd.DataFrame(records)


def plot_mechanism(frame, out):
    rho_order = [1000.0, 1.0, .01]; positions = np.arange(3)
    colours = {"RW-Only": "#0072B2", "BV-RW": "#D55E00", "RW-BV": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, metric, ylabel in zip(
            axes, ["ols_distortion_slope", "ols_r_squared", "residual_distortion_correlation"],
            ["Residual ~ distortion slope", r"OLS $R^2$", "Residual–distortion correlation"]):
        for arm in ARMS:
            values = frame[frame.arm == arm].pivot(index="split_replicate", columns="rho", values=metric)
            ax.errorbar(positions, [values[r].mean() for r in rho_order],
                        yerr=[values[r].std(ddof=1) for r in rho_order], marker="o", capsize=3,
                        label=ARM_LABELS[arm], color=colours[arm])
        ax.axhline(0, color="black", linewidth=.7); ax.set_xticks(positions)
        ax.set_xticklabels(["1000", "1", "0.01"]); ax.set_xlabel(r"$\rho$"); ax.set_ylabel(ylabel)
    axes[0].legend(frameon=True, framealpha=0.9); fig.tight_layout()
    fig.savefig(out / "kinetic_calibration_mechanism.png", dpi=250); plt.close(fig)


def main():
    global FITS_ROOT, DATASET_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--fits-root", type=Path, default=HERE / "_fits")
    p.add_argument("--dataset-root", type=Path, default=HERE / "_datasets")
    p.add_argument("--out", type=Path, default=HERE / "_analysis_output/calibration")
    p.add_argument("--selection", choices=["saved", "normalised-sum"], default="saved",
                   help="Locally reselect completed fits and refit dependent BV stages")
    args = p.parse_args()
    FITS_ROOT, DATASET_ROOT = args.fits_root, args.dataset_root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    overrides = None
    if args.selection == "normalised-sum":
        from figure_scripts.kinetic_simulation.reselect_calibration import rebuild
        backup = out.with_name(out.name + "_before_normalised_selection")
        if not backup.exists():
            shutil.copytree(out, backup)
        overrides = rebuild(sys.modules[__name__], out / "selected_fits")
    frame, distances, indices = analyse(overrides)
    frame.to_csv(out / "calibration_metrics_by_split.csv", index=False)
    summary = frame.groupby(["arm", "rho"]).agg(
        n=("recovery", "size"), recovery_mean=("recovery", "mean"), recovery_sd=("recovery", "std"),
        delta_recovery_mean=("delta_recovery", "mean"), open_population_mean=("open_population", "mean"),
        train_mse_mean=("train_mse", "mean"), val_mse_mean=("val_mse", "mean"),
        train_val_gap_mean=("train_val_gap", "mean"), work_mean=("delta_G_opt_kj", "mean"),
        beta_C_mean=("beta_C", "mean"), beta_H_mean=("beta_H", "mean"),
        hdxer_work_mean=("hdxer_work_kj", "mean")).reset_index()
    summary.to_csv(out / "calibration_summary.csv", index=False)
    mechanism = mechanism_analysis(overrides)
    mechanism.to_csv(out / "calibration_mechanism_by_split.csv", index=False)
    np.savez_compressed(out / "cached_frame_state_assignment.npz", frame_indices=indices,
                        rmsd_open=distances[:, 0], rmsd_closed=distances[:, 1],
                        state=np.where(distances.argmin(1) == 0, "Open", "Closed"))
    metadata = {"target": {"Open": .4, "Closed": .6}, "rho": list(RHOS),
                "recovery_definition": "Open state Recovery (%) = 100 * fitted open population / 0.4; not clipped",
                "selection": args.selection,
                "selection_definition": "min-max normalised training MSE + min-max normalised HDXer work; lower gamma breaks ties" if overrides is not None else "saved ValDX selection",
                "selected_fit_directory": str((out / "selected_fits").resolve()) if overrides is not None else None,
                "noise_replicates": [0], "split_replicates": [0, 1, 2],
                "interpretation": "descriptive calibration; split variation is not independent for staged arms",
                "fits_root": str(FITS_ROOT), "dataset_root": str(DATASET_ROOT),
                "hdxer_work": "Final-stage reported work.dat value at the saved selected training gamma; kJ/mol",
                "work_decomposition": {
                    "temperature_K": 300.0,
                    "kl_units": "nats",
                    "energy_units": "kJ/mol",
                    "definition": "Figure-3 LogPf PMF comparison of final-stage test versus prior",
                    "delta_H_opt": "mean absolute residue-wise change in H_opt",
                    "delta_H_abs": "RT times absolute change in ensemble-average LogPf",
                    "kl_definition": "KL divergence of fitted frame weights from uniform prior",
                    "physical_potential_energy_decomposition": False}}
    (out / "calibration_analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(frame, out)
    plot_work_components(frame, out)
    plot_mechanism(mechanism, out)
    print(summary.to_string(index=False))
    print("\nMechanism summary")
    print(mechanism.groupby(["arm", "rho"])[["ols_distortion_slope", "ols_r_squared",
          "residual_distortion_correlation"]].mean().to_string())


if __name__ == "__main__":
    main()
