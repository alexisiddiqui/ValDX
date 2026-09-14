"""Rebuild kinetic calibration from completed candidates and refit dependent BV stages."""

import contextlib
import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ValDX.reweighting import MaxEnt
from .gamma_selection import select_group

TIMES = [.167, 1., 10., 60., 120.]


def training_directory(analysis, name):
    paths = analysis.train_segs.loc[analysis.train_segs.calc_name == name, "path"].unique()
    if len(paths) != 1:
        raise ValueError(f"Ambiguous training directory for {name}")
    return Path(paths[0]).parent


def features(directory):
    files = sorted(directory.glob("Contacts_chain_0_res_*.tmp"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    residues = np.array([int(p.stem.rsplit("_", 1)[1]) for p in files])
    contacts = np.stack([np.loadtxt(p) for p in files])
    hbonds = np.stack([np.loadtxt(directory / f"Hbonds_chain_0_res_{r}.tmp") for r in residues])
    rates = np.loadtxt(next(directory.glob("*Intrinsic_rates.dat")))
    rate_map = dict(zip(rates[:, 0].astype(int), rates[:, 1]))
    return residues, contacts, hbonds, np.array([rate_map[r] for r in residues])


def prior_parameters(analysis):
    w = np.asarray(analysis.weights.loc[analysis.weights.dataset == "prior", "weights"].iloc[0])
    b = analysis.BV_constants.loc[analysis.BV_constants.dataset == "prior"].iloc[0]
    return w / w.sum(), (float(b.Bc), float(b.Bh))


def select_completed(analysis):
    fitted, work_records, selections = [], {}, []
    betas = analysis.BV_constants.set_index("calc_name")
    for rep, name in enumerate(analysis.train_rep_names):
        directory = training_directory(analysis, name)
        records = []
        for path in directory.glob("reweighting_gamma_*work.dat"):
            gamma, mse, _, work = np.loadtxt(path, ndmin=2)[-1, :4]
            records.append(dict(gamma=gamma, mse=mse, work=work, source=str(path)))
        selected = select_group(pd.DataFrame(records))
        work_path = Path(selected.source)
        weight_path = work_path.with_name(work_path.name.replace("work.dat", "final_weights.dat"))
        w = np.loadtxt(weight_path)
        fitted.append((w / w.sum(), (float(betas.loc[name, "Bc"]), float(betas.loc[name, "Bh"]))))
        work_records[rep] = dict(hdxer_work_kj=float(selected.work),
                                 hdxer_work_gamma=float(selected.gamma),
                                 hdxer_work_source=str(work_path.resolve()))
        selections.append(dict(calc_name=name, old_gamma=float(analysis.train_gammas[rep]),
                               selected_gamma=float(selected.gamma),
                               score=float(selected.selection_score), source=str(work_path.resolve())))
    return fitted, work_records, selections


def update_analysis(original, fitted, prior, work_records):
    """Recalculate predictions and PF diagnostics using the original peptide splits."""
    result = copy.deepcopy(original)
    residues, contacts, hbonds, kint = features(training_directory(original, original.train_rep_names[0]))

    def logpf(parameters):
        weights, (bc, bh) = parameters
        return bc * (contacts @ weights) + bh * (hbonds @ weights)

    # Confirm feature ordering against the original saved full-ensemble LogPfs.
    old_prior = logpf(prior_parameters(original))
    old_rows = original.LogPfs[original.LogPfs.dataset == "prior"]
    assert np.allclose(old_rows.LogPf, old_prior[np.searchsorted(residues, old_rows.Residues)])
    profiles = [logpf(p) for p in fitted]
    prior_profile = logpf(prior)
    segments = original.expt_segs.drop_duplicates("peptide").set_index("peptide")

    def predictions(profile):
        uptake = -np.expm1(-kint[:, None] * np.exp(-profile[:, None]) * np.array(TIMES))
        return {p: uptake[(residues > row.ResStr) & (residues <= row.ResEnd)].mean(axis=0)
                for p, row in segments.iterrows()}

    predicted = [predictions(p) for p in profiles]
    predicted_prior = predictions(prior_profile)
    for index, row in result.LogPfs.iterrows():
        rep = int(row.calc_name.rsplit("_", 1)[1]) - 1
        profile = prior_profile if row.dataset == "prior" else profiles[rep]
        result.LogPfs.at[index, "LogPf"] = profile[np.searchsorted(residues, row.Residues)]

    for index, row in result.weights.iterrows():
        if row.dataset == "prior":
            result.weights.at[index, "weights"] = prior[0].copy()
        elif isinstance(row.weights, (np.ndarray, list)):
            rep = int(row.calc_name.rsplit("_", 1)[1]) - 1
            result.weights.at[index, "weights"] = fitted[rep][0].copy()
        # This column is not used by calibration metrics; avoid retaining an old likelihood.
        result.weights.at[index, "likelihood"] = np.nan
    for index, row in result.BV_constants.iterrows():
        parameters = prior if row.dataset == "prior" else fitted[int(row.calc_name.rsplit("_", 1)[1]) - 1]
        result.BV_constants.loc[index, ["Bc", "Bh"]] = parameters[1]

    for rep in range(3):
        for kind in ("train", "val"):
            frame = getattr(result, f"{kind}_dfs")[rep]
            values = np.array([predicted[rep][p] for p in frame.peptide])
            # Production training predictions are loaded from five-decimal HDXer output.
            frame.loc[:, TIMES] = np.round(values, 5) if kind == "train" else values
    for field in ("merge_df", "HDX_data"):
        frame = getattr(result, field)
        for index, row in frame.iterrows():
            if row.calc_name == "Kinetic":
                continue
            rep = int(row.calc_name.rsplit("_", 1)[1]) - 1
            values = (predicted_prior if row.dataset == "prior" else predicted[rep])[row.peptide]
            frame.loc[index, TIMES] = np.round(values, 5) if row.dataset == "train" else values
    observed = original.merge_df[original.merge_df.calc_name == "Kinetic"].set_index("peptide")
    for index, row in result.analysis_df.iterrows():
        if row.Type not in ("Train", "Val"):
            continue
        rep = int(row.calc_name.rsplit("_", 1)[1]) - 1
        frame = (result.train_dfs if row.Type == "Train" else result.val_dfs)[rep]
        y = observed.loc[frame.peptide, row.time].to_numpy(dtype=float)
        prediction = frame[row.time].to_numpy(dtype=float)
        if pd.notna(row.mse):
            result.analysis_df.at[index, "mse"] = np.mean((prediction - y) ** 2)
        if pd.notna(row.R):
            result.analysis_df.at[index, "R"] = np.corrcoef(prediction, y)[0, 1]
    result.train_gammas = [work_records[r]["hdxer_work_gamma"] for r in range(3)]
    result.val_gammas = result.train_gammas.copy()
    result.hdxer_work_records = work_records
    return result


def refit_bv(original, prior, output):
    fitted, work_records = [], {}
    for rep, name in enumerate(original.train_rep_names):
        source = training_directory(original, name)
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        prefix = str(target / "reweighting_gamma_3x10^0")
        # Same BV-only settings and peptide data as the original run; only prior weights change.
        with (target / "refit.log").open("w") as log, contextlib.redirect_stdout(log):
            model = MaxEnt(do_reweight=False, do_params=True, stepfactor=1e-5,
                           random_initial=False, bv_bc=prior[1][0], bv_bh=prior[1][1])
            w, bc, bh = model.run(gamma=3, data_folders=[str(source)],
                                 kint_file=str(next(source.glob("*Intrinsic_rates.dat"))),
                                 exp_file=str(source / f"{name}_expt_dfracs.dat"), times=TIMES,
                                 iniweights=prior[0] * len(prior[0]), restart_interval=1000,
                                 out_prefix=prefix)
        if not np.isfinite([bc, bh]).all() or not np.allclose(w, prior[0]):
            raise ValueError(f"Invalid BV-only refit for {name}")
        fitted.append((w, (bc, bh)))
        work_path = Path(prefix + "work.dat")
        work_records[rep] = dict(hdxer_work_kj=float(np.loadtxt(work_path, ndmin=2)[-1, 3]),
                                 hdxer_work_gamma=3., hdxer_work_source=str(work_path.resolve()))
        print(f"Refitted {name}: beta_C={bc:.6g}, beta_H={bh:.6g}", flush=True)
    return fitted, work_records


def rebuild(analysis_module, output):
    """Return updated stage objects; preserve all original fitted candidates."""
    output.mkdir(parents=True, exist_ok=True)
    overrides, selections = {}, []
    for rho in analysis_module.RHOS:
        for arm, stage in (("RW-Only", ("RW_bench", 0)), ("BV-RW", ("RW_bench", 1)),
                           ("RW-BV", ("RW_bench", 0))):
            original = analysis_module._load(analysis_module._find_pickle(rho, arm, stage))
            fitted, work_records, selected = select_completed(original)
            rebuilt = update_analysis(original, fitted, prior_parameters(original), work_records)
            overrides[(rho, arm, stage)] = rebuilt
            selections.extend(dict(rho=rho, arm=arm, **row) for row in selected)
            if arm == "RW-BV":
                # Match run_sweep_methods: mean training weights, median BV constants.
                weights = np.mean([entry[0] for entry in fitted], axis=0)
                bv = tuple(rebuilt.BV_constants[["Bc", "Bh"]].median().to_numpy())
                prior = (weights / weights.sum(), bv)
                final_stage = ("BV_bench", 1)
                original_final = analysis_module._load(analysis_module._find_pickle(rho, arm, final_stage))
                final_fits, final_work = refit_bv(original_final, prior, output / f"rho{rho:g}")
                overrides[(rho, arm, final_stage)] = update_analysis(original_final, final_fits, prior, final_work)
    for (rho, arm, stage), result in overrides.items():
        with (output / f"rho{rho:g}_{arm}_{stage[0]}_{stage[1]}_analysis.pkl").open("wb") as handle:
            pickle.dump(result, handle)
    pd.DataFrame(selections).to_csv(output / "gamma_selection.csv", index=False)
    return overrides
