"""Thermodynamic decomposition of a reweighted discrete ensemble."""

from __future__ import annotations
import numpy as np
import pandas as pd

R_KJ_MOL_K = 8.31446261815324e-3


def calc_pmf(weights, prior_weights=None, temperature=300.0, energies_kj=None):
    w = np.asarray(weights, float); w = w / w.sum()
    q = np.full_like(w, 1 / len(w)) if prior_weights is None else np.asarray(prior_weights, float)
    q = q / q.sum()
    mask = (w > 0) & (q > 0)
    kl_divergence = float(np.sum(w[mask] * np.log(w[mask] / q[mask])))
    work = R_KJ_MOL_K * temperature * kl_divergence
    delta_h = np.nan
    if energies_kj is not None:
        energy = np.asarray(energies_kj, float)
        delta_h = float(np.dot(w-q, energy))
    minus_t_delta_s = float(work - delta_h) if np.isfinite(delta_h) else np.nan
    return {"delta_G_opt_kj": float(work), "delta_H_opt_kj": delta_h,
            "-Tdelta_S_opt_kj": minus_t_delta_s,
            "delta_H_abs_kj": abs(delta_h) if np.isfinite(delta_h) else np.nan,
            "kl_divergence_nats": kl_divergence}


def calc_logpf_pmf(logpfs, temperature=300.0, gas_constant_j=8.31):
    """Figure-3 protection-factor PMF decomposition by split replicate.

    ``logpfs`` must contain ``Residues``, ``LogPf``, ``calc_name`` and
    ``dataset`` columns, with matching ``prior`` and ``test`` residue rows.
    The returned quantities reproduce the definitions used in the original
    Figure-3 notebook; they are PF-space diagnostics, not potential energies.
    """
    required = {"Residues", "LogPf", "calc_name", "dataset"}
    missing = required - set(logpfs.columns)
    if missing:
        raise ValueError(f"LogPf dataframe is missing columns: {sorted(missing)}")
    frame = logpfs[logpfs["dataset"].isin(["prior", "test"])].copy()

    def replicate(name):
        try:
            return int(str(name).rsplit("_", 1)[1]) - 1
        except (ValueError, IndexError):
            raise ValueError(f"cannot identify split replicate in {name!r}")

    frame["split_replicate"] = frame["calc_name"].map(replicate)
    results = []
    rt = gas_constant_j * temperature
    for rep, rep_frame in frame.groupby("split_replicate"):
        calculated = {}
        for dataset in ("prior", "test"):
            current = rep_frame[rep_frame["dataset"] == dataset].copy()
            if current.empty:
                raise ValueError(f"missing {dataset} LogPfs for split replicate {rep}")
            current = current.groupby("Residues", as_index=False)["LogPf"].mean().sort_values("Residues")
            average = current["LogPf"].mean()
            beta_star = current["LogPf"].std() / rt
            log_delta = np.abs(current["LogPf"].to_numpy() - average)
            h_opt = rt * np.abs(log_delta)
            q = np.exp(-np.abs(log_delta))
            z_opt = q.sum()
            pi_opt = z_opt * np.exp(-np.abs(h_opt) / rt)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_opt = -gas_constant_j * pi_opt * np.log(pi_opt)
            s_opt = np.nan_to_num(s_opt, nan=0.0, posinf=0.0, neginf=0.0)
            g_opt = h_opt - temperature * s_opt
            calculated[dataset] = pd.DataFrame({
                "Residues": current["Residues"].to_numpy(), "H_opt": h_opt,
                "S_opt": s_opt, "G_opt": g_opt}).set_index("Residues")
            calculated[dataset].attrs.update(avg_logpf=average, beta_star=beta_star, z_opt=z_opt)
        prior, test = calculated["prior"].align(calculated["test"], join="inner", axis=0)
        if prior.empty:
            raise ValueError(f"no common prior/test residues for split replicate {rep}")
        delta_s = (test["S_opt"] - prior["S_opt"]).abs().mean()
        results.append({
            "split_replicate": int(rep),
            "delta_H_opt_kj": float((test["H_opt"] - prior["H_opt"]).abs().mean() / 1000),
            "minus_T_delta_S_opt_kj": float(temperature * delta_s / 1000),
            "delta_G_opt_kj": float((test["G_opt"] - prior["G_opt"]).abs().mean() / 1000),
            "delta_H_abs_kj": float(rt * abs(test.attrs["avg_logpf"] - prior.attrs["avg_logpf"]) / 1000),
            "Z_opt": float(test.attrs["z_opt"]),
        })
    return pd.DataFrame(results).sort_values("split_replicate").reset_index(drop=True)
