import numpy as np
import pandas as pd
from figure_scripts.shared_analysis.recovery_metrics import compute_js_divergence, compute_kl_divergence_uniform, recovery_percent
from figure_scripts.shared_analysis.thermodynamic_metrics import calc_logpf_pmf, calc_pmf


def test_distribution_identities():
    assert compute_kl_divergence_uniform([1, 1, 1]) == 0
    assert compute_js_divergence([.4, .6], [.4, .6]) == 0
    assert recovery_percent([.4, .6], [.4, .6]) == 100


def test_pmf_prior_has_zero_work():
    assert abs(calc_pmf([.4, .6], [.4, .6])["delta_G_opt_kj"]) < 1e-14


def test_information_theoretic_decomposition_with_degenerate_energies():
    result = calc_pmf([.8, .2], energies_kj=[0, 0])
    assert result["delta_H_opt_kj"] == 0
    assert result["delta_G_opt_kj"] == result["-Tdelta_S_opt_kj"]
    assert result["delta_G_opt_kj"] > 0
    assert result["kl_divergence_nats"] > 0


def test_logpf_pmf_reports_both_enthalpy_components():
    data = pd.DataFrame({
        "Residues": [1, 2, 3, 1, 2, 3],
        "LogPf": [1., 2., 3., 2., 4., 7.],
        "calc_name": ["prior_fit_1"] * 3 + ["test_fit_1"] * 3,
        "dataset": ["prior"] * 3 + ["test"] * 3,
    })
    result = calc_logpf_pmf(data).iloc[0]
    assert result["delta_H_opt_kj"] > 0
    assert result["delta_H_abs_kj"] > 0
    assert result["delta_G_opt_kj"] > 0
    assert result["minus_T_delta_S_opt_kj"] > 0
