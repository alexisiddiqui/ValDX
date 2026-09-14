import numpy as np
from figure_scripts.kinetic_simulation.kinetics import kinetic_state_probabilities, kinetic_uptake


def test_analytic_matches_expm_and_conserves_probability():
    pf = np.array([1.000001, 2.0, 100.0, 1e8])
    rates = np.array([1e-6, .1, 10., 1e-4])
    times = np.array([0., .167, 1., 10., 120.])
    for rho in [1000., 1., .01]:
        exact = kinetic_uptake(pf, rates, rho, times, clip=False)
        reference = kinetic_uptake(pf, rates, rho, times, backend="expm", clip=False)
        np.testing.assert_allclose(exact, reference, atol=2e-10, rtol=2e-10)
        states = kinetic_state_probabilities(pf, rates, rho, times)
        np.testing.assert_allclose(states.sum(-1) + exact, 1., atol=1e-15)
        assert np.all(np.diff(exact, axis=1) >= -1e-9)
        assert np.all(exact[:, 0] == 0.)
