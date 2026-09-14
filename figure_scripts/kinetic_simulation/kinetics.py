"""Exact open/closed HDX kinetics with a SciPy-1.3-compatible backend."""

from __future__ import annotations

import numpy as np

RHO_VALUES = (1000.0, 100.0, 10.0, 1.0, 0.1, 0.01)


def ex2_uptake(PF, k_int, times):
    pf, rate, time = _inputs(PF, k_int, times)
    return -np.expm1(-rate[:, None] * time[None, :] / pf[:, None])


def _inputs(PF, k_int, times):
    pf = np.maximum(np.asarray(PF, dtype=float).reshape(-1), 1.000001)
    rate = np.asarray(k_int, dtype=float).reshape(-1)
    time = np.asarray(times, dtype=float).reshape(-1)
    if pf.shape != rate.shape:
        raise ValueError("PF and k_int must have the same shape")
    if np.any(~np.isfinite(pf)) or np.any(~np.isfinite(rate)) or np.any(~np.isfinite(time)):
        raise ValueError("inputs must be finite")
    if np.any(rate < 0) or np.any(time < 0):
        raise ValueError("rates and times must be non-negative")
    return pf, rate, time


def kinetic_state_probabilities(PF, k_int, rho, times, backend="analytic"):
    """Return unexchanged [closed, open] probabilities, shape (residue,time,2)."""
    pf, rate, time = _inputs(PF, k_int, times)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    p_open = 1.0 / pf
    k_close = float(rho) * rate
    k_open = p_open / (1.0 - p_open) * k_close
    p0 = np.column_stack((1.0 - p_open, p_open))

    if backend == "expm":
        from scipy.linalg import expm

        out = np.empty((len(pf), len(time), 2))
        for i, (ko, kc, ki) in enumerate(zip(k_open, k_close, rate)):
            A = np.array([[-ko, kc], [ko, -(kc + ki)]], dtype=float)
            for j, t in enumerate(time):
                out[i, j] = expm(A * t).dot(p0[i])
        return out
    if backend != "analytic":
        raise ValueError("backend must be 'analytic' or 'expm'")

    # exp(At)=c I+s(A-mu I), evaluated through the two non-positive
    # eigenvalues. This is the trace/determinant identity without the
    # overflow-prone exp(mu*t)*cosh(d*t) product.
    tau = -(k_open + k_close + rate)
    delta = k_open * rate
    mu = tau / 2.0
    d = np.sqrt(np.maximum(mu * mu - delta, 0.0))
    lam_plus, lam_minus = mu + d, mu - d
    ep = np.exp(lam_plus[:, None] * time[None, :])
    em = np.exp(lam_minus[:, None] * time[None, :])
    c = 0.5 * (ep + em)
    denom = 2.0 * d[:, None]
    s = np.divide(ep - em, denom, out=np.empty_like(ep), where=denom != 0)
    repeated = d == 0
    if np.any(repeated):
        s[repeated] = time[None, :] * np.exp(mu[repeated, None] * time[None, :])

    # (A-mu I)p0, vectorised over residues.
    bp0_c = (-k_open - mu) * p0[:, 0] + k_close * p0[:, 1]
    bp0_o = k_open * p0[:, 0] + (-(k_close + rate) - mu) * p0[:, 1]
    out = np.empty((len(pf), len(time), 2))
    out[:, :, 0] = c * p0[:, 0, None] + s * bp0_c[:, None]
    out[:, :, 1] = c * p0[:, 1, None] + s * bp0_o[:, None]
    return out


def kinetic_uptake(PF, k_int, rho, times, backend="analytic", clip=True):
    states = kinetic_state_probabilities(PF, k_int, rho, times, backend=backend)
    uptake = 1.0 - states.sum(axis=-1)
    uptake[:, np.asarray(times, dtype=float) == 0] = 0.0
    return np.clip(uptake, 0.0, 1.0) if clip else uptake
