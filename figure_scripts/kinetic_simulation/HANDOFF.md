# Kinetic misspecification benchmark: reduced-scope handoff

## Local selection for the manuscript peptide calibration

The manuscript uses `calibration_peptide_gamma_0` (gamma 1--9). Its figures can
be rebuilt with the local normalised-sum selector; ValDX's production selector
is unchanged. The local score is min-max normalised training MSE plus min-max
normalised HDXer Apparent Work. Constant axes contribute zero, invalid candidates
are excluded, and ties select the lower gamma. No validation or population
recovery information enters selection.

From the repository root:

```bash
python figure_scripts/kinetic_simulation/TeaA_Kinetic_Recovery.py \
  --fits-root figure_scripts/kinetic_simulation/_fits_peptide_calibration_gamma_0 \
  --dataset-root figure_scripts/kinetic_simulation/_datasets_peptide \
  --out figure_scripts/kinetic_simulation/_analysis_output/calibration_peptide_gamma_0 \
  --selection normalised-sum
```

This reuses completed reweighting candidates, recomputes predictions and Work
metrics on the existing splits, and reruns the nine final BVafterRW fits using
the reselected consensus weights. Original fits remain intact; selected-stage
snapshots, BV refit outputs, and gamma provenance are in `selected_fits/` beneath
the analysis output. The preceding figures and tables are preserved in
`calibration_peptide_gamma_0_before_normalised_selection`. The default
`--selection saved` instead reproduces the original ValDX-selected analyses.

## Current decision

The original 36-invocation, two-ensemble sweep is no longer the default. Work proceeds in
phases, using `RW_exponent = [-1]` because the smoke test did not select solutions from the
other exponent ranges.

The three fitting arms remain unchanged:

- `RW-Only`: fixed true BV constants, then reweighting.
- `BV-RW`: fit BV constants against the prior, then reweight under the consensus BV fit.
- `RW-BV`: reweight first, then fit BV constants.

Every invocation has three split replicates. Staged-arm split replicates share the stage-1
consensus and are not independent experimental units.

## Phase 1: minimal calibration (default)

- Ensemble: ISO-Bimodal (`TeaA_filtered.xtc`) only.
- rho: `1000`, `1`, `0.01`.
- Noise: replicate `0` only.
- Split type: `R3` only.
- Split replicates: 3.
- RW exponents: `[-1]`.
- Invocations: 3.

Run:

```bash
python figure_scripts/kinetic_simulation/04_TeaA_kinetic_sweep.py
```

This is a calibration/smoke experiment, not a confirmatory uncertainty analysis. In
particular, staged arms have only one independent noise-level unit, and three rho values are
too few for a stable correlation or a precise breakdown threshold. Use it to verify direction,
fit behavior, runtime, and whether the strongest kinetic stress produces a measurable effect.

## Phase 1b: expanded Bimodal calibration

Run only if Phase 1 is informative and stable:

- Ensemble: ISO-Bimodal only.
- rho: `1000`, `100`, `10`, `1`, `0.01` (five values).
- Noise replicates: `0`, `1`, `2`.
- Split type: `R3` by default.
- Invocations: 15.

```bash
python figure_scripts/kinetic_simulation/04_TeaA_kinetic_sweep.py \
  --preset calibration-expanded
```

An optional second split type is a robustness analysis and is run as separate invocations:

```bash
python figure_scripts/kinetic_simulation/04_TeaA_kinetic_sweep.py \
  --preset calibration-expanded --split-types R3 Sp
```

This doubles Phase 1b to 30 invocations. Split types are intentionally isolated because
ValDX's between-stage consensus must not leak from one split type into another.

## Phase 2: optional Trimodal follow-on

This tests whether intermediate/decoy frames worsen concealment after an effect is established
in ISO-Bimodal.

- Ensemble: ISO-Trimodal (`TeaA_initial_sliced.xtc`).
- rho: `1000`, `0.01`.
- Noise replicates: `0`, `1`, `2`.
- Split type: `R3`.
- Invocations: 6.

```bash
python figure_scripts/kinetic_simulation/04_TeaA_kinetic_sweep.py \
  --preset trimodal-followup
```

Phase 2 is optional. With two rho values it estimates a paired endpoint contrast, not a kinetic
trend or breakdown regime.

## Custom reduced designs

All preset fields can be overridden. For example, a three-rho, one-noise Bimodal check is:

```bash
python figure_scripts/kinetic_simulation/04_TeaA_kinetic_sweep.py \
  --rho 1000 1 0.01 --noise-replicates 0
```

Outputs are keyed by ensemble, split type, rho, and noise replicate:

```text
_fits/{ensemble}/split-{split_type}/rho{rho}/n{noise_replicate}/
```

Timing summaries are written as `_analysis_output/timing_{preset}.json`. Existing smoke-test
outputs in the older path layout are not treated as part of the reduced-scope experiment.

## Analysis interpretation

- All effects remain paired to rho `1000` within a chain.
- Phase 1 reports descriptive changes and fit diagnostics only.
- Bootstrap intervals for staged arms require Phase 1b's three noise replicates; split
  replicates cannot be counted as independent for those arms.
- A breakdown regime requires the expanded rho grid. Do not infer one from Phase 1.
- The Trimodal follow-on reports the rho `0.01` minus rho `1000` contrast and its paired
  uncertainty; it does not estimate a threshold.
- Gaussian clipping is substantial in the generated data (roughly 24--31%) and must remain
  visible in interpretation and provenance.

## Completed gates

- Fresh calc_hdx arrays: 17,283 frames, closed then open.
- EX2 generator regression: exact to written file precision.
- Simulator gates: passed.
- Clean rho=1000 BV integration gate: passed (`beta_C=0.3658`, `beta_H=1.9462`).
