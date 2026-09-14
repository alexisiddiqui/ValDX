# Suggested Open/Closed Kinetic Simulation

## Step 1: Build the equilibrium ensemble

- Use the existing TeaA Iso-Validation system.
- Use the ISO-Bimodal ensemble containing the open and closed conformations.
- Set the true populations to 40% open and 60% closed.
- Calculate the residue-level protection factors ($PF_i$) for this mixture using the standard hard-cutoff BV model.
- Do **not** use the Radou switch-function model for this benchmark. This ensures that the only deliberate model mismatch is kinetic.

**Save:**
- the true open and closed frame weights
- the true residue-level $PF_i$
- the standard BV parameters used to calculate them

## Step 2: Calculate intrinsic exchange rates

For every exchangeable residue $i$, calculate $k_{\mathrm{int},i}$ using the existing HDXer intrinsic-rate calculation with the same:

- sequence
- pH
- temperature
- neighbouring-residue corrections

Exclude prolines and other residues excluded by the current ValDX pipeline.

## Step 3: Convert protection factors into opening equilibria

For each residue, define its equilibrium probability of being open as:

$$p_{\mathrm{open},i} = \frac{1}{PF_i}$$

Then calculate the odds:

$$\frac{p_{\mathrm{open},i}}{1 - p_{\mathrm{open},i}}$$

which is equivalent to:

$$K_{\mathrm{open},i} = \frac{1}{PF_i - 1}$$

If $PF_i \leq 1$, set it to a small value above one (e.g. $1.000001$) before calculating $K_{\mathrm{open},i}$.

## Step 4: Define the kinetic regimes

Define:

$$\rho = \frac{k_{\mathrm{close},i}}{k_{\mathrm{int},i}}$$

Run the following values:

```python
rho_values = [1000, 100, 10, 1, 0.1, 0.01]
```

These represent:

| $\rho$ | Regime |
|---|---|
| 1000 or 100 | strong EX2 |
| 10 | weak EX2 |
| 1 | mixed EX1/EX2 |
| 0.1 | EX1-like |
| 0.01 | strong kinetic-misspecification stress test |

For every residue and every value of $\rho$, calculate:

$$k_{\mathrm{close},i} = \rho \, k_{\mathrm{int},i}$$

$$k_{\mathrm{open},i} = K_{\mathrm{open},i} \, k_{\mathrm{close},i}$$

This changes the opening and closing timescales while preserving the equilibrium open probability, and hence the equilibrium protection factor.

## Step 5: Generate exact kinetic uptake

For each residue, simulate the three-state kinetic scheme:

$$C_i \underset{k_{\mathrm{close},i}}{\overset{k_{\mathrm{open},i}}{\rightleftharpoons}} O_i \xrightarrow{k_{\mathrm{int},i}} D_i$$

where $C$ and $O$ are the unexchanged closed and open states, and $D$ is the exchanged state.

Use the transition-state matrix:

$$A_i = \begin{pmatrix} -k_{\mathrm{open},i} & k_{\mathrm{close},i} \\ k_{\mathrm{open},i} & -\left(k_{\mathrm{close},i} + k_{\mathrm{int},i}\right) \end{pmatrix}$$

Initialise each residue at conformational equilibrium:

$$\mathbf{p}_i(0) = \begin{pmatrix} 1 - p_{\mathrm{open},i} \\ p_{\mathrm{open},i} \end{pmatrix}$$

At each experimental time $t$, calculate:

$$\mathbf{p}_i(t) = \exp(A_i t) \, \mathbf{p}_i(0)$$

The fractional uptake is:

$$d_i(t) = 1 - p_{C,i}(t) - p_{O,i}(t)$$

**Implementation:**

```python
import numpy as np
from scipy.linalg import expm

def kinetic_uptake(PF, k_int, rho, time):
    PF = max(float(PF), 1.000001)

    p_open = 1.0 / PF
    K_open = p_open / (1.0 - p_open)

    k_close = rho * k_int
    k_open = K_open * k_close

    A = np.array([
        [-k_open,                  k_close],
        [ k_open, -(k_close + k_int)]
    ])

    p0 = np.array([
        1.0 - p_open,
        p_open
    ])

    p_t = expm(A * time) @ p0
    uptake = 1.0 - p_t.sum()

    return np.clip(uptake, 0.0, 1.0)
```

## Step 6: Validate the simulator

For the strong-EX2 condition, compare the exact kinetic uptake with the ordinary EX2 approximation:

$$1 - \exp\left(-\frac{k_{\mathrm{int},i}}{PF_i} t\right)$$

For $\rho = 1000$, these curves should nearly overlap.

Produce a debugging plot for approximately ten residues spanning low, medium, and high protection factors.

**Do not proceed until:**

1. the exact kinetic model agrees with the EX2 approximation at $\rho = 1000$;
2. all uptake values lie between zero and one;
3. uptake increases monotonically with time;
4. at $\rho = 1000$, fitting $\beta_C$ and $\beta_H$ against the noise-free data on the unreweighted prior ensemble recovers the true generating values (required for Step 9, Arm 2 to be interpretable).

## Step 7: Construct peptide-level uptake

For every peptide $p$ and time $t$, calculate:

$$D_p(t) = \sum_{i \in p} d_i(t)$$

Apply the same rules as the existing TeaA benchmark:

- exclude the peptide N-terminal amide
- exclude prolines
- apply the same D₂O fraction
- use the same peptide boundaries
- use the same experimental time points

Initially generate noise-free datasets.

If the $\rho = 0.01$ condition produces almost no uptake within the existing time window, retain it as a stress test but also generate a diagnostic log-spaced time course. Do not rescale each kinetic regime independently in the primary benchmark.

## Step 8: Add measurement noise

After confirming the noise-free implementation:

1. Estimate the peptide-level error SD from the existing TeaA or comparable experimental data.
2. Add independent Gaussian measurement noise.
3. Clip uptake to its physically allowed range.
4. Generate replicate datasets for every $\rho$.
5. Use matched random seeds across kinetic regimes so that differences are not driven by different noise realisations.

## Step 9: Fit each dataset with ValDX

Fit the kinetically generated datasets using the existing EX2/BV fitting code without telling the model which $\rho$ generated the data.

Fit at least:

- **ISO-Bimodal** — correct open and closed structural support
- **ISO-Trimodal** — open and closed structures plus incorrect intermediate structures

Run three fitting arms. Whichever stage runs first absorbs the systematic uptake distortion caused by kinetic misspecification, so the ordering of reweighting and BV-parameter optimisation is itself the experimental variable.

### Arm 1 — RW-only (primary)

Reweight with $\beta_C$ and $\beta_H$ fixed at their standard values.

Nothing can absorb the kinetic error except the frame weights, so this is a clean measurement of how much damage kinetic misspecification does to the recovered ensemble. This is the headline result.

**Report:** recovered open/closed populations vs $\rho$, against the true 40/60 split.

### Arm 2 — $\beta$-first, then RW (absorption test)

1. Optimise $\beta_C$ and $\beta_H$ against the data on the **unreweighted prior ensemble**.
2. Freeze those parameters.
3. Reweight using the frozen parameters.

Here the BV parameters get first opportunity to absorb the systematic offset. If they succeed, the fit residual looks healthy and the populations come out close to the truth — while $\beta_C$ and $\beta_H$ have silently drifted away from their true generating values. That drift is the concealment signal, and it is only visible in this ordering.

**Report:** $\beta_C$ and $\beta_H$ vs $\rho$, with the true generating values drawn as horizontal reference lines; recovered populations vs $\rho$ alongside them.

### Arm 3 — staged RW → $\beta$ (realistic pipeline)

Full staged ValDX fitting as normally run: reweighting first, then BV-parameter optimisation.

By the time the parameter stage runs, the weights are already committed and have absorbed the kinetic bias; the subsequent $\beta$ fit cannot undo it. This arm therefore answers "what does the production workflow actually return under EX1 contamination?" rather than testing absorption. Do not interpret it as the concealment test.

**Report:** recovered populations and $\beta$ values vs $\rho$; contrast the populations against Arm 2.

### Controls

- Use identical peptide splits and split seeds across kinetic regimes **and across arms**.
- Arm 2 is only interpretable if the $\beta$-first fit is well-posed: at $\rho = 1000$ it must recover the true generating $\beta_C$ and $\beta_H$ on the prior ensemble. Add this as a fourth gate to the Step 6 validation before running the full sweep.

## Step 10: Analysis

### Design and controls

The sweep is a factorial over 6 $\rho$ values × 3 fitting arms × 2 ensembles (ISO-Bimodal, ISO-Trimodal) × $N$ noise replicates, with matched seeds and matched peptide splits throughout.

$\rho = 1000$ is the **internal control**: it is the condition in which the EX2 assumption used by the fitting code is actually true. Every effect is therefore reported as a paired difference against $\rho = 1000$, within the same arm, ensemble and seed. This removes each ensemble's baseline recoverability — which is not the object of study — and isolates the kinetic effect.

Replicates are noise realisations only; the ensemble and the kinetics are deterministic. Confidence intervals therefore describe sensitivity to measurement noise, not general uncertainty, and must not be presented as the latter. If the stronger claim is wanted, resample the underlying frames as well.

### Endpoint 1 — Damage

**Arm 1, recovery score vs $\rho$.**

The primary result is a monotone decline in recovery as $\rho \to 0$. Report $\Delta$recovery relative to $\rho = 1000$, paired by seed. The recovery score already accounts for weight placed on the incorrect intermediate structures, so the ISO-Trimodal arm additionally answers whether kinetic misspecification actively recruits spurious structural support rather than merely blurring the correct populations. Report the bimodal and trimodal curves together; separation between them is the decoy-recruitment signal.

Extract the **breakdown threshold**: the largest $\rho$ at which recovery degrades beyond the noise-only spread at $\rho = 1000$. This is the practical deliverable — "EX2-based reweighting is safe down to $\rho \approx X$."

### Endpoint 2 — Concealment across the full metric panel

Recovery score alone cannot demonstrate concealment. Concealment is a **dissociation between the metrics a user would consult to judge a fit and the recovery they would actually obtain**. Evaluate the complete set of scoring metrics already constructed in the analysis scripts, at minimum:

- training-set error
- held-out / validation peptide error
- train–validation gap
- **work done** (the reweighting cost — how far the weights had to move from the prior)
- any remaining constructed scores

For each metric $m$ and each arm, compute the association between $m$ and the recovery score across the $\rho$ sweep (rank correlation, plus the metric-vs-recovery scatter coloured by $\rho$).

The central test: in Arm 1, error-based metrics should rise as kinetic error grows, correctly warning the user. The concealment claim is that in Arm 2 they flatten while recovery continues to degrade — i.e. the correlation between fit quality and recovery **collapses**. A metric whose association with recovery survives $\beta$ optimisation retains diagnostic value; one whose association collapses is actively misleading, because it reports a healthy fit on a corrupted ensemble.

**Work done deserves separate attention.** It is the only metric in the panel that does not depend on the forward-model residual — it measures displacement of the weights from the prior rather than agreement with the data. Under kinetic misspecification the weights must travel further to reproduce distorted uptake, so work is expected to rise as $\rho$ falls even where the residual stays flat. If that holds, work is a candidate EX1 detector that survives $\beta$ absorption, which converts a negative result into a usable diagnostic. Test it explicitly: work vs $\rho$, per arm, against the $\rho = 1000$ baseline.

**Report:** a metric × arm matrix of metric-recovery correlations; per-metric traces vs $\rho$ for all three arms on shared axes.

### Endpoint 3 — Absorption vs rescue

**Arm 2, $\beta_C$ and $\beta_H$ vs $\rho$,** with the true generating values as horizontal reference lines and $\rho = 1000$ confirming no drift.

Classify the outcome:

| $\beta$ drift | Recovery | Interpretation |
|---|---|---|
| Yes | Rescued | $\beta$ absorption is protective — parameter freedom helps |
| Yes | Unharmed, but error metrics improved | Pure concealment — the dangerous case |
| Yes | Still degraded | Absorption incomplete |
| No | Degraded | $\beta$ cannot absorb the kinetic mode at all |

Which cell the result occupies is the answer to the PI's question, and each implies a different conclusion.

### Mechanism and detectability

**Per-peptide distortion.** The true EX1 distortion of each peptide is known by construction (exact kinetic uptake minus its EX2 prediction). Regress per-peptide fit residual, and per-peptide contribution to recovery loss, on that known distortion. This establishes whether damage is diffuse or driven by a small number of fast-exchanging peptides.

**Time-resolved residual structure.** Under EX1 the residual should be sign-structured in time rather than randomly distributed. Test whether this structure flags contaminated datasets even where the scalar error metrics look healthy. Together with the work-done result, this determines whether contaminated fits are detectable in practice.
