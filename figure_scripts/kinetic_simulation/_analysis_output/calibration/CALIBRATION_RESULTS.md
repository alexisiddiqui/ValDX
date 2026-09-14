# ISO-Bimodal kinetic calibration results

## Scope

This is the reduced descriptive calibration: rho = 1000, 1, and 0.01; one matched noise
realisation; three R3 split replicates; `RW_exponent = [-1]`; ISO-Bimodal only. Split variation
is descriptive, and the staged-arm split replicates are not independent because they share a
between-stage consensus.

All changes below are paired to rho = 1000 within arm and split replicate.

## Main result

At rho = 0.01, RW-Only and RW-BV produce essentially the same state-population damage:

| Arm | Recovery | Paired delta recovery | Train MSE | Validation MSE | Work (kJ/mol) |
|---|---:|---:|---:|---:|---:|
| RW-Only | 64.00 | -4.26 | 0.2721 | 0.2518 | 5.07 |
| RW-BV | 64.01 | -4.30 | 0.0166 | 0.0135 | 73.60 |
| BV-RW | 70.08 | +1.56 | 0.0154 | 0.0125 | 2.24 |

RW-BV therefore leaves the damaged ensemble essentially unchanged while reducing residual MSE
by more than an order of magnitude relative to RW-Only. In this calibration, fitting BV after
reweighting conceals kinetic misspecification from the standard residual metrics.

PF-space work remains strongly elevated for RW-BV (73.60 versus 9.53 kJ/mol at rho = 1000), so it is a more
promising warning diagnostic than MSE in this arm.

Here “work” is the Figure-3 protection-factor PMF metric calculated from final-stage `LogPfs`,
not `RT × KL` of the frame weights. Frame-weight KL is reported separately in nats.

## BV response

The fitted BV constants drift strongly as misspecification increases:

| Arm | rho | beta_C | beta_H |
|---|---:|---:|---:|
| BV-RW | 1000 | 0.367 | 1.980 |
| BV-RW | 1 | 0.399 | 2.038 |
| BV-RW | 0.01 | 0.677 | 3.103 |
| RW-BV | 1000 | 0.366 | 2.014 |
| RW-BV | 1 | 0.397 | 2.058 |
| RW-BV | 0.01 | 0.672 | 3.085 |

BV-RW is an absorption-with-rescue pattern in this limited Bimodal experiment: beta drift is
large, residuals remain small, and recovery does not decline. RW-BV is absorption-without-rescue:
beta drift makes the forward fit look good while the state-population damage remains.

## Mechanism

For RW-Only at rho = 0.01, fitted residuals correlate strongly with the known segment/time EX1
distortion (`r = -0.959`, mean across splits; OLS R-squared = 0.919). The correlation is nearly
removed after BV fitting:

| Arm | rho | Residual/distortion correlation | OLS R-squared |
|---|---:|---:|---:|
| RW-Only | 0.01 | -0.959 | 0.919 |
| RW-BV | 0.01 | 0.008 | 0.0003 |
| BV-RW | 0.01 | -0.018 | 0.0004 |

This supports the intended mechanism: BV freedom absorbs the structured kinetic residual. The
rho = 1000 regression slope is not interpretable because its kinetic-distortion denominator is
near zero; its correlation is retained only as a control diagnostic.

## Recovery calibration note

The cached 500-frame density sample is 5% Open by closest-reference C-alpha RMSD before
reweighting, whereas the target is 40% Open / 60% Closed. Absolute recovery is consequently
around 68% even at rho = 1000. The primary endpoint is therefore the paired change from the
rho = 1000 internal control, as specified in the design.

## Interpretation limit

These results establish a strong calibration signal but do not provide independent uncertainty
over noise realisations, a breakdown threshold, or a formal concealment confidence interval.
Those claims require the expanded Bimodal experiment with three noise replicates and a denser
rho grid.
