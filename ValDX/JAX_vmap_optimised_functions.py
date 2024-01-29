# functions to be optimised later - these should be mostly numpy operations
import os
# os.environ['JAX_PLATFORM_NAME']='cpu'

from jax import jit, vmap
import jax.numpy as jnp

# ave_lnpi is 3D array of shape [n_segments, n_residues, n_times]
# minuskt_filtered is 2D array of shape [n_segments, n_times]
# segfilters is 3D array of shape [n_segments, n_residues, n_times]
# curr_residue_dfracs is 3D array of shape [n_segments, n_residues, n_times]
# curr_segment_dfracs is 3D array of shape [n_segments, n_residues, n_times]
# exp_dfrac_filtered is 3D array of shape [n_segments, n_residues, n_times]
# n_datapoints is scalar
# curr_MSE is scalar

# @jit
# def calc_work_single_frame(init_lnpi_frame, lambdas, weight, kT):
#     """Calculate apparent work for a single frame."""
#     meanbias = -kT * jnp.sum(lambdas * init_lnpi_frame)
#     biaspot = -kT * jnp.sum(jnp.atleast_2d(lambdas).T * init_lnpi_frame, axis=0)
#     work = weight * jnp.exp((biaspot - meanbias) / kT)
#     return work

# # Now vectorize calc_work_single_frame over the frames axis
# calc_work_vectorized = vmap(calc_work_single_frame, in_axes=(1, None, 1, None))

# @jit
# def calc_work(init_lnpi, lambdas, weights, kT):
#     """Calculate apparent work, vectorized over frames."""
#     work_per_frame = calc_work_vectorized(init_lnpi, lambdas, weights, kT)
#     total_work = jnp.sum(work_per_frame)
#     work = kT * jnp.log(total_work)
#     return work


# @jit
# def calc_trial_dfracs_single_frame(ave_lnpi_frame, segment_filters_frame, filtered_minuskt_frame, filtered_exp_dfracs_frame):
#     """Calculate deuterated fractions and MSE for a single frame."""
#     denom = ave_lnpi_frame * segment_filters_frame
#     residue_dfracs = 1.0 - jnp.exp(jnp.divide(filtered_minuskt_frame, jnp.exp(denom), out=jnp.full(filtered_minuskt_frame.shape, jnp.nan), where=denom != 0))
    
#     segment_dfracs = jnp.nanmean(residue_dfracs, axis=1)
#     segment_dfracs = segment_dfracs[:, jnp.newaxis].repeat(segment_filters_frame.shape[1], axis=1)
#     MSE = jnp.sum((segment_dfracs * segment_filters_frame - filtered_exp_dfracs_frame) ** 2)
    
#     return residue_dfracs, segment_dfracs, MSE

# # Now vectorize calc_trial_dfracs_single_frame over the frames axis
# calc_trial_dfracs_vectorized = vmap(calc_trial_dfracs_single_frame, in_axes=(2, 2, 2, 2), out_axes=(2, 2, 0))

# @jit
# def calc_trial_dfracs(ave_lnpi, segment_filters, filtered_minuskt, filtered_exp_dfracs, n_datapoints):
#     """Calculate deuterated fractions and MSE, vectorized over frames."""
#     residue_dfracs, segment_dfracs, MSE_per_frame = calc_trial_dfracs_vectorized(ave_lnpi, segment_filters, filtered_minuskt, filtered_exp_dfracs)
    
#     # Sum the MSE over all frames and divide by the total number of data points
#     total_MSE = jnp.sum(MSE_per_frame) / n_datapoints
    
#     return residue_dfracs, segment_dfracs, total_MSE


@jit
def calc_bias_factor_mc(lambdas_c, contacts, lambdas_h, h_bonds):
    """
    Calculate bias factor for Monte Carlo Sampling provided values of:
    lambdas_c : jnp.array[n_residues] of lambda values for each residue for contacts
    contacts : jnp.array[n_residues, n_frames] of contacts, on a by-residue & by-frame basis
    lambdas_h : jnp.array[n_residues] of lambda values for each residue for h_bonds
    h_bonds : jnp.array[n_residues, n_frames] of h_bonds, on a by-residue & by-frame basis
    """

    biasfactor = jnp.sum(lambdas_c[:, jnp.newaxis] * contacts
                        + lambdas_h[:, jnp.newaxis] * h_bonds,
                        axis=0)
    return biasfactor

@jit
def calc_bias_factor_grad_single_frame(lambdas, lnpi_frame):
    """Calculate bias factor for a single frame."""
    biasfactor = jnp.sum(lambdas * lnpi_frame)
    return biasfactor

# Now vectorize calc_bias_factor_grad_single_frame over the frames axis
calc_bias_factor_grad_vectorized = vmap(calc_bias_factor_grad_single_frame, in_axes=(None, 1))

@jit
def calc_bias_factor_grad(lambdas, lnpi):
    """Calculate bias factor, vectorized over frames."""
    # Vectorized computation across the frames axis
    biasfactor_per_frame = calc_bias_factor_grad_vectorized(lambdas, lnpi)
    return biasfactor_per_frame

@jit
def calc_new_weight(iniweights, biasfactor):
    """Calculate new weights for provided values of:
    iniweights : jnp.array[n_frames] of current weights for each frame (should sum to 1)
    biasfactor : jnp.array[n_frames] of bias factor for each frame
    """
    newweights = iniweights * jnp.exp(-biasfactor)
    newweights = newweights / jnp.sum(newweights)
    return newweights



@jit
def calc_ave_lnpi(weights, lnpi):
    """Calculate average ln(protection_factor) for provided values of:
    weights : jnp.array[n_frames] of current weights for each frame (should sum to 1)
    lnpi : jnp.array[n_residues, n_frames] of ln(protection_factor), on a by-residue & by-frame basis"""
    ave_lnpi = jnp.sum(weights * lnpi, axis=1)
    return ave_lnpi

# @jit
# def calc_curr_residue_dfracs(ave_lnpi, minuskt, segment_filters):
#     """Calculate current residue deuterated fractions for provided values of:
#     ave_lnpi : jnp.array[n_segments, n_residues, n_times] of average ln(protection_factor) for each residue in each segment
#     minuskt : jnp.array[n_segments, n_times] of -kt rate constants for each segment
#     segment_filters : jnp.array[n_segments, n_residues, n_times] of Boolean filters for each segment
#     """

#     denom = ave_lnpi * segment_filters
#     residue_dfracs = 1.0 - \
#                      jnp.exp(jnp.divide(minuskt, jnp.exp(denom),
#                                       out=jnp.full(minuskt.shape, jnp.nan),
#                                       where=denom != 0))
#     return residue_dfracs

@jit
def calc_segment_and_MSE_from_residue_dfracs(residue_dfracs, segment_filters, filtered_exp_dfracs, n_datapoints):
    """Calculate segment deuterated fractions and MSE to target data for provided values of:
    residue_dfracs : jnp.array[n_segments, n_residues, n_times] of deuterated fractions for each residue in each segment
    segment_filters : jnp.array[n_segments, n_residues, n_times] of Boolean filters for each segment
    filtered_exp_dfracs : jnp.array[n_segments, n_residues, n_times] of experimental deuterated fractions for each residue in each segment
    n_datapoints : int of total number of datapoints
    """

    segment_dfracs = jnp.nanmean(residue_dfracs, axis=1)
    segment_dfracs = segment_dfracs[:, jnp.newaxis, :].repeat(segment_filters.shape[1], axis=1)
    MSE = jnp.sum((segment_dfracs * segment_filters
                  - filtered_exp_dfracs) ** 2) / n_datapoints
    return segment_dfracs, MSE

# @jit
# def calc_lambda_from_segment_dfracs(ave_lnpi, segment_filters, segment_dfracs, filtered_exp_dfracs, filtered_minuskt):
#     """Calculate lambda values for provided values of:
#     ave_lnpi : jnp.array[n_segments, n_residues, n_times] of average ln(protection_factor) for each residue in each segment
#     segment_filters : jnp.array[n_segments, n_residues, n_times] of Boolean filters for each segment
#     segment_dfracs : jnp.array[n_segments, n_residues, n_times] of deuterated fractions for each residue in each segment
#     filtered_exp_dfracs : jnp.array[n_segments, n_residues, n_times] of experimental deuterated fractions for each residue in each segment
#     filtered_minuskt : jnp.array[n_segments, n_times] of -kt rate constants for each segment
#     """
#     denom = ave_lnpi * segment_filters
#     curr_lambdas = jnp.nansum(
#         jnp.sum((segment_dfracs * segment_filters - filtered_exp_dfracs) * \
#                jnp.exp(jnp.divide(filtered_minuskt, jnp.exp(denom),
#                                 out=jnp.full(filtered_minuskt.shape, jnp.nan),
#                                 where=denom != 0)) * \
#                jnp.divide(-filtered_minuskt, jnp.exp(denom),
#                          out=jnp.full(filtered_minuskt.shape, jnp.nan),
#                          where=denom != 0), axis=2) / \
#         (jnp.sum(segment_filters, axis=1)[:, 0])[:, jnp.newaxis], axis=0)
#     return curr_lambdas

@jit
def calc_new_lambdas(curr_lambdas, step_size, gamma_target_lambdas):
    """Calculate new lambda values for provided values of:
    curr_lambdas : jnp.array[n_residues] of current lambda values for each residue
    step_size : float of step size for lambda update
    gamma_target_lambdas : jnp.array[n_residues] of target lambda values for each residue
    """
    

    lambdas = curr_lambdas * (1.0 - step_size) + (step_size * gamma_target_lambdas)
    return lambdas

