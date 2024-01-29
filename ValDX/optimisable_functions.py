# functions to be optimised later - these should be mostly numpy operations

import numpy as np

def calc_work(init_lnpi, lambdas, weights, kT):
    """Calculate apparent work from the provided values of:
       init_lnpi : np.array[n_residues, n_frames] of ln(protection_factor), on a by-residue & by-frame basis
       lambdas : np.array[n_residues] of lambda values for each residue
       weights : np.array[n_frames] of current weights for each frame (should sum to 1)
       kT : value of kT for calculating work. Will determine units of the returned apparent work value.

       Usage:
       calc_work(init_lnpi, lambdas, weights, kT)

       Returns:
       work (float)"""

    # This is the same ave_lnpi calculated in the reweighting.py code but not broadcast to the full 3D array
    ave_lnpi = np.sum(weights * init_lnpi, axis=1)
    meanbias = -kT * np.sum(lambdas * ave_lnpi)
    biaspot = -kT * np.sum(np.atleast_2d(lambdas).T * init_lnpi, axis=0)
    work = np.sum(weights * np.exp((biaspot - meanbias) / kT))
    work = kT * np.log(work)
    return work


def calc_trial_dfracs(ave_lnpi, segment_filters, filtered_minuskt, filtered_exp_dfracs, n_datapoints):
    """For a trial parameter move, calculate deuterated fractions and mean square error to target data using
       the given average ln(protection factors). Average protection factors should be a 3D array of shape
       [n_segments, n_residues, n_times]. This is filtered to calculate the by-segment averages using the provided
       segment_filters Boolean array. The provided filtered_minuskt array (pre-filtered by the same segment_filters array)
       is then used to calculate the deuterated fractions. The provided filtered_exp_dfracs array and n_datapoints are
       then used to calculate the MSE to the target experimental data.

       Requires current average ln(protection factors), segment filters, pre-filtered -kt rate constants, pre-filtered
       target experimental deuterated fractions, and total number of datapoints

       Usage: calc_trial_dfracs(ave_lnpi, segment_filters, filtered_minuskt, filtered_exp_dfracs, n_datapoints)

       Returns: residue_dfracs, segment_dfracs, MSE_to_target"""
    # recalculate the deuterated fractions and MSE with the given ave_lnpi
    denom = ave_lnpi * segment_filters
    residue_dfracs = 1.0 - \
                     np.exp(np.divide(filtered_minuskt, np.exp(denom),
                                      out=np.full(filtered_minuskt.shape, np.nan),
                                      where=denom != 0))

    segment_dfracs = np.nanmean(residue_dfracs, axis=1)
    segment_dfracs = segment_dfracs[:, np.newaxis, :].repeat(segment_filters.shape[1], axis=1)
    MSE = np.sum((segment_dfracs * segment_filters
                  - filtered_exp_dfracs) ** 2) / n_datapoints
    return residue_dfracs, segment_dfracs, MSE

def calc_bias_factor_mc(lambdas_c, contacts, lambdas_h, h_bonds):


    biasfactor = np.sum(lambdas_c[:, np.newaxis] * contacts
                        + lambdas_h[:, np.newaxis] * h_bonds,
                        axis=0)
    return biasfactor

def calc_bias_factor_grad(lambdas, lnpi):
    biasfactor = np.sum(lambdas[:, np.newaxis] * lnpi, axis=0)
    return biasfactor

def calc_new_weight(iniweights, biasfactor):
    newweights = iniweights * np.exp(-biasfactor)
    newweights = newweights / np.sum(newweights)
    return newweights




def calc_ave_lnpi(weights, lnpi):
    ave_lnpi = np.sum(weights * lnpi, axis=1)
    return ave_lnpi


def calc_curr_residue_dfracs(ave_lnpi, minuskt, segment_filters):
    denom = ave_lnpi * segment_filters
    residue_dfracs = 1.0 - \
                     np.exp(np.divide(minuskt, np.exp(denom),
                                      out=np.full(minuskt.shape, np.nan),
                                      where=denom != 0))
    return residue_dfracs


def calc_segment_and_MSE_from_residue_dfracs(residue_dfracs, segment_filters, filtered_exp_dfracs, n_datapoints):
    segment_dfracs = np.nanmean(residue_dfracs, axis=1)
    segment_dfracs = segment_dfracs[:, np.newaxis, :].repeat(segment_filters.shape[1], axis=1)
    MSE = np.sum((segment_dfracs * segment_filters
                  - filtered_exp_dfracs) ** 2) / n_datapoints
    return segment_dfracs, MSE


def calc_lambda_from_segment_dfracs(ave_lnpi, segment_filters, segment_dfracs, filtered_exp_dfracs, filtered_minuskt):

    
    denom = ave_lnpi * segment_filters
    curr_lambdas = np.nansum(
        np.sum((segment_dfracs * segment_filters - filtered_exp_dfracs) * \
               np.exp(np.divide(filtered_minuskt, np.exp(denom),
                                out=np.full(filtered_minuskt.shape, np.nan),
                                where=denom != 0)) * \
               np.divide(-filtered_minuskt, np.exp(denom),
                         out=np.full(filtered_minuskt.shape, np.nan),
                         where=denom != 0), axis=2) / \
        (np.sum(segment_filters, axis=1)[:, 0])[:, np.newaxis], axis=0)
    return curr_lambdas

def calc_new_lambdas(curr_lambdas, step_size, gamma_target_lambdas):

    lambdas = curr_lambdas * (1.0 - step_size) + (step_size * gamma_target_lambdas)
    return lambdas

