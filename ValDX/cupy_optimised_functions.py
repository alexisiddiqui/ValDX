# functions to be optimised later - these should be mostly numpy operations

import cupy as np

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
    
    init_lnpi = np.asarray(init_lnpi)
    lambdas = np.asarray(lambdas)
    weights = np.asarray(weights)
    
    # This is the same ave_lnpi calculated in the reweighting.py code but not broadcast to the full 3D array
    ave_lnpi = np.sum(weights * init_lnpi, axis=1)
    meanbias = -kT * np.sum(lambdas * ave_lnpi)
    lambdas_2d = lambdas[:, np.newaxis] if lambdas.ndim == 1 else lambdas
    biaspot = -kT * np.sum(lambdas_2d.T * init_lnpi, axis=0)
    work = np.sum(weights * np.exp((biaspot - meanbias) / kT))
    work = kT * np.log(work)
    work = np.asnumpy(work)
    return work


# def calc_trial_dfracs(ave_lnpi, segment_filters, filtered_minuskt, filtered_exp_dfracs, n_datapoints):
#     """For a trial parameter move, calculate deuterated fractions and mean square error to target data using
#        the given average ln(protection factors). Average protection factors should be a 3D array of shape
#        [n_segments, n_residues, n_times]. This is filtered to calculate the by-segment averages using the provided
#        segment_filters Boolean array. The provided filtered_minuskt array (pre-filtered by the same segment_filters array)
#        is then used to calculate the deuterated fractions. The provided filtered_exp_dfracs array and n_datapoints are
#        then used to calculate the MSE to the target experimental data.

#        Requires current average ln(protection factors), segment filters, pre-filtered -kt rate constants, pre-filtered
#        target experimental deuterated fractions, and total number of datapoints

#        Usage: calc_trial_dfracs(ave_lnpi, segment_filters, filtered_minuskt, filtered_exp_dfracs, n_datapoints)

#        Returns: residue_dfracs, segment_dfracs, MSE_to_target"""
#     ave_lnpi = np.asarray(ave_lnpi)
#     segment_filters = np.asarray(segment_filters)
#     filtered_minuskt = np.asarray(filtered_minuskt)
#     filtered_exp_dfracs = np.asarray(filtered_exp_dfracs)
#     n_datapoints = np.asarray(n_datapoints)

#     # recalculate the deuterated fractions and MSE with the given ave_lnpi
#     denom = ave_lnpi * segment_filters
#     residue_dfracs = 1.0 - \
#                      np.exp(np.divide(filtered_minuskt, np.exp(denom),
#                                       out=np.full(filtered_minuskt.shape, np.nan),
#                                       where=denom != 0))

#     segment_dfracs = np.nanmean(residue_dfracs, axis=1)
#     segment_dfracs = segment_dfracs[:, np.newaxis, :].repeat(segment_filters.shape[1], axis=1)
#     MSE = np.sum((segment_dfracs * segment_filters
#                   - filtered_exp_dfracs) ** 2) / n_datapoints
    
#     residue_dfracs = np.asnumpy(residue_dfracs)
#     segment_dfracs = np.asnumpy(segment_dfracs)
#     MSE = np.asnumpy(MSE)

#     return residue_dfracs, segment_dfracs, MSE

def calc_bias_factor_mc(lambdas_c, contacts, lambdas_h, h_bonds):
    # biasfactor = np.sum(self.mcsamplvalues['lambdas_c'][:, np.newaxis] * self.runvalues['contacts']
    #                 + self.mcsamplvalues['lambdas_h'][:, np.newaxis] * self.runvalues['hbonds'],
    #                 axis=0) 
    lambdas_c = np.asarray(lambdas_c)
    contacts = np.asarray(contacts)
    lambdas_h = np.asarray(lambdas_h)
    h_bonds = np.asarray(h_bonds)

    biasfactor = np.sum(lambdas_c[:, np.newaxis] * contacts
                        + lambdas_h[:, np.newaxis] * h_bonds,
                        axis=0)
    
    biasfactor = np.asnumpy(biasfactor)

    return biasfactor

def calc_bias_factor_grad(lambdas, lnpi):
    lambdas = np.asarray(lambdas)
    lnpi = np.asarray(lnpi)

    biasfactor = np.sum(lambdas[:, np.newaxis] * lnpi, axis=0)
    biasfactor = np.asnumpy(biasfactor)
    return biasfactor

def calc_new_weight(iniweights, biasfactor):
    iniweights = np.asarray(iniweights)
    biasfactor = np.asarray(biasfactor)

    newweights = iniweights * np.exp(-biasfactor)
    newweights = newweights / np.sum(newweights)
    newweights = np.asnumpy(newweights)
    return newweights




def calc_ave_lnpi(weights, lnpi):
    weights = np.asarray(weights)
    lnpi = np.asarray(lnpi)

    ave_lnpi = np.sum(weights * lnpi, axis=1)
    ave_lnpi = np.asnumpy(ave_lnpi)
    return ave_lnpi


# def calc_curr_residue_dfracs(ave_lnpi, minuskt, segment_filters):
#     ave_lnpi = np.asarray(ave_lnpi)
#     minuskt = np.asarray(minuskt)
#     segment_filters = np.asarray(segment_filters)

#     denom = ave_lnpi * segment_filters
#     residue_dfracs = 1.0 - \
#                      np.exp(np.divide(minuskt, np.exp(denom),
#                                       out=np.full(minuskt.shape, np.nan),
#                                       where=denom != 0))
#     residue_dfracs = np.asnumpy(residue_dfracs)

#     return residue_dfracs


def calc_segment_and_MSE_from_residue_dfracs(residue_dfracs, segment_filters, filtered_exp_dfracs, n_datapoints):
    segment_filters = np.asarray(segment_filters)
    filtered_exp_dfracs = np.asarray(filtered_exp_dfracs)
    n_datapoints = np.asarray(n_datapoints)

    segment_dfracs = np.nanmean(residue_dfracs, axis=1)
    segment_dfracs = segment_dfracs[:, np.newaxis, :].repeat(segment_filters.shape[1], axis=1)
    MSE = np.sum((segment_dfracs * segment_filters
                  - filtered_exp_dfracs) ** 2) / n_datapoints
    segment_dfracs = np.asnumpy(segment_dfracs)
    MSE = np.asnumpy(MSE)

    return segment_dfracs, MSE


# def calc_lambda_from_segment_dfracs(ave_lnpi, segment_filters, segment_dfracs, filtered_exp_dfracs, filtered_minuskt):
#     # denom = self.runvalues['ave_lnpi'] * self.runvalues['segfilters']
#     # curr_lambdas = np.nansum(
#     #     np.sum((self.runvalues['curr_segment_dfracs'] * self.runvalues['segfilters'] - self.runvalues['exp_dfrac_filtered']) * \
#     #            np.exp(np.divide(self.runvalues['minuskt_filtered'], np.exp(denom),
#     #                             out=np.full(self.runvalues['minuskt_filtered'].shape, np.nan),
#     #                             where=denom != 0)) * \
#     #            np.divide(-self.runvalues['minuskt_filtered'], np.exp(denom),
#     #                      out=np.full(self.runvalues['minuskt_filtered'].shape, np.nan),
#     #                      where=denom != 0), axis=2) / \
#     #     (np.sum(self.runvalues['segfilters'], axis=1)[:, 0])[:, np.newaxis], axis=0)
#     # return curr_lambdas
#     ave_lnpi = np.asarray(ave_lnpi)
#     segment_filters = np.asarray(segment_filters)
#     segment_dfracs = np.asarray(segment_dfracs)
#     filtered_exp_dfracs = np.asarray(filtered_exp_dfracs)
#     filtered_minuskt = np.asarray(filtered_minuskt)

    
#     denom = ave_lnpi * segment_filters
#     curr_lambdas = np.nansum(
#         np.sum((segment_dfracs * segment_filters - filtered_exp_dfracs) * \
#                np.exp(np.divide(filtered_minuskt, np.exp(denom),
#                                 out=np.full(filtered_minuskt.shape, np.nan),
#                                 where=denom != 0)) * \
#                np.divide(-filtered_minuskt, np.exp(denom),
#                          out=np.full(filtered_minuskt.shape, np.nan),
#                          where=denom != 0), axis=2) / \
#         (np.sum(segment_filters, axis=1)[:, 0])[:, np.newaxis], axis=0)
#     curr_lambdas = np.asnumpy(curr_lambdas)

#     return curr_lambdas

def calc_new_lambdas(curr_lambdas, step_size, gamma_target_lambdas):
    # self.runvalues['lambdas'] = self.runvalues['lambdas'] * (1.0 - self.runvalues['curr_lambda_stepsize']) + \
    #                             (self.runvalues['curr_lambda_stepsize'] * gamma_target_lambdas)
    curr_lambdas = np.asarray(curr_lambdas)
    gamma_target_lambdas = np.asarray(gamma_target_lambdas)

    lambdas = curr_lambdas * (1.0 - step_size) + (step_size * gamma_target_lambdas)
    lambdas = np.asnumpy(lambdas)
    return lambdas

