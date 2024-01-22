import sys, platform, os
import traceback
import copy as cp

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import camb
from camb import model, initialpower, get_matter_power_interpolator

from cassL import camb_interface as ci
from cassL import user_interface as ui
from cassL import utils

model0 = ci.cosm.loc[0]

A_S_DEFAULT = 2.12723788013E-09

def denormalize_row(lhs_row, priors):
    # Truncate the prior if this is a massless-neutrino cosmology, so that we
    # don't receive NumPy dimensionality complaints.
    priors = priors[:len(lhs_row)]
    
    xrange = np.ptp(priors, axis=1)
    xmin = np.min(priors, axis=1)
    return lhs_row * xrange + xmin
    

def build_cosmology(lhs_row):
    """
    Intended behavior:
        if len(lhs_row) == 3 we're building a sigma12 emulator
        if len(lhs_row) == 4 we're building a massless neutrino emulator
        if len(lhs_row) == 6 we're building a massive neutrino emulator
    """
    # We should replace this function with a function that assumes, e.g.
    # index 0 is om_b, index 1 is om_c, etc.

    if len(lhs_row) not in [3, 4, 6]:
        raise ValueError("The length of the input lhs row does not" + \
            "correspond to any of the three known cases. Please refer to " + \
            "the docstring.")

    # Use Aletheia model 0 as a base
    cosmology = cp.deepcopy(model0)

    cosmology["ombh2"] = lhs_row[0]
    cosmology["omch2"] = lhs_row[1]
    cosmology["n_s"] = lhs_row[2]

    # Incomplete
    if len(lhs_row) > 3:
        cosmology["sigma12"] = lhs_row[3]

    if len(lhs_row) > 4:
        cosmology["A_s"] = lhs_row[4]
        return ci.specify_neutrino_mass(cosmology, lhs_row[5])
    else:
        cosmology["A_s"] = A_S_DEFAULT
        return ci.specify_neutrino_mass(cosmology, 1)


def broadcast_unsolvable(input_cosmology, list_sigma12):
    print("\nThis cell cannot be solved with a nonnegative redshift.")
    print("This is the failed cosmology:\n")
    ui.print_cosmology(input_cosmology)
    print("\nThe closest feasible sigma12 value would yield an error of:",
        utils.percent_error(input_cosmology["sigma12"],
            list_sigma12[len(list_sigma12) - 1]), "%\n")

    return None, None, np.array([np.nan, np.nan])

def direct_eval_cell(input_cosmology, standard_k_axis):
    """
    Returns the power spectrum in Mpc units and the actual sigma12_tilde value
        to which it corresponds.
    """
    num_k_points = len(standard_k_axis)
    
    # redshifts at which to test the cosmology, to best match desired sigma12
    _redshifts=np.flip(np.geomspace(1, 11, 150) - 1)

    z_best = None
    p = None
    list_sigma12 = [None]
    
    while True:
        MEMNeC = ci.balance_neutrinos_with_CDM(input_cosmology, 0)

        # Attempt to solve for the power spectrum at all test redshifts. This
        # call to CAMB shouldn't throw an error on the first iteration of this
        # loop, but as you'll see a few lines later, we may begin to decrease
        # h in order to match the desired sigma12 value. This can sometimes
        # cause CAMB to break even if we satisfy CAMB's nominal requirement
        # that h >= 0.01.
        try:
            _, _, _, list_sigma12 = ci.evaluate_cosmology(MEMNeC, _redshifts,
                fancy_neutrinos=False, k_points=num_k_points,
                hubble_units=False)
        except Exception:
            return broadcast_unsolvable(input_cosmology, list_sigma12)

        interpolator = interp1d(np.flip(list_sigma12), np.flip(_redshifts),
            kind='cubic')

        try:
            # In the vast majority of cases, the following line will generate
            # the ValueError caught in the except clause. However, we have
            # encountered one case where interpolating the power spectrum
            # generated an error. We're including that code here, too, because
            # the solution appears to be the same: just decrease h.
            z_best = interpolator(input_cosmology["sigma12"])
            
            k, _, p, actual_sigma12 = ci.evaluate_cosmology(input_cosmology,
                redshifts=np.array([z_best]), fancy_neutrinos=False,
                k_points=num_k_points) 
                
            if input_cosmology['h'] != model0['h']: # we've touched h,
                # we need to interpolate
                interpolator = interp1d(k, p, kind="cubic")
                p = interpolator(standard_k_axis)
            
            break
            
        except ValueError:
            # we need to start playing with h.
            if input_cosmology['h'] > 0.1:
                input_cosmology['h'] -= 0.1
            elif input_cosmology['h'] >= 0.02:
                # Finer-grained decreases will save the most extreme
                # cosmologies in our priors
                input_cosmology['h'] -= 0.01
            else: # We can't decrease h any further.
                return broadcast_unsolvable(input_cosmology, list_sigma12)
    
    if len(p) == 1: # de-nest the power spectrum
        p = p[0]
    
    # If neutrinos are not massless, we have to run again in order to
    # get the correct sigma12 value...
    if input_cosmology['omnuh2'] != 0:
        _, _, _, actual_sigma12 = ci.evaluate_cosmology(MEMNeC,
            redshifts=np.array([z_best]), fancy_neutrinos=False,
            k_points=num_k_points)
    # De-nest
    actual_sigma12 = actual_sigma12[0]

    if input_cosmology['h'] != model0['h']: # announce that we've touched h
        print("We had to move h to", np.around(input_cosmology['h'], 3))

    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    return p, actual_sigma12, np.array((input_cosmology['h'], float(z_best)))


def interpolate_cell(input_cosmology, standard_k_axis):
    """
    Returns the power spectrum in Mpc units and the actual sigma12_tilde value
        to which it corresponds.

    I concede that the function looks like a mess right now, with debug
    statements littered all over the place.

    This is a demo function until we figure out how to apply the interpolation
    approach to the generation of emu data. Once we have that, we can figure
    out how to re-combine this function with the previous one.

    Possible issues:
    * Should we regenerate the MEMNeC interpolator at the end (i.e., with just
        one redshift value rather than 150), to get better resolution? Or is it
        fine to re-use?
    """
    # This allows us to roughly find the z corresponding to the sigma12 that we
    # want.
    _redshifts=np.flip(np.geomspace(1, 11, 150) - 1)
    z_best = None
    p = None
    list_sigma12 = [None]
    #! Hard code
    k_max = 1.01 * max(standard_k_axis)

    while True:
        MEMNeC = ci.balance_neutrinos_with_CDM(input_cosmology, 0)
       
        #try:
        #! Hard code
        MEMNeC_p_interpolator = ci.cosmology_to_PK_interpolator(MEMNeC,
                redshifts=_redshifts, kmax=k_max)
        
        s12intrp = ci.sigma12_from_interpolator
        sigma12 = lambda z: s12intrp(MEMNeC_p_interpolator, z)
        list_sigma12 = np.array([sigma12(z) for z in _redshifts])
        #except Exception:
        #    return broadcast_unsolvable(input_cosmology, list_sigma12)

        interpolator = interp1d(np.flip(list_sigma12), np.flip(_redshifts),
            kind='cubic')

        try:
            z_best = interpolator(input_cosmology["sigma12"])
            p_interpolator = ci.cosmology_to_PK_interpolator(input_cosmology,
                redshifts=np.array([z_best]), kmax=k_max)
            # The [0] is a de-nesting step, for cleaner formatting.
            p_at_k = lambda k: p_interpolator.P(z_best, k)[0]
            p = np.array([p_at_k(k) for k in standard_k_axis])

            break

        except ValueError:
            # we need to start playing with h.
            if input_cosmology['h'] > 0.1:
                input_cosmology['h'] -= 0.1
            elif input_cosmology['h'] >= 0.02:
                # Finer-grained decreases might save a couple of weird cosmologies
                input_cosmology['h'] -= 0.01
            else: # We can't decrease any further.
                return broadcast_unsolvable(input_cosmology, list_sigma12)

    actual_sigma12 = None

    if input_cosmology['omnuh2'] != 0:
        actual_sigma12 = ci.sigma12_from_interpolator(
            MEMNeC_p_interpolator, z_best)
    else:
        actual_sigma12 = ci.sigma12_from_interpolator(p_interpolator, z_best)


    if input_cosmology['h'] != model0['h']: # announce that we've touched h
        print("We had to move h to", np.around(input_cosmology['h'], 3))

    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    return p, actual_sigma12, np.array((input_cosmology['h'], float(z_best)))


def fill_hypercube(lhs, standard_k_axis, priors=None,
    eval_func=direct_eval_cell, cell_range=None, samples=None,
    write_period=None, save_label="unlabeled", crash_when_unsolvable=False):
    """
    @lhs: this is a list of tuples with which @eval_func is to be evaluated.

    @cell_range: a range object specifying the indices of lhs which still need
        to be evaluated. By default, it is None, which means that the entire
        lhs will be evaluated. This parameter can be used to pick up from where
        previous rusn left off, and to run this method in saveable chunks.

    BE CAREFUL! This function deliberately mutates the lhs object,
        replacing the target sigma12 values with the actual sigma12 values
        used.
    """
    if cell_range is None:
        cell_range = range(len(lhs))
    if samples is None:
        if len(lhs[0]) == 4 or len(lhs[0]) == 6:
            samples = np.zeros((len(lhs), len(standard_k_axis)))
        elif len(lhs[0]) == 3:
            samples = np.zeros(len(lhs))
    
    bundle_parameters = None

    # The rescaling parameters are h and z, so this will be an array of shape
    # (num_spectra, 2). These parameters are not used to train the emu but
    # rather provide debugging information and sure that the output spectra
    # are easily reproducible.
    rescaling_parameters_list = None

    unwritten_cells = 0
    for i in cell_range:
        this_p = None
        this_denormalized_row = denormalize_row(lhs[i], priors)
        this_cosmology = build_cosmology(this_denormalized_row)

        # As of 20.08.2023, this REALLY is a NECESSARY workaround.
        this_actual_sigma12 = None
        these_rescaling_parameters = np.array([np.nan, np.nan])
        try:
            if len(lhs[0]) == 4 or len(lhs[0]) == 6:
                # we're emulating power spectra
                this_p, this_actual_sigma12, these_rescaling_parameters = \
                    eval_func(this_cosmology, standard_k_axis)
            elif len(lhs[0]) == 3: # we're emulating sigma12
                samples[i] = eval_func(this_cosmology)

            # We make sure that len(lhs[0] > 3) because this_p is *supposed* to
            # be None in the event that we're emulating sigma12.
            if (this_p is None and len(lhs[0] > 3)) and crash_when_unsolvable:
                raise ValueError("Cell unsolvable.")
        except camb.CAMBError:
            print("This cell is unsolvable. However, in this case, we " + \
                  "observed a CAMBError rather than a negative redshift. " + \
                  "This suggests that there is a problem with the input " + \
                  "hypercube.")
            if crash_when_unsolvable:
                raise ValueError("Cell unsolvable.")

        if rescaling_parameters_list is None:
            rescaling_parameters_list = these_rescaling_parameters
        else:
            rescaling_parameters_list = np.vstack((rescaling_parameters_list,
                these_rescaling_parameters))

        # We may actually want to remove this if-condition. For now, though, it
        # allows us to repeatedly evaluate a cosmology with the same
        # deterministic result.
        if len(lhs[0]) == 4 or len(lhs[0]) == 6:
            if this_actual_sigma12 is not None:
                lhs[i][3] = this_actual_sigma12

                prior = priors[3]
                this_normalized_actual_sigma12 = \
                    (this_actual_sigma12 - prior[0]) / (prior[1] - prior[0])
                lhs[i][3] = this_normalized_actual_sigma12

            samples[i] = this_p

        print(i, "complete")
        unwritten_cells += 1
        if write_period is not None and unwritten_cells >= write_period:
            np.save("samples_backup_i" + str(i) + "_" + save_label + ".npy",
                samples, allow_pickle=True)

            if len(lhs[0]) != 3:
                np.save("rescalers_backup_i" + str(i) + "_" + save_label + \
                        ".npy", rescaling_parameters_list, allow_pickle=True)
                np.save("hc_backup_i" + str(i) + "_" + save_label + ".npy",
                        lhs, allow_pickle=True)

                unwritten_cells = 0

    return samples, rescaling_parameters_list
