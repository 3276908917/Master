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

MODEL0 = ci.cosm.loc[0]

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
        if 4 we're building a massless neutrino emulator
        if 6 we're building a massive neutrino emulator
        if 8, we're testing the full massive pipeline without w_a and w_0
        if 9, we're testing the full massless pipeline
        if 10, we're testing the full massive pipeline
    """
    # We should replace this function with a function that assumes, e.g.
    # index 0 is om_b, index 1 is om_c, etc.

    if len(lhs_row) not in [3, 4, 6, 8, 9, 10]:
        raise ValueError("The length of the input lhs row does not " + \
            "correspond to any of the known cases. Please refer to " + \
            "the docstring.")

    # Use Aletheia model 0 as a base
    cosmology = cp.deepcopy(MODEL0)

    cosmology["ombh2"] = lhs_row[0]
    cosmology["omch2"] = lhs_row[1]
    cosmology["n_s"] = lhs_row[2]

    # Incomplete
    if len(lhs_row) > 3:
        if len(lhs_row) > 7:
            cosmology["z"] = lhs_row[3]
        else:
            cosmology["sigma12"] = lhs_row[3]

    if len(lhs_row) > 4:
        cosmology["A_s"] = lhs_row[4]
        if len(lhs_row) != 9:
            last_i = len(lhs_row) - 1
            cosmology = ci.specify_neutrino_mass(cosmology, lhs_row[last_i], 1)
    else:
        cosmology["A_s"] = A_S_DEFAULT

    if "omnuh2" not in cosmology: 
        cosmology = ci.specify_neutrino_mass(cosmology, 0, 1)

    if len(lhs_row) > 7:
        cosmology["h"] = lhs_row[5]
        cosmology["OmK"] = lhs_row[6] / cosmology["h"] ** 2
        
        if len(lhs_row) > 8:
            cosmology["w0"] = lhs_row[7]
            cosmology["wa"] = lhs_row[8]
    
    return cosmology

def broadcast_unsolvable(input_cosmology, list_sigma12=None):
    print("\nThis cell cannot be solved with a nonnegative redshift.")
    print("This is the failed cosmology:\n")
    ui.print_cosmology(input_cosmology)
    
    if list_sigma12 is not None:
        print("\nThe closest feasible sigma12 value would yield an error of:",
            utils.percent_error(input_cosmology["sigma12"],
                list_sigma12[len(list_sigma12) - 1]), "%\n")

    return None, np.array([np.nan, np.nan, np.nan])

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
                
            if input_cosmology['h'] != MODEL0['h']: # we've touched h,
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

    if input_cosmology['h'] != MODEL0['h']: # announce that we've touched h
        print("We had to move h to", np.around(input_cosmology['h'], 3))

    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    return p, np.array((actual_sigma12, input_cosmology['h'], float(z_best)))


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
       
        try:
            MEMNeC_p_interpolator = ci.cosmology_to_Pk_interpolator(MEMNeC,
                    redshifts=_redshifts, kmax=k_max, hubble_units=False)

            s12intrp = ci.sigma12_from_interpolator
            sigma12 = lambda z: s12intrp(MEMNeC_p_interpolator, z)
            list_sigma12 = np.array([sigma12(z) for z in _redshifts])
        except Exception:
            return broadcast_unsolvable(input_cosmology, list_sigma12)

        interpolator = interp1d(np.flip(list_sigma12), np.flip(_redshifts),
            kind='cubic')

        try:
            z_best = interpolator(input_cosmology["sigma12"])
            interpolation_redshifts = np.flip(np.linspace(max(0, z_best - 1),
                                                          z_best + 1, 150))

            get_intrp = ci.cosmology_to_Pk_interpolator
            p_intrp = get_intrp(input_cosmology,
                                redshifts=interpolation_redshifts,
                                kmax=k_max, hubble_units=False)
            p = p_intrp.P(z_best, standard_k_axis)

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

    actual_sigma12 = ci.sigma12_from_interpolator(MEMNeC_p_interpolator,
                                                  z_best)

    if input_cosmology['h'] != MODEL0['h']: # announce that we've touched h
        print("We had to move h to", np.around(input_cosmology['h'], 3))

    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    return p, np.array((actual_sigma12, input_cosmology['h'], float(z_best)))
    

def interpolate_nosigma12(input_cosmology, standard_k_axis):
    """
    Returns the power spectrum in Mpc units and the actual sigma12_tilde value
        to which it corresponds.

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
    #! Hard code
    k_max = 1.01 * np.max(standard_k_axis)
    z = input_cosmology["z"]

    get_intrp = ci.cosmology_to_Pk_interpolator
    
    interpolation_redshifts = np.flip(np.linspace(max(0, z - 1), z + 1, 150))
    p_intrp = get_intrp(input_cosmology, redshifts=interpolation_redshifts,
                        kmax=k_max, hubble_units=False)
    p = p_intrp.P(z, standard_k_axis)

    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.
    return p, None


def fill_hypercube_with_sigma12(lhs, priors=None, samples=None,
                                write_period=None,
                                save_label="sigma12_backup_i{}",
                                crash_when_unsolvable=False):
    def eval_func(cosmology):
        # De-nesting
        return ci.evaluate_sigma12(cosmology, [12.], [1.])[0][0]

    if cell_range is None:
        cell_range = range(len(lhs))
    if samples is None:            
        samples = np.zeros(len(lhs))

    unwritten_cells = 0
    for i in cell_range:
        this_denormalized_row = denormalize_row(lhs[i], priors)
        this_cosmology = build_cosmology(this_denormalized_row)

        try:
            samples[i] = eval_func(this_cosmology)
        except camb.CAMBError:
            print("This cell is unsolvable. Since this function requires " + \
                  "no rescaling, your priors are probably extreme.")
            if crash_when_unsolvable:
                raise ValueError("Cell unsolvable.")

        print(i, "complete")
        unwritten_cells += 1

        if write_period is not None and unwritten_cells >= write_period:
            # We add one because the current cell is also unwritten
            np.save(save_label.format(i), samples)
            unwritten_cells = 0

    return samples


def fill_hypercube_with_Pk(lhs, standard_k_axis, priors=None,
                           eval_func=direct_eval_cell, cell_range=None,
                           write_period=None, save_label="Pk",
                           crash_when_unsolvable=False):
    """
    @lhs: this is a list of tuples with which @eval_func is to be evaluated.

    @cell_range: a range object specifying the indices of lhs which still need
        to be evaluated. By default, it is None, which means that the entire
        lhs will be evaluated. This parameter can be used to pick up from where
        previous runs left off, and to run this method in saveable chunks.
    """
    if cell_range is None:
        cell_range = range(len(lhs))

    samples = np.zeros((len(lhs), len(standard_k_axis)))

    # The rescaling parameters are true_sigma12, h and z. Only the sigma12
    # value is used to train the emu but the rest provide debugging information
    # and sure that the output spectra are easily reproducible.
    rescalers_arr = np.zeros((len(lhs), 3))

    unwritten_cells = 0
    for i in cell_range:
        this_p = None
        this_denormalized_row = denormalize_row(lhs[i], priors)
        this_cosmology = build_cosmology(this_denormalized_row)

        try:
            samples[i], rescalers_arr[i] = \
                eval_func(this_cosmology, standard_k_axis)

            if samples[i] is None and crash_when_unsolvable:
                raise ValueError("Cell unsolvable.")
        except camb.CAMBError:
            print("This cell is unsolvable. However, in this case, we " + \
                  "observed a CAMBError rather than a negative redshift. " + \
                  "This suggests that there is a problem with the input " + \
                  "hypercube.")
            if crash_when_unsolvable:
                raise ValueError("Cell unsolvable.")

        actual_sigma12 = rescalers_arr[i][0]
        if not np.isnan(actual_sigma12): # we need to re-normalize
            prior = priors[3]
            normalized = (actual_sigma12 - prior[0]) / (prior[1] - prior[0])
            rescalers_arr[i][0] = normalized

        print(i, "complete")
        unwritten_cells += 1
        if write_period is not None and unwritten_cells >= write_period:
            # We add one because the current cell is also unwritten
            save_start = i - unwritten_cells + 1
            save_end = i + 1

            file_suffix = "_backup_i{}_through_{}_{}.npy"
            file_suffix = file_suffix.format(save_start, i, save_label)

            np.save("Pk" + file_suffix, samples[save_start:save_end])
            np.save("rescalers" + file_suffix,
                    rescalers_arr[save_start:save_end])

            unwritten_cells = 0

    return samples, rescalers_arr


def fill_hypercube_with_sigmaR(lhs, R_axis, priors=None, cell_range=None,
                               write_period=None, save_label="unlabeled",
                               crash_when_unsolvable=False):
    """
    @lhs: this is a list of tuples with which @eval_func is to be evaluated.

    @cell_range: a range object specifying the indices of lhs which still need
        to be evaluated. By default, it is None, which means that the entire
        lhs will be evaluated. This parameter can be used to pick up from where
        previous runs left off, and to run this method in saveable chunks.
    """
    if cell_range is None:
        cell_range = range(len(lhs))

    samples = np.zeros((len(lhs), len(R_axis)))

    unwritten_cells = 0
    for i in cell_range:
        this_denormalized_row = denormalize_row(lhs[i], priors)
        this_cosmology = build_cosmology(this_denormalized_row)

        try:
            samples[i] = ci.evaluate_sigmaR(this_cosmology, R_axis, [1.])[0]

            if samples[i] is None and crash_when_unsolvable:
                raise ValueError("Cell unsolvable.")
        except camb.CAMBError:
            print("This cell is unsolvable. Since this function requires " + \
                  "no rescaling, your priors are probably extreme.")
            if crash_when_unsolvable:
                raise ValueError("Cell unsolvable.")

        print(i, "complete")
        unwritten_cells += 1
        if write_period is not None and unwritten_cells >= write_period:
            # We add one because the current cell is also unwritten
            save_start = i - unwritten_cells + 1
            save_end = i + 1

            file_suffix = "_backup_i{}_through_{}_{}.npy"
            file_suffix = file_suffix.format(save_start, i, save_label)

            np.save("sigmaR" + file_suffix, samples[save_start:save_end])

            unwritten_cells = 0

    return samples
