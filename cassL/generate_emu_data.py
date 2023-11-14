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

def denormalize_row(lhs_row, param_ranges):
    param_ranges = param_ranges[:len(lhs_row)]
    xrange = np.ptp(param_ranges, axis=1)
    xmin = np.min(param_ranges, axis=1)
    return lhs_row * xrange + xmin

    # Some outdated sigma12 nonlinear sampling code which was probably wrong
    # even before it needed to be updated to the current setup

    if "sigma12" in param_ranges:
        prior = param_ranges["sigma12"]
        cosmology["sigma12"] = cosmology["sigma12"] * \
            (prior[1] - prior[0]) + prior[0]
    elif "sigma12_2" in param_ranges:
        #! Make the use of sigma12_2 more user-friendly
        prior = param_ranges["sigma12_2"]
        # sigma12_in actually describes a sigma12_2 value
        sigma12_2 = sigma12_in * (prior[1] - prior[0]) + prior[0]
        cosmology["sigma12"] = np.sqrt(sigma12_2)
    elif "sigma12_root" in param_ranges:
        #! Make the use of sigma12_2 more user-friendly
        prior = param_ranges["sigma12_root"]
        # sigma12_in actually describes a sigma12_2 value
        sigma12_root = sigma12_in * (prior[1] - prior[0]) + prior[0]
        cosmology["sigma12"] = np.square(sigma12_root)


def build_cosmology(lhs_row, param_ranges=None):
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
        return ci.specify_neutrino_mass(cosmology, 0)


def direct_eval_cell(input_cosmology, standard_k_axis, debug=False):
    """
    Returns the power spectrum in Mpc units and the actual sigma12_tilde value
        to which it corresponds.

    I concede that the function looks like a mess right now, with debug
    statements littered all over the place.
    """
    num_k_points = len(standard_k_axis)
    # This allows us to roughly find the z corresponding to the sigma12 that
    # we want.

    MEMNeC = cp.deepcopy(input_cosmology)
    MEMNeC['omch2'] += MEMNeC['omnuh2']
    MEMNeC['omnuh2'] = 0

    MEMNeC = ci.specify_neutrino_mass(MEMNeC, MEMNeC['omnuh2'],
        nnu_massive_in=0)

    _redshifts=np.flip(np.linspace(0, 10, 150))

    _, _, _, list_sigma12 = ci.evaluate_cosmology(MEMNeC, _redshifts,
        fancy_neutrinos=False, k_points=num_k_points, hubble_units=False)

    interpolator = interp1d(np.flip(list_sigma12), np.flip(_redshifts),
        kind='cubic')

    try:
        z_best = interpolator(input_cosmology["sigma12"])
    except ValueError:
        # we need to start playing with h.
        if input_cosmology['h'] <= 0.1:
            print("\nThis cell cannot be solved with a nonnegative redshift.")
            print("This is the failed cosmology:\n")
            ui.print_cosmology(input_cosmology)
            print("\nThe closest feasible sigma_12 value is off by:",
                abs(list_sigma12[len(list_sigma12) - 1] / \
                input_cosmology["sigma12"]) * 100, "%\n")
            return None, None, np.array([np.nan, np.nan])

        input_cosmology['h'] -= 0.1
        return direct_eval_cell(input_cosmology, standard_k_axis, debug)

    p = np.zeros(len(standard_k_axis))

    k, _, p, actual_sigma12 = ci.evaluate_cosmology(input_cosmology,
        redshifts=np.array([z_best]), fancy_neutrinos=False,
        k_points=num_k_points) 
    if input_cosmology['omnuh2'] != 0:
        _, _, _, actual_sigma12 = ci.evaluate_cosmology(MEMNeC,
            redshifts=np.array([z_best]), fancy_neutrinos=False,
            k_points=num_k_points)
    # De-nest
    actual_sigma12 = actual_sigma12[0]

    if input_cosmology['h'] != model0['h']: # we've touched h,
        # we need to interpolate
        print("We had to move h to", np.around(input_cosmology['h'], 3))

        interpolator = interp1d(k, p, kind="cubic")
        p = interpolator(standard_k_axis)

    if len(p) == 1: # then de-nest
        p = p[0]

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

    MEMNeC = cp.deepcopy(input_cosmology)
    MEMNeC['omch2'] += MEMNeC['omnuh2']
    MEMNeC['omnuh2'] = 0

    MEMNeC = ci.specify_neutrino_mass(MEMNeC,
        MEMNeC['omnuh2'], nnu_massive_in=0)

    _redshifts=np.flip(np.linspace(0, 10, 150))

    MEMNeC_p_interpolator = ci.cosmology_to_PK_interpolator(MEMNeC,
            redshifts=_redshifts, kmax=10)
    list_sigma12 = np.array([
        ci.s12_from_interpolator(MEMNeC_p_interpolator, z) for z in _redshifts
    ])

    interpolator = interp1d(np.flip(list_sigma12), np.flip(_redshifts),
        kind='cubic')

    try:
        z_best = interpolator(input_cosmology["sigma12"])
    except ValueError:
        # we need to start playing with h.
        if input_cosmology['h'] <= 0.1:
            print("\nThis cell cannot be solved with a nonnegative redshift.")
            print("This is the failed cosmology:\n")
            ui.print_cosmology(input_cosmology)
            print("\nThe closest feasible sigma_12 value is off by:",
                abs(list_sigma12[len(list_sigma12) - 1] / \
                input_cosmology["sigma12"]) * 100, "%\n")
            return None, None, np.array([np.nan, np.nan])

        input_cosmology['h'] -= 0.1
        return interpolate_cell(input_cosmology, standard_k_axis)

    p = np.zeros(len(standard_k_axis))

    p_interpolator = ci.cosmology_to_PK_interpolator(input_cosmology,
            redshifts=np.array([z_best]), kmax=10)

    actual_sigma12 = ci.s12_from_interpolator(p_interpolator, z_best)

    if input_cosmology['omnuh2'] != 0:
        actual_sigma12 = ci.s12_from_interpolator(
            MEMNeC_p_interpolator, z_best)

    p = np.array([p_interpolator.P(z_best, k)[0] for k in standard_k_axis])
    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    return p, actual_sigma12, np.array((input_cosmology['h'], float(z_best)))


def fill_hypercube(lhs, standard_k_axis, priors=None,
    eval_func=direct_eval_cell, cell_range=None, samples=None,
    write_period=None, save_label="unlabeled"):
    """
    @lhs: this should be a list of tuples to
        evaluate kp at.
        #! This is confusing with the param_ranges label. Maybe we can call it
            cosmo_configs or something?

    @cell_range adjust this value in order to pick up from where you
        left off, and to run this method in saveable chunks.

    BE CAREFUL! This function deliberately mutates the lhs object,
        replacing the target sigma12 values with the actual sigma12 values used.
    """
    if cell_range is None:
        cell_range = range(len(lhs))
    if samples is None:
        if len(lhs[0]) == 4 or len(lhs[0]) == 6:
            samples = np.zeros((len(lhs), len(standard_k_axis)))
        elif len(lhs[0]) == 3:
            samples = np.zeros(len(lhs))

    # Recent change: now omega_nu comes last
    
    bundle_parameters = None
    
    # We don't need this check because we can always just truncate the prior
    # array.
    #if len(lhs[0]) != len(priors):
    #    raise ValueError("Dimension disagreement between priors and LHS.")

    # This just provides debugging information
    rescaling_parameters_list = None

    unwritten_cells = 0
    for i in cell_range:
        this_p = None
        this_denormalized_row = denormalize_row(lhs[i], param_ranges=priors)
        this_cosmology = build_cosmology(this_denormalized_row)

        # We're only making the following switch in order to test out the
        # interpolator approach.
        #this_p, this_actual_sigma12, these_rescaling_parameters = \
        #    direct_eval_cell(this_cosmology, standard_k_axis)

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
        except camb.CAMBError:
            print("This cell is unsolvable. However, in this case, we " + \
                  "observed a CAMBError rather than a negative redshift. " + \
                  "This suggests that there is a problem with the input " + \
                  "hypercube.")

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

                if "sigma12" in priors: # we have to normalize
                    prior = priors["sigma12"]
                    this_normalized_actual_sigma12 = \
                        (this_actual_sigma12 - prior[0]) / (prior[1] - prior[0])
                    lhs[i][3] = this_normalized_actual_sigma12
                #! Make the use of sigma12_2 more user-friendly
                elif "sigma12_2" in priors: # we have to square and normalize
                    this_actual_sigma12_2 = np.square(this_actual_sigma12)
                    prior = priors["sigma12_2"]
                    this_normalized_actual_sigma12_2 = \
                        (this_actual_sigma12_2 - prior[0]) / (prior[1] - prior[0])
                    lhs[i][3] = this_normalized_actual_sigma12_2
                elif "sigma12_root" in priors: # we have to square and
                    # normalize
                    this_actual_sigma12_root = np.sqrt(this_actual_sigma12)
                    prior = priors["sigma12_root"]
                    this_normalized_actual_sigma12_root = \
                        (this_actual_sigma12_root - prior[0]) / \
                            (prior[1] - prior[0])
                    lhs[i][3] = this_normalized_actual_sigma12_root

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
