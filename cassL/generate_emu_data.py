# It seems like tkgrid.py might have been intended to help me with this,
# but I don't know how to use it

# Short-cut:
# Documents\GitHub\Master\CAKE21\modded_repo

import sys, platform, os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import camb
from camb import model, initialpower, get_matter_power_interpolator

from cassL import camb_interface as ci
import copy as cp

cosm = ci.cosm
model0 = cosm.loc[0]

''' AndreaP thinks that npoints=300 should be a good balance of accuracy and
computability for our LH.'''
NPOINTS = 300

A_S_DEFAULT = 2.12723788013000E-09

import sys, traceback
import copy as cp

# These values help with the following function.
# However, neither of these belongs here, we should find a different home.
disregard_keys = ["OmB", "OmC", "OmM", "z(4)", "z(3)", "z(2)", "z(1)", "z(0)",
    "Lbox", "sigma8", "Name", "nnu_massive", "EOmDE"]

# For some reason, these keys are not printed by default, so we need to
# explicitly mention them.
# res_keys = ["sigma12", "omnuh2"]

def print_cosmology(cosmology):
    for key in cosmology.keys():
        if key not in disregard_keys:
            print(key, cosmology[key])
    #for key in res_keys:
    #    print(key, cosmology[key])


def build_cosmology(om_b_in, om_c_in, ns_in, sigma12_in, As_in, om_nu_in,
    param_ranges=None):
    # Use Aletheia model 0 as a base
    cosmology = cp.deepcopy(ci.cosm.iloc[0])

    cosmology["ombh2"] = om_b_in
    cosmology["omch2"] = om_c_in
    cosmology["n_s"] = ns_in
    # Incomplete
    cosmology["sigma12"] = sigma12_in
    cosmology["A_s"] = As_in

    if param_ranges is not None:
        prior = param_ranges["ombh2"]
        cosmology["ombh2"] = cosmology["ombh2"] * (prior[1] - prior[0]) + \
            prior[0]

        prior = param_ranges["omch2"]
        cosmology["omch2"] = cosmology["omch2"] * (prior[1] - prior[0]) + \
            prior[0]

        prior = param_ranges["n_s"]
        cosmology["n_s"] = cosmology["n_s"] * (prior[1] - prior[0]) + prior[0]

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

        if "A_s" in param_ranges:
            prior = param_ranges["A_s"]
            cosmology["A_s"] = cosmology["A_s"] * (prior[1] - prior[0]) + \
                prior[0]

    ''' Actually the last argument is not really important and is indeed just
        the default value. I'm writing this out explicitly because we're still
        in the debugging phase and so my code should always err on the verbose
        side.'''
    nnu_massive = 0 if om_nu_in == 0 else 1

    if param_ranges is not None and "omnuh2" in param_ranges:
        prior = param_ranges["omnuh2"]
        om_nu_in = om_nu_in * (prior[1] - prior[0]) + prior[0]

    return ci.specify_neutrino_mass(cosmology, om_nu_in,
        nnu_massive_in=nnu_massive)

def fill_hypercube(parameter_values, standard_k_axis,
    param_ranges=None, massive_neutrinos=True,
    cell_range=None, samples=None, write_period=None, save_label="unlabeled"):
    """
    @parameter_values: this should be a list of tuples to
        evaluate kp at.

    @cell_range adjust this value in order to pick up from where you
        left off, and to run this method in saveable chunks.

    BE CAREFUL! This function deliberately mutates the parameter_values object,
        replacing the target sigma12 values with the actual sigma12 values used.
    """
    if cell_range is None:
        cell_range = range(len(parameter_values))
    if samples is None:
        samples = np.zeros((len(parameter_values), NPOINTS))

    # Recent change: now omega_nu comes last

    bundle_parameters = lambda row: build_cosmology(row[0], row[1], row[2],
        row[3], row[4], row[5], param_ranges)

    if massive_neutrinos == False:
        bundle_parameters = lambda row: build_cosmology(row[0], row[1], row[2],
            row[3], A_S_DEFAULT, 0, param_ranges)

    # This just provides debugging information
    rescaling_parameters_list = None

    unwritten_cells = 0
    for i in cell_range:
        this_p = None
        this_cosmology = bundle_parameters(parameter_values[i])

        # We're only making the following switch in order to test out the
        # interpolator approach.
        #this_p, this_actual_sigma12, these_rescaling_parameters = \
        #    evaluate_cell(this_cosmology, standard_k_axis)
        this_p, this_actual_sigma12, these_rescaling_parameters = \
            interpolate_cell(this_cosmology, standard_k_axis,
            using_andrea_code=False)

        if rescaling_parameters_list is None:
            rescaling_parameters_list = these_rescaling_parameters
        else:
            rescaling_parameters_list = np.vstack((rescaling_parameters_list,
                these_rescaling_parameters))

        # We may actually want to remove this if-condition. For now, though, it
        # allows us to repeatedly evaluate a cosmology with the same
        # deterministic result.
        if this_actual_sigma12 is not None:
            parameter_values[i][3] = this_actual_sigma12

            if "sigma12" in param_ranges: # we have to normalize
                prior = param_ranges["sigma12"]
                this_normalized_actual_sigma12 = \
                    (this_actual_sigma12 - prior[0]) / (prior[1] - prior[0])
                parameter_values[i][3] = this_normalized_actual_sigma12
            #! Make the use of sigma12_2 more user-friendly
            elif "sigma12_2" in param_ranges: # we have to square and normalize
                this_actual_sigma12_2 = np.square(this_actual_sigma12)
                prior = param_ranges["sigma12_2"]
                this_normalized_actual_sigma12_2 = \
                    (this_actual_sigma12_2 - prior[0]) / (prior[1] - prior[0])
                parameter_values[i][3] = this_normalized_actual_sigma12_2

        samples[i] = this_p

        print(i, "complete")
        unwritten_cells += 1
        if write_period is not None and unwritten_cells >= write_period:
            np.save("samples_backup_i" + str(i) + "_" + save_label + ".npy",
                samples, allow_pickle=True)
            np.save("redshifts_backup_i" + str(i) + "_" + save_label + ".npy",
                rescaling_parameters_list, allow_pickle=True)
            np.save("hc_backup_i" + str(i) + "_" + save_label + ".npy",
                parameter_values, allow_pickle=True)
            unwritten_cells = 0
    return samples, rescaling_parameters_list

def evaluate_cell(input_cosmology, standard_k_axis, debug=False):
    """
    Returns the power spectrum in Mpc units and the actual sigma12_tilde value
        to which it corresponds.

    I concede that the function looks like a mess right now, with debug
    statements littered all over the place.
    """
    # This allows us to roughly find the z corresponding to the sigma12 that we
    # want.

    MEMNeC = cp.deepcopy(input_cosmology)
    MEMNeC['omch2'] += MEMNeC['omnuh2']
    MEMNeC['omnuh2'] = 0

    MEMNeC = ci.specify_neutrino_mass(MEMNeC,
        MEMNeC['omnuh2'], nnu_massive_in=0)

    _redshifts=np.flip(np.linspace(0, 10, 150))
   
    if debug:
        print("\nMEMNeC:")
        print_cosmology(MEMNeC)
        print("\nOriginal cosmology:")
        print_cosmology(input_cosmology)
        print("\n")

    _, _, _, list_sigma12 = ci.evaluate_cosmology(MEMNeC, _redshifts,
        fancy_neutrinos=False, k_points=NPOINTS, hubble_units=False)

    #print(list_s12)
    if debug:
        print("Maximum s12:", max(list_sigma12))
        import matplotlib.pyplot as plt
        # Original intersection problem we're trying to solve
        plt.plot(_redshifts, list_sigma12);
        plt.axhline(input_cosmology["sigma12"], c="black")
        plt.title("$\sigma_{12}$ vs. $z$")
        plt.ylabel("$\sigma_{12}$")
        plt.xlabel("$z$")
        plt.show()

    # remember that list_s12[0] corresponds to the highest value z
    if debug:
        print("Target sigma12:", input_cosmology["sigma12"])

    interpolator = interp1d(np.flip(list_sigma12), np.flip(_redshifts),
        kind='cubic')

    try:
        z_best = interpolator(input_cosmology["sigma12"])
    except ValueError:
        # we need to start playing with h.
        if input_cosmology['h'] <= 0.1:
            print("\nThis cell is hopeless. Here are the details:\n")
            print_cosmology(input_cosmology)
            print("\nThe extent of failure is:",
                abs(list_sigma12[len(list_sigma12) - 1] / \
                input_cosmology["sigma12"]) * 100, "%\n")
            return None, None, None

        input_cosmology['h'] -= 0.1
        return evaluate_cell(input_cosmology, standard_k_axis, debug)

    if debug:
        print("recommended redshift", z_best)

    p = np.zeros(len(standard_k_axis))

    k, _, p, actual_sigma12 = ci.evaluate_cosmology(input_cosmology,
        redshifts=np.array([z_best]), fancy_neutrinos=False,
        k_points=NPOINTS) 
    if input_cosmology['omnuh2'] != 0:
        _, _, _, actual_sigma12 = ci.evaluate_cosmology(MEMNeC,
            redshifts=np.array([z_best]), fancy_neutrinos=False,
            k_points=NPOINTS)
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


def interpolate_cell(input_cosmology, standard_k_axis, debug=False,
    using_andrea_code=True):
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

    if debug:
        print("\nMEMNeC:")
        print_cosmology(MEMNeC)
        print("\nOriginal cosmology:")
        print_cosmology(input_cosmology)
        print("\n")
    
    MEMNeC_p_interpolator = None
    if using_andrea_code:
        MEMNeC_p_interpolator = ci.andrea_interpolator(MEMNeC)
    else:
        MEMNeC_p_interpolator = ci.cosmology_to_PK_interpolator(MEMNeC,
            redshifts=_redshifts, kmax=10)
    list_sigma12 = np.array([
        ci.s12_from_interpolator(MEMNeC_p_interpolator, z) for z in _redshifts
    ])

    #print(list_s12)
    if debug:
        print("Maximum s12:", max(list_sigma12))
        import matplotlib.pyplot as plt
        # Original intersection problem we're trying to solve
        plt.plot(_redshifts, list_sigma12);
        plt.axhline(input_cosmology["sigma12"], c="black")
        plt.title("$\sigma_{12}$ vs. $z$")
        plt.ylabel("$\sigma_{12}$")
        plt.xlabel("$z$")
        plt.show()

    # remember that list_s12[0] corresponds to the highest value z
    if debug:
        print("Target sigma12:", input_cosmology["sigma12"])

    interpolator = interp1d(np.flip(list_sigma12), np.flip(_redshifts),
        kind='cubic')

    try:
        z_best = interpolator(input_cosmology["sigma12"])
    except ValueError:
        # we need to start playing with h.
        if input_cosmology['h'] <= 0.1:
            print("\nThis cell is hopeless. Here are the details:\n")
            print_cosmology(input_cosmology)
            print("\nThe extent of failure is:",
                abs(list_sigma12[len(list_sigma12) - 1] / \
                input_cosmology["sigma12"]) * 100, "%\n")
            return None, None, None

        input_cosmology['h'] -= 0.1
        return interpolate_cell(input_cosmology, standard_k_axis, debug,
            using_andrea_code)

    if debug:
        print("recommended redshift", z_best)

    p = np.zeros(len(standard_k_axis))

    p_interpolator = None
    if using_andrea_code:
        p_interpolator = ci.andrea_interpolator(input_cosmology)
    else:
        p_interpolator = ci.cosmology_to_PK_interpolator(input_cosmology,
            redshifts=np.array([z_best]), kmax=10)
    
    actual_sigma12 = ci.s12_from_interpolator(p_interpolator, z_best)

    if input_cosmology['omnuh2'] != 0:
        actual_sigma12 = ci.s12_from_interpolator(
            MEMNeC_p_interpolator, z_best)
    # De-nest
    # actual_sigma12 = actual_sigma12[0]

    if using_andrea_code:
        p = np.array([p_interpolator.P(z_best, k) for k in standard_k_axis])
    else:
        p = np.array([p_interpolator.P(z_best, k)[0] for k in standard_k_axis])
    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    return p, actual_sigma12, np.array((input_cosmology['h'], float(z_best)))
