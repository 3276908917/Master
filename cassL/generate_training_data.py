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

import camb_interface as ci
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

def build_cosmology(om_b_in, om_c_in, ns_in, sigma12_in, As_in, om_nu_in):
    # Use Aletheia model 0 as a base
    cosmology = cp.deepcopy(ci.cosm.iloc[0])
    
    cosmology["ombh2"] = om_b_in
    cosmology["omch2"] = om_c_in
    cosmology["n_s"] = ns_in
    # Incomplete
    cosmology["sigma12"] = sigma12_in
    cosmology["A_s"] = As_in

    ''' Actually the last argument is not really important and is indeed just
        the default value. I'm writing this out explicitly because we're still
        in the debugging phase and so my code should always err on the verbose
        side.'''
    nnu_massive = 0 if om_nu_in == 0 else 1

    return ci.specify_neutrino_mass(cosmology, om_nu_in,
        nnu_massive_in=nnu_massive)

def fill_hypercube(parameter_values, standard_k_axis, massive_neutrinos=True,
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
        row[3], row[4], row[5])

    if massive_neutrinos == False:
        bundle_parameters = lambda row: build_cosmology(row[0], row[1], row[2],
            row[3], A_S_DEFAULT, 0)

    # This just provides debugging information
    redshifts_used = np.array([])

    unwritten_cells = 0
    for i in cell_range:
        #print(i, "computation initiated")
        p = None
        #try:
            #print("beginning p-spectrum computation")
        cosmology = bundle_parameters(parameter_values[i])
        
        # kp returns (in this order): p-spectrum, actual_sigma12, z_best
        
        p, actual_sigma12, z_best = evaluate_cell(cosmology, standard_k_axis)
        redshifts_used = np.append(redshifts_used, z_best)
        
        # We may actually want to remove this if-condition. For now, though, it
        # allows us to repeatedly evaluate a cosmology with the same
        # deterministic result.
        if actual_sigma12 is not None:        
            parameter_values[i][3] = actual_sigma12
        
        #print("p-spectrum computation complete!")
        #except ValueError:
        ''' Don't let unreasonable sigma12 values crash the program; ignore
        them for now. It's not clear to me why unreasonable sigma12 values
        sometimes (albeit quite rarely) raise ValueErrors. One would think
        that that situation would be adequately handled by the h=0.01 check
        in kp.
        '''
        #    traceback.print_exc(limit=1, file=sys.stdout)
        #except Exception: 
        #    traceback.print_exc(limit=1, file=sys.stdout)
        
        samples[i] = p
        
        print(i, "complete")
        unwritten_cells += 1
        if write_period is not None and unwritten_cells >= write_period:
            np.save("samples_backup_i" + str(i) + "_" + save_label + ".npy",
                samples, allow_pickle=True)
            np.save("redshifts_backup_i" + str(i) + "_" + save_label + ".npy",
                redshifts_used, allow_pickle=True)
            np.save("hc_backup_i" + str(i) + "_" + save_label + ".npy",
                parameter_values, allow_pickle=True)
            unwritten_cells = 0
    return samples, redshifts_used

def evaluate_cell(cosmology, standard_k_axis, debug=False):
    """
    Returns the power spectrum in Mpc units and the actual sigma12_tilde value
        to which it corresponds.

    I concede that the function looks like a mess right now, with debug
    statements littered all over the place.
    """
    # This allows us to roughly find the z corresponding to the sigma12 that we
    # want.

    tilde_cosmology = cp.deepcopy(cosmology)
    tilde_cosmology['omch2'] += tilde_cosmology['omnuh2']
    tilde_cosmology['omnuh2'] = 0

    tilde_cosmology = ci.specify_neutrino_mass(tilde_cosmology,
        tilde_cosmology['omnuh2'], nnu_massive_in=0)

    _redshifts=np.flip(np.linspace(0, 10, 150))
   
    if debug:
        print("\nTilde cosmology:")
        print_cosmology(tilde_cosmology)
        print("\nTrue cosmology:")
        print_cosmology(cosmology)
        print("\n")

    _, _, _, list_sigma12 = ci.evaluate_cosmology(tilde_cosmology, _redshifts,
        fancy_neutrinos=False, k_points=NPOINTS, hubble_units=False)

    # debug block

    #print(list_s12)
    if debug:
        print("Maximum s12:", max(list_sigma12))
        import matplotlib.pyplot as plt
        # Original intersection problem we're trying to solve
        plt.plot(_redshifts, list_sigma12);
        plt.axhline(cosmology["sigma12"], c="black")
        plt.title("$\sigma_{12}$ vs. $z$")
        plt.ylabel("$\sigma_{12}$")
        plt.xlabel("$z$")
        plt.show()
        # Now it's a zero-finding problem
        plt.plot(_redshifts, list_sigma12 - cosmology["sigma12"]);
        plt.axhline(0, c="black")
        plt.title("$\sigma_{12} - \sigma^{\mathrm{goal}}_{12}$ vs. $z$")
        plt.xlabel("$z$")
        plt.ylabel("$\sigma_{12} - \sigma^{\mathrm{goal}}_{12}$")
        plt.show()
    
    list_sigma12 -= cosmology["sigma12"] # now it's a zero-finding problem
    
    # remember that list_s12[0] corresponds to the highest value z
    if debug:
        print("Discrepancy between maximal achievable sigma12 and target", 
            list_sigma12[len(list_sigma12) - 1])
        print("Desired", cosmology["sigma12"])
    if list_sigma12[len(list_sigma12) - 1] < 0:
        # we need to start playing with h.
        if cosmology['h'] <= 0.1:
            print("\nThis cell is hopeless. Here are the details:\n")
            print_cosmology(cosmology)
            print("\nThe extent of failure is:",
                abs(list_sigma12[len(list_sigma12) - 1] / \
                cosmology["sigma12"]) * 100, "%\n")
            return None, None, None

        cosmology['h'] -= 0.1
        return evaluate_cell(cosmology, standard_k_axis)

    z_step = _redshifts[0] - _redshifts[1]
    interpolator = interp1d(np.flip(_redshifts), np.flip(list_sigma12),
        kind='cubic')
        
    # Newton's method requires that I already almost know the answer, so it's
    # poorly suited to our problem. This generic root finder works better.
    z_best = root_scalar(interpolator,
        bracket=(np.min(_redshifts), np.max(_redshifts))).root

    if debug:
        print("recommended redshift", z_best)

    p = np.zeros(len(standard_k_axis))

    k, _, p, actual_sigma12 = ci.evaluate_cosmology(cosmology,
        redshifts=np.array([z_best]), fancy_neutrinos=False,
        k_points=NPOINTS) 
    if cosmology['omnuh2'] != 0:
        _, _, _, actual_sigma12 = ci.evaluate_cosmology(tilde_cosmology,
            redshifts=np.array([z_best]), fancy_neutrinos=False,
            k_points=NPOINTS)
    # De-nest
    actual_sigma12 = actual_sigma12[0]

    if cosmology['h'] != model0['h']: # we've touched h, we need to interpolate
        print("We had to move h to", np.around(cosmology['h'], 3))
        
        interpolator = interp1d(k, p, kind="cubic")
        p = interpolator(standard_k_axis)

    if len(p) == 1: # then de-nest
        p = p[0] 

    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    return p, actual_sigma12, z_best
