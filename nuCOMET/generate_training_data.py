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
import pandas as pd
import re

import camb_interface as ci

cosm = pd.read_csv("cosmologies.dat", sep='\s+')
model0 = cosm.loc[0]

''' AndreaP thinks that npoints=300 should be a good balance of accuracy and
computability for our LH.'''
NPOINTS = 300

import sys, traceback
import copy as cp
import camb_interface

def build_cosmology(om_b_in, om_c_in, ns_in, om_nu_in, sigma12_in, As_in):
    # Use Aletheia model 0 as a base
    cosmology = cp.deepcopy(camb_interface.cosm.iloc[0])
    
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

    return camb_interface.specify_neutrino_mass(cosmology, nnu_massive,
        nnu_massive_in=nnu_massive)

def fill_hypercube(parameter_values, standard_k_axis, cell_range=None,
    samples=None, write_period=None):
    """
    @parameter_values: this should be a list of tuples to
        evaluate kp at.

    @cell_range adjust this value in order to pick up from where you
        left off, and to run this method in saveable chunks.
    """
    if cell_range is None:
        cell_range = range(len(parameter_values))
    if samples is None:
        samples = np.zeros((len(parameter_values), NPOINTS))

    unwritten_cells = 0
    for i in cell_range:
        #print(i, "computation initiated")
        config = parameter_values[i]
        #print(config, "\n", config[4])
        p = None
        #try:
            #print("beginning p-spectrum computation")
        cosmology = build_cosmology(config[0], config[1], config[2],
            config[4], config[3], config[5])
        p = kp(cosmology, standard_k_axis)
            #print("p-spectrum computation complete!")
        #except ValueError:
            # Don't let unreasonable sigma12 values crash the program; ignore
            # them for now.
        #    traceback.print_exc(limit=1, file=sys.stdout)
        samples[i] = p
        
        print(i, "complete")
        unwritten_cells += 1
        if write_period is not None and unwritten_cells >= write_period:
            np.save("samples_backup_i" + str(i) + ".npy", samples,
                allow_pickle=True)
            unwritten_cells = 0
    return samples

def kp(cosmology, standard_k_axis,
    _redshifts=np.flip(np.linspace(0, 1100, 150)), solvability_known=False):
    """
    Returns the scale axis and power spectrum in Mpc units

    @h_in=0.67 starts out with the model 0 default for Aletheia, and we
        will decrease it if we cannot get the desired sigma12 with a
        nonnegative redshift.
    """
    _, _, _, list_sigma12 = ci.kzps(cosmology, _redshifts,
        fancy_neutrinos=False, k_points=300)
    # debug block
    
    import matplotlib.pyplot as plt
    #print(list_s12)
    if False:
        plt.plot(_redshifts, list_sigma12);
        plt.axhline(cosmology["sigma12"], c="black")
        plt.title("$\sigma_{12}$ vs. $z$")
        plt.ylabel("$\sigma_{12}$")
        plt.xlabel("$z$")
        plt.show()
     
    # debug block
    if False:
        plt.plot(_redshifts, list_sigma12 - cosmology["sigma12"]);
        plt.axhline(0, c="black")
        plt.title("$\sigma_{12} - \sigma^{\mathrm{goal}}_{12}$ vs. $z$")
        plt.xlabel("$z$")
        plt.ylabel("$\sigma_{12} - \sigma^{\mathrm{goal}}_{12}$")
        plt.show()
    
    list_sigma12 -= cosmology["sigma12"] # now it's a zero-finding problem
    
    # remember that list_s12[0] corresponds to the highest value z
    if list_sigma12[len(list_sigma12) - 1] < 0 and cosmology['h'] > 0.01:
        ''' we need to start playing with h.
        To save on computation, let's check if even the minimum allowed value
        rescues the problem.
        '''
        if not solvability_known:
            try:
                limiting_case = cp.deepcopy(cosmology)
                limiting_case['h'] = 0.01
                kp(cosmology, standard_k_axis, _redshifts=_redshifts)
            except ValueError:
                print("This cell is hopeless. Moving on...")
                return None

        ''' Now we know that modifying h will eventually fix the situation,
        so we start decreasing h. We also set a flag to make sure we never
        repeat this check.'''
        cosmology['h'] -= 0.01
        return kp(cosmology, standard_k_axis, _redshifts=_redshifts,
            solvability_known=True)

    z_step = _redshifts[0] - _redshifts[1]
    interpolator = interp1d(np.flip(_redshifts), np.flip(list_sigma12),
        kind='cubic')
        
    # Newton's method requires that I already almost know the answer, so it's
    # poorly suited to our problem. This generic root finder works better.
    z_best = root_scalar(interpolator,
        bracket=(np.min(_redshifts), np.max(_redshifts))).root

    if z_step > 0.05: # The z-resolution is too low, we need to recurse
        # '0.05' here is pretty computationally expensive;
        # if the program doesn't run fast enough let's kick it up.
        new_floor = max(z_best - z_step, 0)
        # I don't know if the last scattering really should be our cap, but it
        # seems like a reasonable cap to me.
        new_ceiling = min(1100, z_best + z_step)
        '''What is the point of the 150 limit? Why can't CAMB simply give me a
            bigger interpolator?? '''
        return kp(cosmology, standard_k_axis,
            _redshifts=np.flip(np.linspace(new_floor, new_ceiling, 150)),
            solvability_known=True)
    else: # Our current resolution is satisfactory, let's return a result
        p = np.zeros(len(standard_k_axis))

        if cosmology['h'] == model0['h']: # if we haven't touched h,
            # we don't need to interpolate.
            _, _, p, _ = ci.kzps(cosmology, zs=np.array([z_best]),
                fancy_neutrinos=False, k_points=300)
           
        else: # it's time to interpolate
            if cosmology['h'] > 0.01: # this check ensures that the
                # notification appears only once.
                print("We had to move the value of h.")
            
            # Andrea and Ariel agree that this should use k_hunit=False
            PK = ci.kzps_interpolator(cosmology, zs=_redshifts,
                fancy_neutrinos=False, z_points=150)

            p = PK.P(z_best, standard_k_axis)

        if len(p) == 1:
            p = p[0] 

        # We don't need to return k because we take for granted that all
        # runs will have the same k axis.

        #print(p)
        #print(p is None)
        #plt.plot(p); plt.show()
        
        return p
