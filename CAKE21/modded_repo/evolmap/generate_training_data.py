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

cosm = pd.read_csv("cosmologies.dat", sep='\s+')
model0 = cosm.loc[0]

''' AndreaP thinks that npoints=300 should be a good balance of accuracy and
computability for our LH.'''
NPOINTS = 300

import sys, traceback

def fill_hypercube(parameter_values):
    """
    @parameter_values: this should be a list of tuples to
        evaluate kp at.
    """
    samples = np.zeros((len(parameter_values), 2, NPOINTS))
    for i in range(len(parameter_values)):
        config = parameter_values[i]
        #print(config, "\n", config[4])
        k, p = None, None
        try:
            k, p = kp(config[0], config[1], config[2], config[4], config[3])
        except ValueError:
            # Don't let unreasonable sigma12 values crash the program; ignore
            # them for now.
            traceback.print_exc(limit=1, file=sys.stdout)
        
        samples[i, 0] = k
        samples[i, 1] = p
        
        print(i)
    return samples

def kp(om_b_in, om_c_in, ns_in, om_nu_in, sigma12_in,
    _redshifts=np.flip(np.linspace(0, 1100, 150))):
    """
    This is a pared-down demo version of kzps, it only considers
    redshift zero.

    Returns the scale axis and power spectrum in Mpc units
    """
    # model0 stuff is assumed to get the initial pspectrum that we'll rescale
    h_in = 0.67
    As_in = 2.12723788013000E-09
    OmK_in = 0
    '''I didn't want to assume this last line yet; part of the experiment is to
    see if OmK is correctly automatically set to 0. Unfortunately, set_cosmology
    demands an omk value, I guess we can ask Ariel how to run this test then.'''
    
    w0_in = -1.00
    wa_in = 0.00 
   
    # This sucks. See spectra.py kzps for a more detailed complaint.
    mnu_in = om_nu_in * camb.constants.neutrino_mass_fac / \
        (camb.constants.default_nnu / 3.0) ** 0.75
    nnu_massive=1 if mnu_in != 0 else 0
   
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0 = h_in * 100,
        ombh2=om_b_in,
        omch2=om_c_in,
        omk=OmK_in,
        tau=0.0952, # desperation argument
        mnu=mnu_in,
        num_massive_neutrinos=nnu_massive,
        neutrino_hierarchy="degenerate" # 1 eigenstate approximation; our
        # neutrino setup (see below) is not valid for inverted/normal
        # hierarchies.
    )
    
    pars.num_nu_massless = 3.046 - nnu_massive
    pars.nu_mass_eigenstates = nnu_massive
    stop_i = pars.nu_mass_eigenstates + 1
    pars.nu_mass_numbers[:stop_i] = \
        list(np.ones(len(pars.nu_mass_numbers[:stop_i]), int))
    pars.num_nu_massive = 0
    if nnu_massive != 0:
        pars.num_nu_massive = sum(pars.nu_mass_numbers[:stop_i])

    # Last three are desperation arguments
    pars.InitPower.set_params(As=As_in, ns=ns_in, r=0, nt=0.0, ntrun=0.0)
    
    ''' The following seven lines are desperation settings
    If we ever have extra time, we can more closely study what each line does
    '''
    # This is a desperation line in light of the previous line. The previous
    # line seems to have served me well enough so far, but BSTS.
    pars.NonLinear = camb.model.NonLinear_none
    pars.WantCls = False
    pars.WantScalars = False
    pars.Want_CMB = False
    pars.DoLensing = False
    pars.YHe = 0.24   
    pars.set_accuracy(AccuracyBoost=2)
  
    pars.set_matter_power(redshifts=_redshifts, kmax=10.0 / h_in,
        nonlinear=False)

    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    
    list_s12 = results.get_sigmaR(12, var1=8, var2=8, hubble_units=False)

    '''Also: you can't root out the negative z later, you have to do it as
    early as here... '''

    # debug block
    '''
    import matplotlib.pyplot as plt
    #print(list_s12)
    plt.plot(_redshifts, list_s12);
    plt.axhline(sigma12_in)
    plt.show()
    '''

    list_s12 -= sigma12_in # now it's a zero-finding problem
    
    # debug block
    '''
    plt.plot(_redshifts, list_s12);
    plt.axhline(0)
    plt.show()
    '''
    
    z_step = _redshifts[0] - _redshifts[1]
    interpolator = interp1d(np.flip(_redshifts), np.flip(list_s12),
        kind='cubic')
        
    # Newton's method requires that I already almost know the answer, so it's
    # poorly suited to our problem. This generic root finder works better.
    z_best = root_scalar(interpolator,
        bracket=(np.min(_redshifts), np.max(_redshifts))).root
    
    if z_step > 0.05: # this is pretty computationally expensive;
        # if the program doesn't run fast enough let's kick it up to 1
        new_floor = max(z_best - z_step, 0)
        # I don't know if the last scattering really should be our cap, but it
        # seems like a reasonable cap to me.
        new_ceiling = min(1100, z_best + z_step)
        '''What is the point of the 150 limit? Why can't CAMB simply give me a
            bigger interpolator?? '''
        return kp(om_b_in, om_c_in, ns_in, om_nu_in, sigma12_in,
            _redshifts=np.flip(np.linspace(new_floor, new_ceiling, 150)))
    else:

        pars.set_matter_power(redshifts=np.array([z_best]), kmax=10.0 / h_in,
            nonlinear=False)

        results = camb.get_results(pars)
        results.calc_power_spectra(pars)

        k, z, p = results.get_matter_power_spectrum(
            minkh=1e-4 / h_in, maxkh=10.0 / h_in, npoints = NPOINTS,
            var1=8, var2=8
        )
        
        if len(p) == 1:
            p = p[0] 

        return k * h_in, p * h_in ** 3
