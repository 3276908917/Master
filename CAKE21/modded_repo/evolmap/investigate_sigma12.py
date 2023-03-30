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

def kp(om_b_in, om_c_in, ns_in, om_nu_in):
    """
    This is a pared-down demo version of kzps, it only considers
    redshift zero.

    Returns the scale axis and power spectrum in Mpc units
    """
    # model0 stuff is assumed to get the initial pspectrum that we'll rescale
    OmK_in = 0
    '''I didn't want to assume this last line yet; part of the experiment is to
    see if OmK is correctly automatically set to 0. Unfortunately,
    set_cosmology demands an omk value, I guess we can ask Ariel how to run
    this test then.'''
    
    w0_in = -1.00
    wa_in = 0.00
    h_in = 0.01
    As_in = 1e-9
   
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
  
    pars.set_matter_power(redshifts=[0], kmax=10.0 / h_in,
        nonlinear=False)

    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    
    s12 = results.get_sigmaR(12, var1=8, var2=8, hubble_units=False)
    return s12

