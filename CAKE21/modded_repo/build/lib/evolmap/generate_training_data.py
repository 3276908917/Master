 # It seems like tkgrid.py might have been intended to help me with this,
 # but I don't know how to use it

import sys, platform, os
import numpy as np
from scipy.interpolate import interp1d

#import camb
#from camb import model, initialpower
import pandas as pd
import re

cosm = pd.read_csv("cosmologies.dat", sep='\s+')
model0 = cosm.loc[0]

NPOINTS = 10000

def fill_hypercube(parameter_values):
    """
    @parameter_values: this should be a list of tuples to
        evaluate kp at.
    """
    samples = np.zeros((len(parameter_values), 2, NPOINTS))
    for i in range(len(parameter_values)):
        config = parameter_values[i]
        k, p = kp(config[0], config[1], config[2])
        samples[i, 0] = k
        samples[i, 1] = p
        print(i)

def kp(om_b, om_c, h):
    """
    This is a pared-down demo version of kzps, it only considers
    redshift zero.

    Returns the scale axis and power spectrum in Mpc units
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0 = h * 100,
        ombh2=om_b,
        omch2=om_c,
        omk=model0["OmK"],
        mnu=0,
        num_massive_neutrinos=0,
        neutrino_hierarchy="degenerate" # 1 eigenstate approximation;
        # our neutrino setup (see below) is not valid for inverted/normal hierarchies.
    )

    pars.InitPower.set_params(As=model0["A_s"], ns=model0["n_s"])
    pars.set_dark_energy(w=model0["w0"], wa=float(model0["wa"]), dark_energy_model='ppf')

    pars.set_matter_power(redshifts=np.array([0]), kmax=10.0, nonlinear=False)
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)

    k, z, p = results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=10.0, npoints = 10000,
        var1=8, var2=8
    )

    return k * h, p * h ** 3

"""
Here is some demo code that I used to get to start this off:
ombh2 = 0.022445
omch2 = 0.120567
hc = evolmap.lhc.generate_samples({'om_b': [0.9 * ombh2, 1.1 * omch2], 'om_c': [0.9 * omch2, 1.1 * omch2], 'h': [.603, .737]}, 100, 100)
"""
