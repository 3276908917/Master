import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import camb
from camb import model, initialpower
import pandas as pd
import re

cosm = pd.read_csv("cosmologies.dat", sep='\s+')

zs = np.array([0])

from copy import deepcopy as dc
# z_in = parse_redshifts(0)
z_in = np.array([0]) # I don't see a redshift in the paper;
# are they always assuming z=0?

model0 = cosm.loc[0]

modelK = dc(model0)
# Now it's safe to modify this object into the Kiakotou stuff.
modelK["OmM"] = 0.25
modelK["OmB"] = 0.04
modelK["h"] = 0.7
omnuh2_in = modelK["OmM"] * 0.04

# Derived quantities
modelK["ombh2"] = modelK["OmB"] * modelK["h"] ** 2
OmCDM = modelK["OmM"] - modelK["OmB"]
modelK["omch2"] = OmCDM * modelK["h"] ** 2

pars = camb.CAMBparams()
omch2_in = modelK["omch2"]
mnu_in = 0

# nnu_in = 3.046
nnu_massive = 3
mnu_in = omnuh2_in * camb.constants.neutrino_mass_fac / \
    (camb.constants.default_nnu / 3.0) ** 0.75

omch2_in -= omnuh2_in

pars.set_cosmology(
    H0=modelK["h"] * 100,
    ombh2=modelK["ombh2"],
    omch2=omch2_in,
    omk=modelK["OmK"],
    mnu=mnu_in,
    num_massive_neutrinos=nnu_massive#,
    #neutrino_hierarchy="normal"
)

pars.num_nu_massless = 3.046 - nnu_massive
pars.nu_mass_eigenstates = nnu_massive

pars.nu_mass_numbers[:(pars.nu_mass_eigenstates + 1)] = \
    list(np.ones(len(pars.nu_mass_numbers[:(pars.nu_mass_eigenstates + 1)]), int) * 0)
pars.num_nu_massive = 0
if nnu_massive != 0:
    pars.num_nu_massive = sum(pars.nu_mass_numbers[:(pars.nu_mass_eigenstates + 1)])
pars.nu_mass_numbers
print(pars.num_nu_massive)

pars.InitPower.set_params(As=modelK["A_s"], ns=modelK["n_s"])

#pars.DarkEnergy = camb.dark_energy.DarkEnergyPPF(w=modelK["w0"], wa=float(modelK["wa"]))
pars.set_dark_energy(w=modelK["w0"], wa=float(modelK["wa"]), dark_energy_model='ppf')

# The following bogus line can expose whether the simulation is paying attention to DE
# (for some reason, the pair (-100, -100) did not look strange at all!)
# pars.set_dark_energy(w=-0, wa=0, dark_energy_model='ppf')

# To change the the extent of the k-axis,
# change the following line as well as the "get_matter_power_spectrum" call
pars.set_matter_power(redshifts=zs, kmax=10.0,
    nonlinear=False)#, accurate_massive_neutrino_transfers=True)
#pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
results.calc_power_spectra(pars)

k, z, p = results.get_matter_power_spectrum(
    minkh=3e-3, maxkh=3.0, npoints = 10000,
    var1=8, var2=8
)
sigma12 = results.get_sigmaR(12, hubble_units=False)
