import numpy as np
import pandas as pd
import camb
import re

'''Keep in mind that this is NOT the same file as the original
"cosmology_Aletheia.dat" that Ariel gave us! If you use the unaltered version,
you will get a segfault'''
path_to_me = "/home/lfinkbei/Documents/Master/CAMB_demo/shared/"
cosm = pd.read_csv(path_to_me + "data/cosmologies.dat", sep='\s+')

# The following code is somewhat hard;
# I'm not sure how better to do it.
redshift_column = re.compile("z.+")

def kzps(mlc, omnuh2_in, massive_neutrinos=False, zs = [0], nnu_massive_in=3):
    """
    Returns the scale axis, redshifts, power spectrum, and sigma12
    of a Lambda-CDM model
    @param mlc : "MassLess Cosmology"
        a dictionary of values
        for CAMBparams fields
    @param omnuh2_in : neutrino physical mass density
    @param sigma12 : if the spectrum should be rescaled,
        this parameter determines the desired sigma12 of the model
    @massive_neutrinos : if this is True,
        the value in omnuh2_in is used to set omnuh2.
        If this is False,
        the value in omnuh2_in is simply added to omch2.
    """ 
    pars = camb.CAMBparams()
    omch2_in = mlc["omch2"]
    mnu_in = 0
    
    nnu_massive = 0
    if massive_neutrinos:
        '''This is a horrible workaround, and I would like to get rid of it
        ASAP The following line destroys dependence on TCMB and
        neutrino_hierarchy, possibly more.'''        
        mnu_in = omnuh2_in * camb.constants.neutrino_mass_fac / \
            (camb.constants.default_nnu / 3.0) ** 0.75 
        omch2_in -= omnuh2_in
        nnu_massive = nnu_massive_in
    pars.set_cosmology(
        H0=mlc["h"] * 100,
        ombh2=mlc["ombh2"],
        omch2=omch2_in,
        omk=mlc["OmK"],
        mnu=mnu_in,
        num_massive_neutrinos=nnu_massive
    )
    
    pars.num_nu_massless = 3.046 - nnu_massive
    pars.nu_mass_eigenstates = nnu_massive
    stop_i = pars.nu_mass_eigenstates + 1
    pars.nu_mass_numbers[:stop_i] = \
        list(np.ones(len(pars.nu_mass_numbers[:stop_i]), int))
    pars.num_nu_massive = 0
    if nnu_massive != 0:
        pars.num_nu_massive = sum(pars.nu_mass_numbers[:stop_i])
    
    pars.InitPower.set_params(As=mlc["A_s"], ns=mlc["n_s"])
    
    pars.set_dark_energy(w=mlc["w0"], wa=float(mlc["wa"]),
        dark_energy_model='ppf')
    
    # To change the the extent of the k-axis,
    # change the following line as well as the "get_matter_power_spectrum" call
    pars.set_matter_power(redshifts=zs, kmax=10.0, nonlinear=False)
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)

    k, z, p = results.get_matter_power_spectrum(
        minkh=3e-3, maxkh=3.0, npoints = 1000000,
        var1=8, var2=8
    )
    sigma12 = results.get_sigmaR(12, hubble_units=False)
    
    return k, z, p, sigma12 
