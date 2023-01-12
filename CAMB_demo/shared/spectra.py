import numpy as np
import pandas as pd
import camb
import re

'''Keep in mind that this is NOT the same file as the original
"cosmology_Aletheia.dat" that Ariel gave us! If you use the unaltered version,
you will get a segfault'''
path_to_me = "/home/lfinkbei/Documents/Master/CAMB_demo/shared/"
cosm = pd.read_csv(path_to_me + "data/cosmologies.dat", sep='\s+')

omega_nu = np.array([0.0006356, 0.002, 0.006356])
# Add corresponding file accessors, to check our work later
omnu_strings = np.array(["0.0006", "0.002", "0.006"])

# The following code is somewhat hard;
# I'm not sure how better to do it.
redshift_column = re.compile("z.+")

''' To check our work, we'll need the correct solutions. '''
file_base = path_to_me + "data/power_nu/Aletheia_powernu_zorig_nu"

powernu = {}
for omnu in omnu_strings:
    powernu[omnu] = []

    for i in range(0, 9): # iterate over models
        powernu[omnu].append([])
        for j in range(0, 5): # iterate over snapshots
            powernu[omnu][i].append(pd.read_csv(file_base + omnu + "_caso" + \
                str(i) + "_000" + str(j) + ".dat",
                names=["k", "P_no", "P_nu", "ratio"], sep='\s+'))

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
        neutrino_hierarchy, possibly more. But CAMB does not accept omnuh2 as
        an input, so I have to reverse-engineer it somehow.
        
        Also, should we replace default_nnu with something else in the
        following expression? Even if we're changing N_massive to 1,
        N_total_eff = 3.046 nonetheless, right?'''
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
    
    pars.InitPower.set_params(As=mlc["A_s"], ns=mlc["n_s"])
    
    pars.set_dark_energy(w=mlc["w0"], wa=float(mlc["wa"]),
        dark_energy_model='ppf')
    '''
    To change the the extent of the k-axis,
    change the following line as well as the "get_matter_power_spectrum" call
    
    In some cursory tests, the accurate_massive_neutrino_transfers
    flag did not appear to significantly alter the outcome.
    '''
    pars.set_matter_power(redshifts=zs, kmax=10.0, nonlinear=False)
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    
    # The flags var1=8 and var2=8 indicate that we are looking at the
    # power spectrum of CDM + baryons (i.e. neutrinos excluded).
    k, z, p = results.get_matter_power_spectrum(
        minkh=3e-3, maxkh=3.0, npoints = 1000000,
        var1=8, var2=8
    )
    sigma12 = results.get_sigmaR(12, hubble_units=False)
    
    return k, z, p, sigma12 

def parse_redshifts(model_num):
    """
    Return the list of amplitude-equalized redshifts
    given for a particular model in the Aletheia dat file.
    
    This function is intended to return the redshifts
    in order from high (old) to low (recent),
    which is the order that CAMB will impose
    if not already used.
    """
    z = []
    model = cosm.loc[model_num]
    
    for column in cosm.columns:
        if redshift_column.match(column):
            z.append(model[column])
            
    # No need to sort these because they should already
    # be in descending order.
    return np.array(z)

def truncator(big_x, big_y, small_x, small_y, interpolation=True):
    """
    Truncate big arrays based on small_arrays,
    then interpolate the small arrays over
    the truncated big_x domain.
    @returns:
        trunc_x: truncated big_x array
        trunc_y: truncated big_y array
        aligned_y: interpolation of small_y over trunc_x
    """
    # What is the most conservative lower bound?
    lcd_min = max(min(small_x), min(big_x))
    # What is the most conservative upper bound?
    lcd_max = min(max(small_x), max(big_x))
    
    # Eliminate points outside the conservative bounds
    mask = np.all([[big_x <= lcd_max], [big_x >= lcd_min]], axis=0)[0]
    trunc_x = big_x[mask]
    trunc_y = big_y[mask]
    
    aligned_y = small_y[mask]
    # Is the spacing different in big_x and small_x?
    if interpolation:
        interpolator = interp1d(small_x, small_y, kind="cubic")
        aligned_y = interpolator(trunc_x)
    
    return trunc_x, trunc_y, aligned_y
