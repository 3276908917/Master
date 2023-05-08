path_base_linux = "/home/lfinkbei/Documents/"
path_base_rex = "C:/Users/Lukas/Documents/GitHub/"
path_base_otto = "T:/GitHub/"
path_base = path_base_linux

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import camb
import re
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import copy as cp

'''Keep in mind that this is NOT the same file as the original
"cosmology_Aletheia.dat" that Ariel gave us! If you use the unaltered version,
you will get a segfault'''
path_to_me = path_base + "Master/CAMB_notebooks/shared/"
cosm = pd.read_csv(path_to_me + "data/cosmologies.dat", sep='\s+')

omegas_nu = np.array([0.0006356, 0.002148659574468, 0.006356, 0.01])
# Add corresponding file accessors, to check our work later
omnu_strings = np.array(["0.0006", "0.002", "0.006", "0.01"])

# The following code is somewhat hard;
# I'm not sure how better to do it.
redshift_column = re.compile("z.+")

'''! We really ought to merge the next three functions'''

def define_powernu(relative_path, omeganu_strings=None):
    file_base = file_base = path_to_me + relative_path

    def iterate_over_models_and_redshifts(accessor="0.002"):
        nested_spectra = []
        for i in range(0, 9): # iterate over models
            nested_spectra.append([])
            for j in range(0, 5): # iterate over snapshots
                nested_spectra[i].append(pd.read_csv(file_base + \
                    accessor + "_caso" + str(i) + "_000" + str(j) + ".dat",
                    names=["k", "P_no", "P_nu", "ratio"], sep='\s+'))

        return nested_spectra

    if omeganu_strings is None:
        return iterate_over_models_and_redshifts()

    powernu = {}

    for i in range(len(omeganu_strings)):
        file_accessor = omeganu_strings[i]
        powernu_key = omnu_strings[i]

        powernu[powernu_key] = \
            iterate_over_models_and_redshifts(file_accessor)

    return powernu

### Just some standard colors and styles for when I plot several models
# together.
colors = ["green", "blue", "brown", "red", "black", "orange", "purple",
          "magenta", "cyan"] * 200

#styles = ["solid", "dotted", "dashed", "dashdot", "solid", "dotted", "dashed",
#    "dashdot"]
# Line styles are unfortunately too distracting in plots as dense as those with
# which we are here dealing; make everything solid
styles = ["solid"] * 200

def is_matchable(target, cosmology):
    # I thought bigger sigma12 values were supposed to come with lower z,
    # but the recent matching results have got me confused.
    _, _, _, s12_big = kzps(cosmology, 0, nu_massive=False, zs=[0])

def match_sigma12(target, tolerance, cosmology,
    _redshifts=np.flip(np.linspace(0, 1100, 150)), _min=0):
    """
        Return a redshift at which to evaluate the power spectrum of cosmology
    @cosmology such that the sigma12_massless value of the power spectrum is
    within @tolerance (multiplicative discrepancy) of @target.
 
    @target this is the value of sigma12_massless at the assumed redshift
        (e.g. typically at z=2.0 for a standard Aletheia model-0 setup).
    @cosmology this is the cosmology for which we want to find a sigma12 value,
        so this will typically be an exotic or randomly generated cosmology
    @tolerance ABS((target - sigma12_found) / target) <= tolerance is the
        stopping condition for the binary search that this routine uses.
    @_z: the redshift to test at. This is part of the internal logic of the
        function and should not be referenced elsewhere.
    """
    # If speed is an issue, let's reduce the k samples to 300, or at least
    # add num_samples as a function parameter to kzps

    # First, let's probe the half-way point.
    # We're assuming a maximum allowed redshift of $z=2$ for now.

    #print(_z)
    _, _, _, list_s12 = kzps(cosmology, 0, nu_massive=False, zs=_redshifts)

    import matplotlib.pyplot as plt
    #print(list_s12)
    if False:
        plt.plot(_redshifts, list_s12);
        plt.axhline(sigma12_in)
        plt.show()
     
    # debug block
    if False:
        plt.plot(_redshifts, list_s12 - sigma12_in);
        plt.axhline(0)
        plt.show()
    

    list_s12 -= target # now it's a zero-finding problem

    # For some reason, flipping both arrays helps the interpolator
    # But I should come back and check this, I'm not sure if this was just a
    # patch for the Newton method
    interpolator = interp1d(np.flip(_redshifts), np.flip(list_s12),
        kind='cubic')
    try:
        z_best = root_scalar(interpolator, bracket=(np.min(_redshifts),
            np.max(_redshifts))).root
    except ValueError:
        print("No solution.")
        return None # there is no solution

    _, _, _, s12_out = kzps(cosmology, 0, nu_massive=False, zs=[z_best])
    discrepancy = (s12_out[0] - target) / target 
    if abs(discrepancy) <= tolerance:
        return z_best
    else:
        z_step = _redshifts[0] - _redshifts[1]
        new_floor = max(0, z_best - z_step)
        new_ceil = min(1100, z_best + z_step)
        return match_s12(target, tolerance, cosmology, _redshifts = \
            np.flip(np.linspace(new_floor, new_ceil, 150)))

def get_As_matched_cosmology(A_s=2.12723788013000E-09):
    """
    Unfortunately, all of these bounds are hard-coded. Maybe we can read in a
    table for this?

    The default A_s value corresponds to model 0

    Warning! You may have to throw out some of the cosmologies that you get
    from this routine because I am nowhere guaranteeing that the sigma12 you
    want actually corresponds to a positive redshift... of course, this
    wouldn't be a problem if CAMB allowed negative redshifts.
    """
    row = {}

    # Shape pars: CONSTANT ACROSS MODELS
    row['ombh2'] = 0.022445
    row['omch2'] = 0.120567
    row['n_s'] = 0.96

    row['h'] = np.random.uniform(0.2, 1)
   
    # Given h, the following are now fixed:
    row['OmB'] = row['ombh2'] / row['h'] ** 2
    row['OmC'] = row['omch2'] / row['h'] ** 2
    row['OmM'] = row['OmB'] + row['OmC'] 

    row['OmK'] = 0
    row['OmL'] = 1 - row['OmM'] - row['OmK']
    
    #~ Do we have any constraints on h besides Aletheia?
    # I ask because this seems like a pretty small window.
    #ditto
    row['w0'] = np.random.uniform(-2., -.5)
    # ditto
    row['wa'] = np.random.uniform(-0.5, 0.5)

    row['A_s'] = A_s

    return row

def get_random_cosmology():
    """
    Unfortunately, all of these bounds are hard-coded. Maybe we can read in a
    table for this?
    """
    row = {}

    # Shape pars: CONSTANT ACROSS MODELS
    row['ombh2'] = 0.022445
    row['omch2'] = 0.120567
    row['n_s'] = 0.96

    row['h'] = np.random.uniform(0.2, 1)
   
    # Given h, the following are now fixed:
    row['OmB'] = row['ombh2'] / row['h'] ** 2
    row['OmC'] = row['omch2'] / row['h'] ** 2
    row['OmM'] = row['OmB'] + row['OmC'] 

    row['OmK'] = np.random.uniform(-0.05, 0)
    row['OmL'] = 1 - row['OmM'] - row['OmK']
    
    #~ Do we have any constraints on h besides Aletheia?
    # I ask because this seems like a pretty small window.
    #ditto
    row['w0'] = np.random.uniform(-2., -.5)
    # ditto
    row['wa'] = np.random.uniform(-0.5, 0.5)

    A_min = np.exp(1.61) / 10 ** 10
    A_max = np.exp(5) / 10 ** 10
    row['A_s'] = np.random.uniform(A_min, A_max)

    #~ Should we compute omnuh2 here, or leave that separate?

    #~ Should we also specify pars not specified by the Aletheia data
        # table, for example tau or the CMB temperature?

    return row
        
def boltzmann_battery(onh2s, onh2_strs, skips_omega = [0, 2],
    skips_model=[8], skips_snapshot=[1, 2, 3], hubble_units=False,
    models=cosm, fancy_neutrinos=False, k_points=100000):
    """
    Return format uses an architecture that closely agrees with that of Ariel's
    in the powernu results:
    spec_sims
        omnuh2 str
            model index
                snapshot index
                    quantity of interest

    Although this agreement is an added benefit, the main point is simply to
    have a cleaner and more versatile architecture than the mess of separate
    arrays returned previously. So even if the "ground truth" object should
    eventually cease to agree in shape, this function already returns a much
    more pleasant object.

    Another difference with boltzmann_battery: this function automatically
    uses h_units=False, which should further bring my object into agreement
    with powernu. This is more debatable than simple architecture cleanup, so I
    will leave this as a flag up to the user.
    """
    assert type(onh2s) == list or type(onh2s) == np.ndarray, \
        "if you want only one omega value, you must still nest it in a list"
    assert type(onh2_strs) == list or type(onh2_strs) == np.ndarray, \
        "if you want only one omega value, you must still nest it in a list"
    assert len(onh2s) == len(onh2_strs), "more or fewer labels than points"
    
    spec_sims = {}

    for this_omnu_index in range(len(onh2s)):
        print(this_omnu_index % 10, end='')
        this_omnu = onh2s[this_omnu_index]
        this_omnu_str = onh2_strs[this_omnu_index]
        if this_omnu_index in skips_omega:
            spec_sims[this_omnu_str] = None
            continue
        spec_sims[this_omnu_str] = []
        for mindex, row in models.iterrows():
            if mindex in skips_model:
                # For example, I don't yet understand how to implement model 8
                spec_sims[this_omnu_str].append(None)
                continue
                
            h = row["h"]
            spec_sims[this_omnu_str].append([])
       
            z_input = parse_redshifts(mindex)
            if None in z_input:
                spec_sims[this_omnu_str][m_index] = None
                continue

            #print("z_input", z_input)
            #print("total Zs", len(z_input)) 
            for snap_index in range(len(z_input)):
                '''
                since z_input is ordered from z large to z small,
                and snap indices run from z large to z small,
                z_index = snap_index in this case and NOT in general
                '''
                #print(z_index)
                if snap_index in skips_snapshot:
                    #print("skipping", z_index)
                    spec_sims[this_omnu_str][mindex].append(None)
                    continue
                #print("using", z_index)
                inner_dict = {}
                z = z_input[snap_index]
             
                massless_nu_cosmology = specify_neutrino_mass(
                    row, 0, nnu_massive_in=0)
                massless_tuple = kzps(massless_nu_cosmology, zs=[z],
                    fancy_neutrinos=fancy_neutrinos, k_points=k_points,
                    hubble_units=hubble_units)
                inner_dict["k"] = massless_tuple[0]
                inner_dict["P_no"] = massless_tuple[2]
                inner_dict["s12_massless"] = massless_tuple[3]

                massive_nu_cosmology = specify_neutrino_mass(
                    row, this_omnu, nnu_massive_in=1)
                # Adjust CDM density so that we have the same total matter
                # density as before:
                massive_nu_cosmology["omch2"] -= this_omnu

                massive_tuple = kzps(massive_nu_cosmology, zs=[z],
                    fancy_neutrinos=fancy_neutrinos, k_points=k_points,
                    hubble_units=hubble_units)
                inner_dict["P_nu"] = massless_tuple[2]
                inner_dict["s12_massive"] = massive_tuple[3]
                
                # Temporary addition, for debugging
                inner_dict["z"] = z_input[snap_index]               
 
                assert np.array_equal(massless_tuple[0], massive_tuple[0]), \
                   "assumption of identical k axes not satisfied!"
                    
                spec_sims[this_omnu_str][mindex].append(inner_dict) 

    return spec_sims

def make_neutrinos_fancy(pars, nnu_massive):
    """
    This is a kzps helper function which enables the infamous fancy neutrino.
    In practice, this only overrides the effective number of massless neutrinos
    by a small amount, which nonetheless leads to irreconcilable discrepancies
    compared to observations.
    """
    pars.num_nu_massless = 3.046 - nnu_massive
    pars.nu_mass_eigenstates = nnu_massive
    stop_i = pars.nu_mass_eigenstates + 1
    pars.nu_mass_numbers[:stop_i] = \
        list(np.ones(len(pars.nu_mass_numbers[:stop_i]), int))
    pars.num_nu_massive = 0
    if nnu_massive != 0:
        pars.num_nu_massive = sum(pars.nu_mass_numbers[:stop_i])

def apply_universal_output_settings(pars):
    """
    This is a kzps helper function which modifies the desired accuracy of CAMB
    and which disables certain outputs in which we are not interested.
    """

    ''' The following lines are desperation settings
    If we ever have extra time, we can more closely study what each line does
    '''
    pars.WantCls = False
    pars.WantScalars = False
    pars.Want_CMB = False
    pars.DoLensing = False
    pars.YHe = 0.24
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False

def input_dark_energy(pars, w0, wa):
    """
    Helper function for input_cosmology.
    Handles dark energy by using a PPF model (which allows a wide range of
        w0 and wa values) unless w0 is -1 and wa is zero, in which case we use
        the two-fluid approximation.
    """
    # Default is fluid, so we don't need an 'else'
    if w0 != -1 or wa != 0:
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')

def specify_neutrino_mass(mlc, omnuh2_in, nnu_massive_in=1):
    """
    Helper function for input_cosmology.
    This returns modified copy of the input dictionary object, which
    corresponds to a cosmology with massive neutrinos.
    """
    
    '''This is a horrible workaround, and I would like to get rid of it
    ASAP. It destroys dependence on TCMB and
    neutrino_hierarchy, possibly more. But CAMB does not accept omnuh2 as
    an input, so I have to reverse-engineer it somehow.
    
    Also, should we replace default_nnu with something else in the
    following expression? Even if we're changing N_massive to 1,
    N_total_eff = 3.046 nonetheless, right?'''
    full_cosmology = cp.deepcopy(mlc) 

    full_cosmology["mnu"] = omnuh2_in * camb.constants.neutrino_mass_fac / \
        (camb.constants.default_nnu / 3.0) ** 0.75 
    
    #print("The mnu value", mnu_in, "corresponds to the omnuh2 value",
    #    omnuh2_in)
    #full_cosmology["omch2"] -= omnuh2_in
    ''' The removal of the above line is a significant difference, but it
        allows us to transition more naturally into the emulator sample
        generating code.'''

    full_cosmology["nnu_massive"] = nnu_massive_in

    return full_cosmology

def input_cosmology(cosmology, hubble_units=False): 
    """
    Helper function for kzps.
    Read entries from a dictionary representing a cosmological configuration.
    Then write these values to a CAMBparams object and return. 
    
    Possible mistakes:
    A. We're setting "omk" with OmK * h ** 2. Should I have used OmK? If so,
        the capitalization here is nonstandard.
    """

    pars = camb.CAMBparams()

    h = cosmology["h"]

    # tau is a desperation argument
    ### Why are we using the degenerate hierarchy? Isn't that wrong?
    pars.set_cosmology(
        H0=h * 100,
        ombh2=cosmology["ombh2"],
        omch2=cosmology["omch2"],
        omk=cosmology["OmK"],
        mnu=cosmology["mnu"],
        num_massive_neutrinos=cosmology["nnu_massive"],
        tau=0.0952, # for justification, ask Matteo
        neutrino_hierarchy="degenerate" # 1 eigenstate approximation; our
        # neutrino setup (see below) is not valid for inverted/normal
        # hierarchies.
    )
 
    pars.InitPower.set_params(As=cosmology["A_s"], ns=cosmology["n_s"],
        r=0, nt=0.0, ntrun=0.0) # the last three are desperation arguments

    input_dark_energy(pars, cosmology["w0"], float(cosmology["wa"]))

    pars.Transfer.kmax = 10.0 if hubble_units else 10.0 / h

    return pars

def obtain_pspectrum(pars, zs=[0], k_points=100000, hubble_units=False):
    """
    Helper function for kzps.
    Given a fully set-up pars function, return the following in this order:
        scale axis, redshifts used, power spectra, and sigma12 values.

    """

    ''' To change the the extent of the k-axis, change the following line as
    well as the "get_matter_power_spectrum" call. '''
    pars.set_matter_power(redshifts=zs, kmax=20.0 / pars.h, nonlinear=False)
    
    results = camb.get_results(pars)

    sigma12 = results.get_sigmaR(12, hubble_units=False)
    
    '''
    In some cursory tests, the accurate_massive_neutrino_transfers
    flag did not appear to significantly alter the outcome.
    
    The flags var1=8 and var2=8 indicate that we are looking at the
    power spectrum of CDM + baryons (i.e. neutrinos excluded).
    '''
    k, z, p = results.get_matter_power_spectrum(
        minkh=1e-4 / pars.h, maxkh=10.0 / pars.h, npoints = k_points,
        var1=8, var2=8
    )
   
    # De-nest for the single-redshift case:
    if len(p) == 1:
        p = p[0]

    if not hubble_units:
        k *= pars.h
        p /= pars.h ** 3

    return k, z, p, sigma12

def kzps(cosmology, zs = [0], fancy_neutrinos=False, k_points=100000,
    hubble_units=False):
    """
    Returns the scale axis, redshifts, power spectrum, and sigma12
        of a massless-neutrino Lambda-CDM model
    @param cosmology : a dictionary of value for CAMBparams fields
    @param zs : redshifts at which to evaluate the model
        If you would like to fix the sigma12 value, specify this in the mlc
        dictionary and set this parameter to None. If you would not like to
        fix the sigma12 value, make sure that the mlc dictionary does not
        contain a non-None sigma12 entry.

        UPDATE: I changed my mind for now, let's say that z takes in case both
            are specified.
    @param omnuh2_in : neutrino physical mass density
    @fancy_neutrinos: flag sets whether we attempt to impose a neutrino
        scheme on CAMB after we've already set the physical density. The
        results seem to be inconsistent with observation.
    """
    ''' Retire this code block until we figure out z dominance
    if zs is None:
        assert "sigma12" in cosmology.keys(), \
            "Redshift and sigma12 cannot be supplied simultaneously."
    else:
        assert "sigma12" not in cosmology.keys() or mlc["sigma12"] is None, \
            "Redshift and sigma12 cannot be supplied simultaneously."
    '''

    pars = input_cosmology(cosmology)
    
    if fancy_neutrinos:
        make_neutrinos_fancy(pars, cosmology["nnu_massive"])
    
    apply_universal_output_settings(pars)
    
    return obtain_pspectrum(pars, zs, k_points=k_points,
        hubble_units=hubble_units) 

def obtain_pspectrum_interpolator(pars, zs=[0], z_points=150,
    hubble_units=False):
    """
    Helper function for kzps.
    Given a fully set-up pars function, return the following in this order:
        scale axis, redshifts used, power spectra, and sigma12 values.

    """

    ''' To change the the extent of the k-axis, change the following line as
    well as the "get_matter_power_spectrum" call. '''
    pars.set_matter_power(redshifts=zs, kmax=10.0 / pars.h, nonlinear=False)
    
    results = camb.get_results(pars)

    '''
    In some cursory tests, the accurate_massive_neutrino_transfers
    flag did not appear to significantly alter the outcome.
    
    The flags var1=8 and var2=8 indicate that we are looking at the
    power spectrum of CDM + baryons (i.e. neutrinos excluded).
    '''
    kmax = max(standard_k_axis) if hubble_units else max(standard_k_axis) / h

    PK = results.get_matter_power_interpolator(pars,
        zmin=min(_redshifts), zmax=max(_redshifts), nz_step=z_points,
        k_hunit=hubble_units, kmax=kmax, nonlinear=False, var1=8, var2=8,
        hubble_units=False
    )
   
    return PK

def kzps_interpolator(cosmology, zs = [0], fancy_neutrinos=False,
    z_points=150, hubble_units=False):
    """
    This is a really rough function, I'm just trying to test out an idea.
    """

    pars = input_cosmology(cosmology, hubble_units)
    
    if fancy_neutrinos:
        make_neutrinos_fancy(pars, cosmology["nnu_massive"])
    
    apply_universal_output_settings(pars)
    
    return obtain_pspectrum_interpolator(pars, zs, z_points=z_points,
        hubble_units=hubble_units) 

def model_ratios(snap_index, sims, canvas, massive=True, skips=[],
    subplot_indices=None, active_labels=['x', 'y'], title="Ground truth",
    omnuh2_str="0.002", models=cosm, suppress_legend=False):
    """
    Why is this a different function from above?
    There are a couple of annoying formatting differences with the power nu
    dictionary which add up to an unpleasant time trying to squeeze it into the
    existing function...

    Here, the baseline is always model 0,
    but theoretically it should be quite easy
    to generalize this function further.
    """
    P_accessor = None
    if massive == True:
         P_accessor = "P_nu"
    elif massive==False:
        P_accessor = "P_no"
 
    baseline_h = models.loc[0]["h"]
    baseline_k = correct_sims[0][snap_index]["k"]
    
    baseline_p = sims[0][snap_index]["P_nu"] / \
        sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p = sims[0][snap_index][P_accessor]
    
    plot_area = canvas # if subplot_indices is None
    if subplot_indices is not None:
        if type(subplot_indices) == int:
            plot_area = canvas[subplot_indices]
        else: # we assume it's a 2d grid of plots
            plot_area = canvas[subplot_indices[0], subplot_indices[1]]
        # No need to add more if cases because an n-d canvas of n > 2 makes no
        # sense.
    
    k_list = []
    rat_list = []
    for i in range(1, len(correct_sims)):
        if i in skips:
            continue # Don't know what's going on with model 8
        this_h = models.loc[i]["h"]
        this_k = correct_sims[i][snap_index]["k"]
        
        this_p = correct_sims[i][snap_index]["P_nu"] / \
            correct_sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p = correct_sims[i][snap_index][P_accessor]
        
        truncated_k, truncated_p, aligned_p = \
            truncator(baseline_k, baseline_p, this_k,
                this_p, interpolation=True)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, aligned_p / truncated_p,
                 label=label_in, c=colors[i], linestyle=styles[i])
       
        k_list.append(truncated_k)
        rat_list.append(aligned_p / truncated_p)
 
    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
   
    ylabel =  r"$x_i / x_0$"
    if P_accessor is not None:
        if massive == True:
            ylabel = r"$P_\mathrm{massive} / P_\mathrm{massive, model \, 0}$"
        if massive == False:
            ylabel = r"$P_\mathrm{massless} / P_\mathrm{massless, model \, 0}$"
    
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
    
    plot_area.set_title(title + r": $\omega_\nu$ = " + omnuh2_str + \
        "; Snapshot " + str(snap_index))
    if not suppress_legend:
        plot_area.legend()

    return k_list, rat_list

def compare_wrappers(k_list, p_list, correct_sims, snap_index,
    canvas, massive, subscript, title, skips=[], subplot_indices=None,
    active_labels=['x', 'y']):
    """
    Python-wrapper (i.e. Lukas') simulation variables feature the _py ending
    Fortran (i.e. Ariel's) simulation variables feature the _for ending
    """
    
    P_accessor = None
    if massive == True:
        P_accessor = "P_nu"
    elif massive == False:
        P_accessor = "P_no"
    x_mode = P_accessor is None

    # Remember, the returned redshifts are in increasing order
    # Whereas snapshot indices run from older to newer
    z_index = 4 - snap_index

    baseline_h = cosm.loc[0]["h"]
    
    baseline_k_py = k_list[0] * baseline_h
    
    baseline_p_py = None
    if x_mode:
        baseline_p_py = p_list[0][z_index]
    else:
        baseline_p_py = p_list[0][z_index] / baseline_h ** 3
    
    baseline_k_for = correct_sims[0][snap_index]["k"]
    
    baseline_p_for = correct_sims[0][snap_index]["P_nu"] / \
        correct_sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p_for = correct_sims[0][snap_index][P_accessor]
    
    plot_area = None
    if subplot_indices is None:
        plot_area = canvas
    elif type(subplot_indices) == int:
        plot_area = canvas[subplot_indices]
    else:
        plot_area = canvas[subplot_indices[0], subplot_indices[1]]

    # k_list is the LCD because Ariel has more working models than I do
    for i in range(1, len(k_list)):
        if i in skips:
            continue
        this_h = cosm.loc[i]["h"]
        
        this_k_py = k_list[i] * this_h
        this_p_py = None
        if x_mode==False:
            this_p_py = p_list[i][z_index] / this_h ** 3
        else:
            this_p_py = p_list[i][z_index]

        this_k_for = correct_sims[i][snap_index]["k"]
        
        this_p_for = correct_sims[i][snap_index]["P_nu"] / \
            correct_sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p_for = correct_sims[i][snap_index][P_accessor]

        truncated_k_py, truncated_p_py, aligned_p_py = \
            truncator(baseline_k_py, baseline_p_py, this_k_py,
                this_p_py, interpolation=this_h != baseline_h)
        y_py = aligned_p_py / truncated_p_py

        truncated_k_for, truncated_p_for, aligned_p_for = \
            truncator(baseline_k_for, baseline_p_for, this_k_for,
            this_p_for, interpolation=this_h != baseline_h)
        y_for = aligned_p_for / truncated_p_for

        truncated_k, truncated_y_py, aligned_p_for = \
            truncator_neutral(truncated_k_py, y_py, truncated_k_for, y_for) 

        label_in = "model " + str(i)
        plot_area.plot(truncated_k,
            truncated_y_py / aligned_p_for, label=label_in, c=colors[i],
            linestyle=styles[i])

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
    
    ylabel = None
    if x_mode:
        ylabel = r"$ж_i/ж_0$"
    else:
        ylabel = r"$y_\mathrm{py} / y_\mathrm{fortran}$"
    
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
 
    plot_area.set_title(title)
    plot_area.legend()

    plot_area.set_title(title)
    plot_area.legend()

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
    try:
        model = cosm.loc[model_num]
    
        for column in cosm.columns:
            if redshift_column.match(column):
                z.append(model[column])
    except (ValueError, KeyError):
        z = [3, 2, 1, 0]
            
    # No need to sort these because they should already
    # be in descending order.
    return np.array(z)

def truncator(base_x, base_y, obj_x, obj_y):
    # This doesn't make the same assumptions as truncator
    # But of course it's terrible form to leave both of these functions here.
    """
    Throw out base_x values until
        min(base_x) >= min(obj_x) and max(base_x) <= max(obj_x)    
    then interpolate the object arrays over the truncated base_x domain.
    @returns:
        trunc_x: truncated base_x array, which is now common to both y arrays
        trunc_y: truncated base_y array
        aligned_y: interpolation of obj_y over trunc_x
    """
    # What is the most conservative lower bound?
    lcd_min = max(min(obj_x), min(base_x))
    # What is the most conservative upper bound?
    lcd_max = min(max(obj_x), max(base_x))
    
    # Eliminate points outside the conservative bounds
    mask_base = np.all([[base_x <= lcd_max], [base_x >= lcd_min]], axis=0)[0]
    trunc_base_x = base_x[mask_base]
    trunc_base_y = base_y[mask_base]
   
    mask_obj = np.all([[obj_x <= lcd_max], [obj_x >= lcd_min]], axis=0)[0]
    trunc_obj_x = obj_x[mask_obj]
    trunc_obj_y = obj_y[mask_obj]
 
    interpolator = interp1d(obj_x, obj_y, kind="cubic")
    aligned_y = interpolator(trunc_base_x)

    #print(len(trunc_base_x), len(aligned_y)) 
    return trunc_base_x, trunc_base_y, aligned_y
