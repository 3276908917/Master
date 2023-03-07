# This flag is sort of dumb, but it works as long as I'm the only one running
# the code in this repo.
linux = True

path_base_linux = "/home/lfinkbei/Documents/"
path_base_windows = "C:/Users/Lukas/Documents/GitHub/"
path_base = path_base_linux if linux else path_base_windows

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import camb
import re
from scipy.interpolate import interp1d

'''Keep in mind that this is NOT the same file as the original
"cosmology_Aletheia.dat" that Ariel gave us! If you use the unaltered version,
you will get a segfault'''
path_to_me = path_base + "Master/CAMB_notebooks/shared/"
cosm = pd.read_csv(path_to_me + "data/cosmologies.dat", sep='\s+')

omegas_nu = np.array([0.0006356, 0.002148659574468, 0.006356])#, 0.01])
# Add corresponding file accessors, to check our work later
omnu_strings = np.array(["0.0006", "0.002", "0.006"])#, "0.01"])

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

powernu2 = []
for i in range(0, 9): # iterate over models
    powernu2.append([])
    for j in range(0, 5): # iterate over snapshots
        powernu2[i].append(pd.read_csv(file_base + omnu + "_caso" + \
            str(i) + "_000" + str(j) + ".dat",
            names=["k", "P_no", "P_nu", "ratio"], sep='\s+'))

powernu3 = {}
for omnu in omnu_strings:
    powernu3[omnu] = []

    for i in range(0, 9): # iterate over models
        powernu3[omnu].append([])
        for j in range(0, 5): # iterate over snapshots
            powernu3[omnu][i].append(pd.read_csv(file_base + omnu + "_caso" + \
                str(i) + "_000" + str(j) + ".dat",
                names=["k", "P_no", "P_nu", "ratio"], sep='\s+'))

colors = ["green", "blue", "brown", "red", "black", "orange", "purple",
          "magenta", "cyan"] * 2

#styles = ["solid", "dotted", "dashed", "dashdot", "solid", "dotted", "dashed",
#    "dashdot"]
# Line styles are unfortunately too distracting in plots as dense as those with
# which we are here dealing; make everything solid
styles = ["solid"] * 18

def match_s12(target, tolerance, cosmology):
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
    """
    return 23

def get_cosmology(gen_order = ["M", "L"]):
    """
    gen order determines the order in which we generate random numbers deciding
    the energy budget of the cosmology. The order will determine where the bias
    lies, because whatever is generated earlier will be skewed in favor of
    larger values. For example, consider the default argument value. If we
    generate M first, we have access to the full range of values [0, 1]. When
    we generate L, we have access only to [0, 1 - M]. Then K is fully
    determined.

    I have an idea for how to advance this function: I am certain that in one
    of the papers that I've read for this project, the authors gave a data
    table which talked about the typical parameter ranges used in emulators.
    If any of these ranges is larger than the one's encoded below, let's seize!
    
    Unfortunately, all of these bounds are hard-coded. Maybe we can read in a
    table for this?
    """
    row = {}

    # Shape parameters: CONSTANT ACROSS MODELS
    row['ombh2'] = 0.022445
    row['omch2'] = 0.120567
    row['n_s'] = 0.96

    row['h'] = np.random.uniform(0.55, 0.79)
   
    # Given h, the following are now fixed:
    row['OmB'] = row['ombh2'] / row['h'] ** 2
    row['OmC'] = row['omch2'] / row['h'] ** 2
    row['OmM'] = row['OmB'] + row['OmC'] 

    row['OmK'] = np.random.uniform(-0.05, 0)
    row['OmL'] = 1 - row['OmM'] - row['OmK']
    
    #~ Do we have any constraints on h besides Aletheia?
    # I ask because this seems like a pretty small window.
    #ditto
    row['w0'] = np.random.uniform(-0.85, -1.15)
    # ditto
    row['wa'] = np.random.uniform(-0.20, 0.20)

    row['A_s'] = np.random.uniform(1.78568440085517E-09, 2.48485942677850E-09)

    #~ Should we compute omnuh2 here, or leave that separate?

    #~ Should we also specify parameters not specified by the Aletheia data
        # table, for example tau or the CMB temperature?

    return row

def boltzmann_battery(onh2, skips=[8]):
    """
    This is basically legacy code. Most of the notebooks still use this
    function, but I am trying to phase it out in favor of better_battery.
    """
    k_massless_list = []
    z_massless_list = []
    p_massless_list = []
    s12_massless_list = []

    k_massive_list = []
    z_massive_list = []
    p_massive_list = []
    s12_massive_list = []

    for index, row in cosm.iterrows():
        if index in skips:
            # For example, I don't yet understand how to implement model 8
            continue
        
        z_in = parse_redshifts(index)
        k, z, p, s12 = kzps(row, onh2, nu_massive=False, zs=z_in)
        k_massless_list.append(k)
        z_massless_list.append(z)
        p_massless_list.append(p)
        s12_massless_list.append(s12)
        
        k, z, p, s12 = kzps(row, onh2, nu_massive=True, zs=z_in)
        k_massive_list.append(k)
        z_massive_list.append(z)
        p_massive_list.append(p)
        s12_massive_list.append(s12)

    return k_massless_list, z_massless_list, p_massless_list, \
        s12_massless_list, k_massive_list, z_massive_list, p_massive_list, \
        s12_massive_list
        
def better_battery(onh2s, onh2_strs, skips_omega = [0, 2],
    skips_model=[8], skips_snapshot=[1, 2, 3], h_units=False,
    models=cosm):
    """
    Similar procedure to boltzmann_battery, but with an architecture that
    more closely agrees with that of Ariel's in the powernu results.
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

    for om_index in range(len(onh2s)):
        print(om_index)
        om = onh2s[om_index]
        om_str = onh2_strs[om_index]
        if om_index in skips_omega:
            spec_sims[om_str] = None
            continue
        spec_sims[om_str] = []
        for mindex, row in models.iterrows():
            h = row["h"]
            if mindex in skips_model:
                # For example, I don't yet understand how to implement model 8
                spec_sims[om_str].append(None)
                continue
            spec_sims[om_str].append([])
       
            z_input = parse_redshifts(mindex)
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
                    spec_sims[om_str][mindex].append(None)
                    continue
                #print("using", z_index)
                inner_dict = {}
                z = z_input[snap_index]
              
                '''
                Double check that I have it right: k multiplied by h;
                    P divided by h^3
                '''
                massless_tuple = kzps(row, om, nu_massive=False, zs=[z])
                inner_dict["k"] = massless_tuple[0] if h_units \
                    else massless_tuple[0] * h
                inner_dict["P_no"] = massless_tuple[2] if h_units \
                    else massless_tuple[2] / h ** 3
                inner_dict["s12_massive"] = massless_tuple[3]

                massive_tuple = kzps(row, om, nu_massive=True, zs=[z])
                inner_dict["P_nu"] = massive_tuple[2] if h_units \
                    else massive_tuple[2] / h ** 3
                inner_dict["s12_massless"] = massive_tuple[3]
                
                # Temporary addition, for debugging
                inner_dict["z"] = z_input[snap_index]               
 
                assert np.array_equal(massless_tuple[0], massive_tuple[0]), \
                   "assumption of identical k axies not satisfied!"
                    
                spec_sims[om_str][mindex].append(inner_dict) 

    return spec_sims

def kzps(mlc, omnuh2_in, nu_massive=False, zs = [0], nnu_massive_in=1):
    """
    Returns the scale axis, redshifts, power spectrum, and sigma12
    of a Lambda-CDM model
    @param mlc : "MassLess Cosmology"
        a dictionary of values
        for CAMBparams fields
    @param omnuh2_in : neutrino physical mass density
    @nu_massive : if this is True,
        the value in omnuh2_in is used to set omnuh2.
        If this is False,
        the value in omnuh2_in is simply added to omch2.

    Possible mistakes:
    A. We're setting "omk" with OmK * h ** 2. Should I have used OmK? If so,
        the capitalization here is nonstandard.
    """ 
    pars = camb.CAMBparams()
    omch2_in = mlc["omch2"]
 
    mnu_in = 0
    nnu_massive = 0
    h = mlc["h"]

    if nu_massive:
        '''This is a horrible workaround, and I would like to get rid of it
        ASAP The following line destroys dependence on TCMB and
        neutrino_hierarchy, possibly more. But CAMB does not accept omnuh2 as
        an input, so I have to reverse-engineer it somehow.
        
        Also, should we replace default_nnu with something else in the
        following expression? Even if we're changing N_massive to 1,
        N_total_eff = 3.046 nonetheless, right?'''
        mnu_in = omnuh2_in * camb.constants.neutrino_mass_fac / \
            (camb.constants.default_nnu / 3.0) ** 0.75 
        #print("The mnu value", mnu_in, "corresponds to the omnuh2 value",
        #    omnuh2_in)
        omch2_in -= omnuh2_in
        nnu_massive = nnu_massive_in

    # tau is a desperation argument
    pars.set_cosmology(
        H0=h * 100,
        ombh2=mlc["ombh2"],
        omch2=omch2_in,
        omk=mlc["OmK"] * h ** 2,
        mnu=mnu_in,
        num_massive_neutrinos=nnu_massive,
        tau=0.0952, # just like in Matteo's notebook, at least (but maybe I got
            # this value from somewhere else...
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
    
    pars.InitPower.set_params(As=mlc["A_s"], ns=mlc["n_s"],
        r=0, nt=0.0, ntrun=0.0) # the last three are desperation arguments
    pars.set_matter_power(redshifts=zs, kmax=20.0 / h, nonlinear=False)
    
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

    # desperation if statement
    if mlc["w0"] != -1 or float(mlc["wa"]) !=0:
        pars.set_dark_energy(w=mlc["w0"], wa=float(mlc["wa"]),
            dark_energy_model='ppf')
    '''
    To change the the extent of the k-axis,
    change the following line as well as the "get_matter_power_spectrum" call
    
    In some cursory tests, the accurate_massive_neutrino_transfers
    flag did not appear to significantly alter the outcome.
    '''
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    
    # The flags var1=8 and var2=8 indicate that we are looking at the
    # power spectrum of CDM + baryons (i.e. neutrinos excluded).
    k, z, p = results.get_matter_power_spectrum(
        minkh=1e-4 / h, maxkh=10.0 / h, npoints = 100000,
        var1=8, var2=8
    )
    sigma12 = results.get_sigmaR(12, hubble_units=False)
   
    # De-nest for the single-redshift case:
    if len(p) == 1:
        p = p[0] 
    return k, z, p, sigma12 

def model_ratios_old(k_list, p_list, snap_index, canvas, subscript, title,
    skips=[], subplot_indices=None, active_labels=['x', 'y'], x_mode=False):
    """
    Plot the ratio of @p_list[i] to @p_list[0] for all i.

    Here, the baseline is always model 0, but theoretically it should be quite
    easy to generalize this function further.
    """
    z_index = 4 - snap_index
    # Remember, the returned redshifts are in increasing order
    # Whereas snapshot indices run from older to newer
    baseline_h = cosm.loc[0]["h"]
    baseline_k = k_list[0] * baseline_h

    baseline_p = None
    if x_mode==False:
        baseline_p = p_list[0][z_index] / baseline_h ** 3
    else:
        baseline_p = p_list[0][z_index]
    
    plot_area = canvas # if subplot_indices is None
    if subplot_indices is not None:
        if type(subplot_indices) == int:
            plot_area = canvas[subplot_indices]
        else: # we assume it's a 2d grid of plots
            plot_area = canvas[subplot_indices[0], subplot_indices[1]]
        # No need to add more if cases because an n-d canvas of n > 2 makes no
        # sense.

    for i in range(1, len(k_list)):
        if i in skips:
            continue
        this_h = cosm.loc[i]["h"]
        this_k = k_list[i] * this_h
        
        this_p = None
        if x_mode==False:
            this_p = p_list[i][z_index] / this_h ** 3
        else:
            this_p = p_list[i][z_index]

        truncated_k, truncated_p, aligned_p = truncator(baseline_k, baseline_p,
            this_k, this_p, interpolation=True)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, aligned_p / truncated_p, label=label_in,
            c=colors[i], linestyle=styles[i])

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
    
    ylabel = None
    if x_mode == False:
        ylabel = r"$P_\mathrm{" + subscript + "} /" + \
            r" P_\mathrm{" + subscript + ", model \, 0}$"
    else:
        ylabel = r"$x_i / x_0$"

    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
    
    plot_area.set_title(title)
    plot_area.legend()

def model_ratios_true(snap_index, correct_sims, canvas, massive=True, skips=[],
    subplot_indices=None, active_labels=['x', 'y'], title="Ground truth",
    omnuh2_str="0.002", models=cosm):
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
    
    baseline_p = correct_sims[0][snap_index]["P_nu"] / \
        correct_sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p = correct_sims[0][snap_index][P_accessor]
    
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

def truncator_neutral(base_x, base_y, obj_x, obj_y):
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
