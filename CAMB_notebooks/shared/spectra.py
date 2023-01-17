import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import camb
import re
from scipy.interpolate import interp1d

'''Keep in mind that this is NOT the same file as the original
"cosmology_Aletheia.dat" that Ariel gave us! If you use the unaltered version,
you will get a segfault'''
path_to_me = "/home/lfinkbei/Documents/Master/CAMB_notebooks/shared/"
cosm = pd.read_csv(path_to_me + "data/cosmologies.dat", sep='\s+')

omegas_nu = np.array([0.0006356, 0.002, 0.006356])
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

powernu2 = []
for i in range(0, 9): # iterate over models
    powernu2.append([])
    for j in range(0, 5): # iterate over snapshots
        powernu2[i].append(pd.read_csv(file_base + omnu + "_caso" + \
            str(i) + "_000" + str(j) + ".dat",
            names=["k", "P_no", "P_nu", "ratio"], sep='\s+'))

colors = ["green", "blue", "brown", "red", "black", "orange", "purple",
          "magenta", "cyan"]

#styles = ["solid", "dotted", "dashed", "dashdot", "solid", "dotted", "dashed",
#    "dashdot"]
# Line styles are unfortunately too distracting in plots as dense as those with
# which we are here dealing; make everything solid
styles = ["solid"] * 9

def boltzmann_battery(onh2, skips=[8]):
    """
    The returns are kind of ugly here, but my hand is somewhat tied by the
    state of the existing code. It might be worthwhile to reform this at some
    point.

    For example, the following kind of object would be more powerful:
    spec_sims[omega_nu][massive][model 0][snapshot]["k"]
    i.e. a dictionary within an array within an array within a dictionary
        within a dictionary.

    In case the user wants to calculate just for one omega_nu value (to save
    time in a perfectly reasonable way), we could probably re-use the
    formatting, but simply have spec_sims[omega_nu != desired_omega_nu] return
    None.
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
        k, z, p, s12 = kzps(row, onh2, massive_neutrinos=False,
                           zs=z_in)
        k_massless_list.append(k)
        z_massless_list.append(z)
        p_massless_list.append(p)
        s12_massless_list.append(s12)
        
        k, z, p, s12 = kzps(row, onh2, massive_neutrinos=True,
                           zs=z_in)
        k_massive_list.append(k)
        z_massive_list.append(z)
        p_massive_list.append(p)
        s12_massive_list.append(s12)

    return k_massless_list, z_massless_list, p_massless_list, \
        s12_massless_list, k_massive_list, z_massive_list, p_massive_list, \
        s12_massive_list
        
def better_battery(onh2s, onh2_strs, skips=[8]):
    """
    The returns are kind of ugly here, but my hand is somewhat tied by the
    state of the existing code. It might be worthwhile to reform this at some
    point.

    For example, the following kind of object would be more powerful:
    spec_sims[omega_nu][massive][model 0][snapshot]["k"]
    i.e. a dictionary within an array within an array within a dictionary
        within a dictionary.
        
    to better pass to powernu:
    spec_sims[onh2_str][model i][snap_index][P_accessor]

    In case the user wants to calculate just for one omega_nu value (to save
    time in a perfectly reasonable way), we could probably re-use the
    formatting, but simply have spec_sims[omega_nu != desired_omega_nu] return
    None.
    """
    assert type(onh2s) == list or type(onh2s) == np.ndarray, \
        "if you want only one omega value, you must still nest it in a list"
    assert type(onh2_strs) == list or type(onh2_strs) == np.ndarray, \
        "if you want only one omega value, you must still nest it in a list"
    assert len(onh2s) == len(onh2_strs), "more or fewer labels than points"
    
    spec_sims = {}
    
    k_massless_list = []
    z_massless_list = []
    p_massless_list = []
    s12_massless_list = []

    k_massive_list = []
    z_massive_list = []
    p_massive_list = []
    s12_massive_list = []

    for om_index in range(len(onh2s)):
        spec_sims[onh2_strs[om_index]] = []
        for mindex, row in cosm.iterrows():
            spec_sims[onh2_strs[om_index]].append([])
            if mindex in skips:
                # For example, I don't yet understand how to implement model 8
                continue
        
            z_input = parse_redshifts(mindex)
        
            for z_index in range(len(z_input)):
                z = z_input[z_index]
                spec_sims[onh2_strs[om_index]][mindex].append({})
                k_massless, _, p_massless, s12_massless = \
                    kzps(row, onh2s[om_index], massive_neutrinos=False, zs=[z])
                spec_sims[onh2_strs[om_index]][mindex][z_index]["k"] = \
                    k_massless
                spec_sims[onh2_strs[om_index]][mindex][z_index]["P_no"] = \
                    p_massless
                spec_sims[onh2_strs[om_index]][mindex][z_index]["s12_massless"] = \
                    s12_massless
                
                k_massive, _, p_massive, s12_massive = \
                    kzps(row, onh2s[om_index], massive_neutrinos=True, zs=[z])
                assert np.array_equal(k_massless, k_massive), "assumption" + \
                    " of identical k axies turned out to be incorrect..."
                
                spec_sims[onh2_strs[om_index]][mindex][z_index]["P_nu"] = \
                    p_massive
                spec_sims[onh2_strs[om_index]][mindex][z_index]["s12_massive"] = \
                    s12_massive   

    return spec_sims

def kzps(mlc, omnuh2_in, massive_neutrinos=False, zs = [0], nnu_massive_in=1):
    """
    Returns the scale axis, redshifts, power spectrum, and sigma12
    of a Lambda-CDM model
    @param mlc : "MassLess Cosmology"
        a dictionary of values
        for CAMBparams fields
    @param omnuh2_in : neutrino physical mass density
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
        minkh=1e-4, maxkh=10.0, npoints = 100000,
        var1=8, var2=8
    )
    sigma12 = results.get_sigmaR(12, hubble_units=False)
    
    return k, z, p, sigma12 

def model_ratios(k_list, p_list, snap_index, canvas, subscript, title,
    skips=[], subplot_indices=None, active_labels=['x', 'y']):
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
    baseline_p = p_list[0][z_index] / baseline_h ** 3
    
    plot_area = canvas if subplot_indices is None else \
        canvas[subplot_indices[0], subplot_indices[1]]

    for i in range(1, len(k_list)):
        if i in skips:
            continue
        this_h = cosm.loc[i]["h"]
        this_k = k_list[i] * this_h
        this_p = p_list[i][z_index] / this_h ** 3

        truncated_k, truncated_p, aligned_p = \
            truncator(baseline_k, baseline_p, this_k,
                      this_p, interpolation=this_h != baseline_h)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k,
            aligned_p / truncated_p, label=label_in, c=colors[i],
            linestyle=styles[i])

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
    
    ylabel = r"$P_\mathrm{" + subscript + "} /" + \
        r" P_\mathrm{" + subscript + ", model \, 0}$"
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
    
    plot_area.set_title(title)
    plot_area.legend()

def model_ratios_true(snap_index, onh2_str, canvas, massive=True, skips=[],
    subplot_indices=None, active_labels=['x', 'y']):
    """
    Why is this a different function from above?
    There are a couple of annoying formatting differences with the power nu
    dictionary which add up to an unpleasant time trying to squeeze it into the
    existing function...

    Here, the baseline is always model 0,
    but theoretically it should be quite easy
    to generalize this function further.
    """
    P_accessor = "P_nu" if massive else "P_no"  
    baseline_h = cosm.loc[0]["h"]
    baseline_k = powernu[onh2_str][0][snap_index]["k"]
    baseline_p = powernu[onh2_str][0][snap_index][P_accessor]
    
    plot_area = canvas if subplot_indices is None else \
        canvas[subplot_indices[0], subplot_indices[1]]
    
    for i in range(1, len(powernu[onh2_str])):
        if i in skips:
            continue # Don't know what's going on with model 8
        this_h = cosm.loc[i]["h"]
        this_k = powernu[onh2_str][i][snap_index]["k"]
        this_p = powernu[onh2_str][i][snap_index][P_accessor]
    
        truncated_k, truncated_p, aligned_p = \
            truncator(baseline_k, baseline_p, this_k,
                this_p, interpolation=this_h != baseline_h)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, aligned_p / truncated_p,
                 label=label_in, c=colors[i], linestyle=styles[i])
        
    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
    
    ylabel = r"$P_\mathrm{massive} / P_\mathrm{massive, model \, 0}$" if \
        massive else r"$P_\mathrm{massless} / P_\mathrm{massless, model \, 0}$"
    
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
    
    plot_area.set_title(r"Ground truth: $\omega_\nu$ = " + onh2_str + "; " + \
             "Snapshot " + str(snap_index))
    plot_area.legend()

def model_ratios_true2(snap_index, canvas, massive=True, skips=[],
    subplot_indices=None, active_labels=['x', 'y']):
    P_accessor = "P_nu" if massive else "P_no"  
    baseline_h = cosm.loc[0]["h"]
    baseline_k = powernu2[0][snap_index]["k"]
    baseline_p = powernu2[0][snap_index][P_accessor]
    
    plot_area = canvas if subplot_indices is None else \
        canvas[subplot_indices[0], subplot_indices[1]]
    
    for i in range(1, len(powernu2)):
        if i in skips:
            continue # Don't know what's going on with model 8
        this_h = cosm.loc[i]["h"]
        this_k = powernu2[i][snap_index]["k"]
        this_p = powernu2[i][snap_index][P_accessor]
    
        truncated_k, truncated_p, aligned_p = \
            truncator(baseline_k, baseline_p, this_k,
                this_p, interpolation=this_h != baseline_h)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, aligned_p / truncated_p,
                 label=label_in, c=colors[i], linestyle=styles[i])
        
    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
    
    ylabel = r"$P_\mathrm{massive} / P_\mathrm{massive, model \, 0}$" if \
        massive else r"$P_\mathrm{massless} / P_\mathrm{massless, model \, 0}$"
    
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
    
    plot_area.set_title(r"Ground truth: $\omega_\nu = 0.002$; " + \
             "Snapshot " + str(snap_index))
    plot_area.legend()

def compare_wrappers(k_list, p_list, onh2_str, snap_index, canvas, massive,
    subscript, title, skips=[], subplot_indices=None, active_labels=['x', 'y']):
    """
    Python-wrapper (i.e. Lukas') simulation variables feature the _py ending
    Fortran (i.e. Ariel's) simulation variables feature the _for ending
    """
    # Remember, the returned redshifts are in increasing order
    # Whereas snapshot indices run from older to newer
    z_index = 4 - snap_index

    # Don't need to double-up on this one; we're comparing like to like
    baseline_h = cosm.loc[0]["h"]
    
    baseline_k_py = k_list[0] * baseline_h
    baseline_p_py = p_list[0][z_index] / baseline_h ** 3
    
    P_accessor = "P_nu" if massive else "P_no"  
    baseline_k_for = powernu[onh2_str][0][snap_index]["k"]
    baseline_p_for = powernu[onh2_str][0][snap_index][P_accessor]
    
    plot_area = None
    if subplot_indices is None:
        plot_area = canvas
    elif type(subplot_indices) == int:
        plot_area = canvas[subplot_indices]
    else:
        plot_area = canvas[subplot_indices[0], subplot_indices[1]]

    # k_list is the LCD because Ariel has more working models
    for i in range(1, len(k_list)):
        # I'm going to have to interpolate between these...
        if i in skips:
            continue
        # Don't need to double-up on this one; we're comparing like to like
        this_h = cosm.loc[i]["h"]
        
        this_k_py = k_list[i] * this_h
        this_p_py = p_list[i][z_index] / this_h ** 3
        
        this_k_for = powernu[onh2_str][i][snap_index]["k"]
        this_p_for = powernu[onh2_str][i][snap_index][P_accessor]

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
    
    ylabel = r"$y_\mathrm{py} / y_\mathrm{fortran}$"
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
    
    plot_area.set_title(title)
    plot_area.legend()

    plot_area.set_title(title)
    plot_area.legend()

def compare_wrappers2(k_list, p_list, snap_index, canvas, massive, subscript,
    title, skips=[], subplot_indices=None, active_labels=['x', 'y']):
    """
    Python-wrapper (i.e. Lukas') simulation variables feature the _py ending
    Fortran (i.e. Ariel's) simulation variables feature the _for ending
    """
    # Remember, the returned redshifts are in increasing order
    # Whereas snapshot indices run from older to newer
    z_index = 4 - snap_index

    # Don't need to double-up on this one; we're comparing like to like
    baseline_h = cosm.loc[0]["h"]
    
    baseline_k_py = k_list[0] * baseline_h
    baseline_p_py = p_list[0][z_index] / baseline_h ** 3
    
    P_accessor = "P_nu" if massive else "P_no"  
    baseline_k_for = powernu2[0][snap_index]["k"]
    baseline_p_for = powernu2[0][snap_index][P_accessor]
    
    plot_area = None
    if subplot_indices is None:
        plot_area = canvas
    elif type(subplot_indices) == int:
        plot_area = canvas[subplot_indices]
    else:
        plot_area = canvas[subplot_indices[0], subplot_indices[1]]

    # k_list is the LCD because Ariel has more working models
    for i in range(1, len(k_list)):
        # I'm going to have to interpolate between these...
        if i in skips:
            continue
        # Don't need to double-up on this one; we're comparing like to like
        this_h = cosm.loc[i]["h"]
        
        this_k_py = k_list[i] * this_h
        this_p_py = p_list[i][z_index] / this_h ** 3
        
        this_k_for = powernu2[i][snap_index]["k"]
        this_p_for = powernu2[i][snap_index][P_accessor]

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
