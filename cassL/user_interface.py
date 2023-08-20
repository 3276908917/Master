import numpy as np
from scipy.interpolate import interp1d
import GPy
from cassL import lhc
from cassL import train_emu as te

# In keeping with the format of values typically quoted in the literature for
# the scalar mode amplitude (see, for example, Spurio Mancini et al, 2021 ),
# we here define our prior ranges with the use of exponentiation.
A_MIN = np.exp(1.61) / 10 ** 10
A_MAX = np.exp(5) / 10 ** 10

# The MINI qualifier indicates that these belong to the "classic" prior ranges
# (see the function get_param_ranges).
A_MINI_MIN = np.exp(2.35) / 10 ** 10
A_MINI_MAX = np.exp(3.91) / 10 ** 10

def percent_error(trusted, tested):
    """
    I don't have a great place for this function, but I'm tired of copying and
    pasting it.
    """
    return 100 * (tested - trusted) / trusted
    
def closest_index(array, target_value):
    """
    This function is used in train_emu.
    But I think we should split off a utils script, because percent_error and
    closest_index are not exactly user interface functions...
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array argument must be a numpy array")
        
    if array.ndim != 1:
        raise ValueError("This function can only accept 1D arrays.")
    
    for e in array:
        if not isinstance(e, np.floating) and not isinstance(e, np.integer):
            raise ValueError("The array must contain only numerical elements.")
    
    if not isinstance(target_value, float) and \
        not isinstance(target_value, int):
        raise TypeError("value argument must be numerical")
        
    if not np.all(array == np.sort(array)):
        raise ValueError("The array must already be sorted, or else the " +
            "returned index will be ambiguous.")
    
    error_list = abs(array - target_value)
    return error_list.index(min(error_list))
    

def get_param_ranges(priors="COMET", massive_neutrinos=True):
    """
    !
    Return a dictionary of arrays where each key is a cosmological parameter
    over which the emulator will be trained. The first entry of each array is
    the parameter's lower bound, the second entry is the parameter's upper
    bound.

    @priors string indicating which set of parameters to use.
        "MEGA": the original goal for this project, which unfortunately
            suffered from too many empty cells. The code has gone through
            several crucial bug fixes since switching to a different set of
            priors, so we need to test this prior suite again and re-assess the
            rate of empty cells.
        "classic": a prior range with widths in between those of "COMET" and
            "MEGA." We need to test this prior suite again to see if it still
            suffers from a large number of empty cells.
        "COMET" as of 19.06.23, this is the default for the emulator. It is
            the most restrictive of the three options and is intended to
            totally eliminate the problem of empty cells, so that a complete
            LHC can be used to train a demonstration emulator. The hope is for
            the demonstration emulator trained over such priors to be extremely
            accurate due to the very narrow permissible parameter values.

    @massive_neutrinos should be set to False when one is training the emulator
        for massless neutrinos. This is because the massless neutrino emulator
        should be using two fewer parameters--A_s and omega_nu_h2 are no longer
        appropriate.
    """
    param_ranges = {}

    if priors == "MEGA":
        param_ranges = {
            'ombh2': [0.005, 0.28],
            'omch2': [0.001, 0.99], # max 0.3?
            'n_s': [0.7, 1.3], # expand?
            'sigma12': [0.2, 1], # based on Sanchez et al 21 and
                # Sanchez 20, fig 2 
        }
    elif priors == "classic": # This is useful for a demo run. 
        param_ranges = {
            'ombh2': [0.01875, 0.02625],
            'omch2': [0.05, 0.255],
            'n_s': [0.84, 1.1],
            'sigma12': [0.2, 1], # based on Sanchez et al 21; Sanchez 20 fig 2 
        }
    elif priors=="COMET":
        param_ranges = {
            'ombh2': [0.0205, 0.02415],
            'omch2': [0.085, 0.155],
            'n_s': [0.92, 1.01],
            'sigma12': [0.2, 1], # based on Sanchez et al 21; Sanchez 20 fig 2 
        }

    if massive_neutrinos:
        if priors == "MEGA":
            param_ranges['A_s'] = [A_MIN, A_MAX]
        elif priors == "classic":
            param_ranges['A_s'] = [A_MINI_MIN, A_MINI_MAX]
        elif priors == "COMET": 
            param_ranges['A_s'] = [1.15e-9, A_MINI_MAX]

        param_ranges['omnuh2'] = [0., 0.01]

    return param_ranges

def initialize(num_samples=100, num_trials=100, priors="COMET",
    massive_neutrinos=True):
    """
    Try to save the results of previous runs as npy files, to ensure consistency
    of outcomes.
    """
    param_ranges = get_param_ranges(priors, massive_neutrinos)

    # We're keeping Omega_K = 0 fixed for now

    # tau must be an evolution parameter if we're not including it here
    hc, list_min_dist = lhc.generate_samples(param_ranges, num_samples,
        num_trials)

    return hc, list_min_dist

def load(hc_file="hc.py", samples_file="samples.npy"):
    """
    Return the hypercube and samples arrays based on the given file names. 
    """
    return np.load(hc_file), np.load(samples_file)

def homogenize_k_axes(samples):
    """
        This takes as input a two-column matrix of arrays (first dimension: k,
    second dimension: P(k)). Then it takes the first row as the baseline
    such that all other P(k) are interpolated as though they had the same
    k axis as the first row.
    """

    # First, let's find the rows with the smallest k_max and largest k_min:
    max_min = np.min(samples[0][0])
    max_min_i =  0   
    min_max = np.max(samples[0][0])
    min_max_i = 0

    for i in range(1, len(samples)):
        min_ = np.min(samples[i][0])
        max_ = np.max(samples[i][0])
        if min_ > max_min:
            max_min = min_
            max_min_i = i
        if max_ < min_max:
            min_max = max_
            min_max_i = i

    base_k = samples[max_min_i][0]
    base_P = samples[max_min_i][1]
    interpd_spectra = np.zeros((len(samples), len(base_k)))

    if max_min_i != min_max_i:
        obj_k = samples[min_max_i][0]
        obj_P = samples[max_min_i][1]

        base_k, base_P, aligned_P = truncator(base_k, base_P, obj_k, obj_P) 

        interpd_spectra = np.zeros((len(samples), len(base_k)))
        interpd_spectra[min_max_i] = aligned_P

    for i in range(len(samples)):
        if i != max_min_i and i != min_max_i:
            _, _, interpd_spectra[i] = \
                truncator(base_k, base_P, samples[i][0], samples[i][1])
     
    return base_k, interpd_spectra

def truncator(base_x, base_y, obj_x, obj_y):
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

    #print(min(trunc_obj_x), max(trunc_obj_x))
    #print(min(obj_x), max(obj_x))
    #print(min(trunc_base_x), max(trunc_base_x))
 
    interpolator = interp1d(obj_x, obj_y, kind="cubic")
    aligned_y = interpolator(trunc_base_x)

    #print(len(trunc_base_x), len(aligned_y)) 
    return trunc_base_x, trunc_base_y, aligned_y

def get_model(X, Y):
    # Does it work if three parameters predict a nine-thousand element array?
    
    # Should we use a 3D kernel because the X row is 3D?
    # Or should the kernel match the Y dimensions?
    ker = GPy.kern.Matern52(3, ARD=True) + GPy.kern.White(3)
    m = GPy.models.GPRegression(X, Y, ker)
    m.optimize(messages=True, max_f_eval=1000)
    return m   

def build_train_and_test_sets(scenario_file_handle):
    """
    Options in the scenario file:
    * A header with a comment explaining, in English, the basic gist of the
        scenario.
    * A totally-unique scenario name.
    * Flag: do we need to also create a corresponding new test hypercube?
    * Number of k points associated with each spectrum.
    * Number of samples associated with the hypercube, default 5k.
    
    THIS FUNCTION SHOULD ISSUE A VERY LOUD WARNING IF ANY VALUES ARE IMPLIED
    FROM DEFAULT VALUES!!!
    
    (have a very specific directory with the following subdirectories: \
        1. Initial LHCs (i.e. stuff spat out by lhs.py functions)
            New directories not necessary, because some scenarios will only
            generate one file anyway.
        2. Backups
            At its peak, this folder will contain six files from the current
            run: three for the most recent backups and three for the backups
            before that. Then the old backups will be promptly deleted.
        3. Final products
            * Create a new directory for each scenario!!
    """
    return 23

def build_and_test_emulator(X_train, Y_train, X_test, Y_test, priors):
    """
    Build a new Gaussian process regression over X_train and Y_train, then
    test its accuracy using X_test and Y_test.
    This function automatically throws out bad entries in all X and Y inputs.
    
    Just like the code in train_emu.py, this cannot yet handle the case of
    different sampling distributions, e.g. root- or square-sampling in sigma12.
    """
    X_train_clean, Y_train_clean = \
        te.eliminate_unusable_entries(X_train, Y_train)
    trainer = te.Emulator_Trainer(X_train_clean, Y_train_clean, priors)
    trainer.train()
    
    X_test_clean, Y_test_clean = \
        te.eliminate_unusable_entries(X_test, Y_test)
    # trainer.test(X
