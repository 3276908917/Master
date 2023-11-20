import numpy as np
from scipy.interpolate import interp1d

def box(item):
    """
    This utility fn is handy when passing an argument to a fn which expects a
    list, but when the argument might be a scalar.
    """
    if isinstance(item, np.ndarray) or isinstance(item, list):
        return item
    
    if not isinstance(item, int) and not isinstance(item, float):
        raise Warning("Box was called on a single non-numeric item!")
        
    return np.array([item])
    
    
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


def normalize(array):
    min_ = np.min(array)
    max_ = np.max(array)
    range_ = max_ - min_
    shifted = array - min_
    return shifted / range_


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
    
def homogenize_k_axes(samples):
    """
    This takes as input a two-column matrix of arrays (first dimension: k,
    second dimension: P(k)). In other words, samples[i][1] is a power spectrum
    and samples[i][0] is the set of k values to which this power spectrum
    corresponds.
    
    Then it takes the most narrow row of k values as the baseline
    such that all other P(k) are interpolated to share k axes.
    """

    # First, let's find the rows with the smallest k_max and largest k_min:
    max_mink = np.min(samples[0][0])
    max_mink_i = 0   
    min_maxk = np.max(samples[0][0])
    min_maxk_i = 0

    for i in range(1, len(samples)):
        min_ = np.min(samples[i][0])
        max_ = np.max(samples[i][0])
        if min_ > max_mink:
            max_mink = min_
            max_mink_i = i
        if max_ < min_maxk:
            min_maxk = max_
            min_maxk_i = i

    base_k = samples[max_mink_i][0]
    base_P = samples[max_mink_i][1]
    interpd_spectra = np.zeros((len(samples), len(base_k)))

    if max_mink_i != min_maxk_i:
        # There is no single k array whose bounds fall within everyone else's.
        # Therefore, we use the k array with the smallest k_max to pare down
        # (via the truncator function) the k array with the largest k_min
        obj_k = samples[min_maxk_i][0]
        obj_P = samples[min_maxk_i][1]

        base_k, base_P, aligned_P = truncator(base_k, base_P, obj_k, obj_P) 

        interpd_spectra = np.zeros((len(samples), len(base_k)))
        interpd_spectra[min_maxk_i] = aligned_P
        interpd_spectra[max_mink_i] = base_P
    else:
        interpd_spectra[max_mink_i] = base_P

    for i in range(len(samples)):
        if i != max_mink_i and i != min_maxk_i:
            _, _, interpd_spectra[i] = \
                truncator(base_k, base_P, samples[i][0], samples[i][1])
     
    return base_k, interpd_spectra
    
def verify_LHS_dim(lhc_dim, priors):
    if lhc_dim == 3:
        if "sigma_12" in priors:
            raise ValueError("Found sigma12 in priors, but LHS is 3D!")
    if lhc_dim == 4: # this had better be a massless-neutrino case
        if "A_s" in priors:
            raise ValueError("Found A_s in priors, but LHS is 4D!")
        if "omnuh2" in priors:
            raise ValueError("Found omnuh2 in priors, but LHS is 4D!")
        if "sigma12" not in priors:
            raise ValueError("LHS is 4D, but sigma12 not found in priors!")
    elif lhc_dim == 6: # this had better be a massive-neutrino case
        if "A_s" not in priors:
            raise ValueError("LHS is 6D, but A_s not found in priors!")
        if "omnuh2" not in priors:
            raise ValueError("LHS is 6D, but omnuh2 not found in priors!")
        if "sigma12" not in priors:
            raise ValueError("LHS is 6D, but sigma12 not found in priors!")
    else: # unrecognized
        print("The LHS ought to have either three, four or six parameters" + \
                "per row! If you are trying to expand the parameter set, " + \
               "you will have to directly access the other scripts yourself.")
