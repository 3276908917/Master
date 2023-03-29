import numpy as np
from scipy.interpolate import interp1d
import GPy
import evolmap.lhc

A_MIN = np.exp(1.61) / 10 ** 10
A_MAX = np.exp(5) / 10 ** 10

# Here is some demo code that I used to start this off:                    

def test():
    model0 = np.array([0.022445, 0.120567, 0.67])
    model0 = model0.reshape(3, 1)
    hc, samples = load("evolmap/hc.npy", "evolmap/samples.npy")
    base_k, homogenized_P = homogenize_k_axes(samples)
    gp_model = get_model(hc, homogenized_P)

    return gp_model.predict_noiseless(model0)

def initialize(num_samples=100, num_trials=100):
    """
    Try to save the results of previous runs as npy files, to ensure consistency
    of outcomes.
    """

    # We're keeping A_s fixed, right?
    # We're also keeping Omega_K = 0 fixed
    
    # tau must be an evolution parameter if we're not including it here

    hc, list_min_dist = evolmap.lhc.generate_samples({                                             
        'om_b': [0.005, 0.28],                                         
        'om_c': [0.001, 0.99],    # max 0.3?                                     
        #'h': [0.2, 1.0],
        'n_s': [0.7, 1.3], # expand?
        #'tau': [0.01, 0.8], 
        #'w0': [-2, -0.5],
        #'wa': [-0.5, 0.5],
        'sigma12': [0.2, 1], # based on Sanchez et al 21 and
            # Sanchez 20, figure 2 
        'om_nu': [0.0006356, 0.01],
        'A_s': [A_MIN, A_MAX]
    }, num_samples, num_trials)

    return hc, list_min_dist

    ''' Now we want to attach a power spectrum to each cosmology. What values
        do we take for the remaining parameters? It shouldn't matter, as long
        as they yield the same sigma_12. So you could assume model0, then
        simply rescale the result according to the principle of evolution
        mapping.'''

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
 
