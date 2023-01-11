import numpy as np
from scipy.interpolate import interp1d

#Here is some demo code that I used to get to start this off:                    

def initialize():
    """
    This function is no longer necessary if you have .npy files from a previous
    run.
    """
    ombh2 = 0.022445                                                                
    omch2 = 0.120567                                                                
    hc = evolmap.lhc.generate_samples({                                             
        'om_b': [0.9 * ombh2, 1.1 * omch2],                                         
        'om_c': [0.9 * omch2, 1.1 * omch2],                                         
        'h': [.603, .737]}, 100, 100)                                               

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

    base_x = samples[max_min_i][0]
    base_y = samples[max_min_i][1]
    homo_samples = np.zeros((len(samples), 2, len(base_x)))

    if max_min_i != min_max_i:
        obj_x = samples[min_max_i][0]
        obj_y = samples[max_min_i][1]

        base_x, base_y, aligned_y = truncator(base_x, base_y, obj_x, obj_y) 

        homo_samples = np.zeros((len(samples), 2, len(base_x)))

        homo_samples[max_min_i][1] = base_y
        homo_samples[min_max_i][1] = aligned_y

    for i in range(len(samples)):
        homo_samples[i][0] = base_x
        if i != max_min_i and i != min_max_i:
            # I'm not 100% sure about this underscore dummy syntax
            _, _, homo_samples[i][1] = \
                truncator(base_x, base_y, samples[i][0], samples[i][1])
     
    return homo_samples

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
    mask_base = np.all([[base_x < lcd_max], [base_x > lcd_min]], axis=0)[0]
    trunc_base_x = base_x[mask_base]
    trunc_base_y = base_y[mask_base]
   
    mask_obj = np.all([[obj_x < lcd_max], [obj_x > lcd_min]], axis=0)[0]
    trunc_obj_x = obj_x[mask_obj]
    trunc_obj_y = obj_y[mask_obj]

    #print(min(trunc_obj_x), max(trunc_obj_x))
    #print(min(obj_x), max(obj_x))
    #print(min(trunc_base_x), max(trunc_base_x))
 
    interpolator = interp1d(obj_x, obj_y, kind="cubic")
    aligned_y = interpolator(trunc_base_x)
    
    return trunc_base_x, trunc_base_y, aligned_y
