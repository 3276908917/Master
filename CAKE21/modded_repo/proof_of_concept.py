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

    if max_min_i != min_max_i:
        obj_x = samples[min_max_i][0]
        obj_y = samples[max_min_i][1]

        base_x, base_y, aligned_y = truncator(base_x, base_y, obj_x, obj_y) 

        # The following assignment might throw some errors because we gave a
        # fixed np.zeros type of array, and the truncated arrays will certainly
        # not fill it up. But let's worry about that when we hit the problem.
        
        #samples[max_min_i][0] = trunc_x
        samples[max_min_i][1] = base_y
        #samples[min_max_i][0] = trunc_x
        samples[min_max_i][1] = aligned_y

    for i in range(len(samples)):
        samples[i][0] = base_x
        if i != max_min_i and i != min_max_i:
            # I'm not 100% sure about this underscore dummy syntax
            _, _, samples[i][1] = \
                truncator(base_x, base_y, samples[i][0], samples[i][1])
     
    return samples

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
    mask = np.all([[base_x < lcd_max], [base_x > lcd_min]], axis=0)[0]
    trunc_x = base_x[mask]
    trunc_y = base_y[mask]
    
    interpolator = interp1d(obj_x, obj_y, kind="cubic")
    aligned_y = interpolator(base_x)
    
    return trunc_x, trunc_y, aligned_y
