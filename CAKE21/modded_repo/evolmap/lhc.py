"""

Module for generating and dealing with
latin hypercubes.

"""

import numpy as np
from pyDOE import lhs
from six import iteritems
from scipy.spatial.distance import cdist
from collections import OrderedDict


class LhcException(Exception):
    """
    Class for Exceptions in the
    latin hypercube module

    """
    pass


def generate_samples(ranges, n_samples, n_trials=0, validation=False):
    """
    Generate samples on a latin hypercube

    Parameters
    ----------
    ranges : collections:OrderedDict
        an ordered dictionary, the order
        of the outputs in latin hypercube
        is set by this order
    n_samples : int
        The number of samples to generate
    n_trials : int
        Regenerate cube this number of times
        to find the cube that maximises the
        minimum distance between samples
    validation : bool (Optional)
        Generate random verification data

    Returns
    -------
    samples : 2d array
        the latin hypercube
    """
    
    '''
    Lukas:
    Did you mean to say, "if not"?
    Anyway, it would be bad if you did, because for some reason,
    isinstance(OrderedDict({"one": 1, "two": 2}), type(OrderedDict)) == False
    one should instead write
    isinstance(OrderedDict({"one": 1, "two": 2}), OrderedDict)
    '''
    if isinstance(ranges, type(OrderedDict)):
        raise LhcException('Ranges argument is meant to be an '
                           'OrderedDict object')

    n_params = len(ranges)

    # If validation is True, we generate a random distribution of points
    if validation:
        samples = np.random.rand(n_samples, n_params)
    # Else, we generate the training sample using a routine from pyDOE
    # that creates a LHC
    else:
        '''
        This is the external function, 'lhs'. It needs to know the size
        of the parameter space, the number of points you want, and the
        criterion to select points. 'center' means that it first divides
        each dimension of the parameter space into n_samples equispaced
        intervals and it picks the middle point of the chosen interval
        '''
        samples = lhs(n=n_params, samples=n_samples, criterion='center')
        # We want to get a good configuration for the LHC, that maximises
        # the minimum distance among points.
        dist = cdist(samples, samples, metric='euclidean')
        min_dist = np.amin(dist[dist > 0])
        # Iterating this for n_trials times
        for n in range(n_trials):
            samples_new = lhs(n=n_params, samples=n_samples,
                              criterion='center')
            dist = cdist(samples_new, samples_new, metric='euclidean')
            min_dist_new = np.amin(dist[dist > 0])
            # If the new minimum distance is larger than the one we stored,
            # then we pick this configuration
            if (min_dist_new > min_dist):
                min_dist = min_dist_new
                samples = samples_new

    '''
    The lhs function returns a LHC normalized within the range [0,1]
    for each dimension. We have to rescale the LHC to match the range
    we want for our cosmological parameters
    '''
    for n, (par, par_range) in enumerate(iteritems(ranges)):
        samples[:, n] = samples[:, n] * (par_range[1] - par_range[0]) \
            + par_range[0]

    return samples
