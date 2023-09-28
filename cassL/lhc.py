"""

Module for generating and dealing with
latin hypercubes.

"""

import numpy as np
from pyDOE2 import lhs
from six import iteritems
from scipy.spatial.distance import cdist
from collections import OrderedDict
import time

def generate_samples(ranges, n_samples, n_trials=0, validation=False):
    r"""
    Generate samples on a latin hypercube

    This is the function that served us well for a long time. It should be the
    last outdated function to be deleted.

    Parameters
    ----------
    ranges : collections:OrderedDict
        an ordered dictionary, the order of the outputs in latin hypercube is
        set by this order
    n_samples : int
        The number of samples to generate
    n_trials : int
        Regenerate cube this number of times to find the cube that maximises
        the minimum distance between samples
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
        list_min_dist = []
        '''
        This is the external function, 'lhs.' It needs to know the size
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
                list_min_dist.append((n, min_dist_new))
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

    return samples, np.array(list_min_dist)

def long_term_LHC_builder(param_ranges, n_samples, label="unlabeled"):
    n_params = len(param_ranges, comparator=np.greater)
    """
    This is like multithread_LHC_builder but not multithreaded...
    We should really collapse these functions...

    Parameters
    ----------
    @comparator
        To see if hypercube A is better than hypercube B, we use the if
        statement
        if comparator(A's minimum separation, B's minimum separation)
        That is to say, if this if-condition evaluates to True, we consider A
        as better than B.
        The two obvious options are np.greater and np.less. np.greater is the
        default and should be used if you are unsure.
    """

    '''
    This is the external function, 'lhs.' It needs to know the size
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
    i = 0
    while True:
        samples_new = lhs(n=n_params, samples=n_samples,
                          criterion='center')
        dist = cdist(samples_new, samples_new, metric='euclidean')
        min_dist_new = np.amin(dist[dist > 0])
        # If the new minimum distance is larger than the one we stored,
        # then we pick this configuration and save it, broadcasting the
        # update
        if (comparator(min_dist_new, min_dist)):
            min_dist = min_dist_new
            samples = samples_new
            print("New best (" + str(min_dist_new) + ") found after", i,
                  "trials. New minimum distance is " + str(min_dist_new) + \
                  ". Saving...")

            '''
            The lhs function returns a LHC normalized within the range [0,1]
            for each dimension. We have to rescale the LHC to match the range
            we want for our cosmological parameters
            '''
            for n, (par, param_range) in enumerate(iteritems(param_ranges)):
                samples[:, n] = samples[:, n] * \
                    (param_range[1] - param_range[0]) + param_range[0]
            np.save("best_lhc_" + label + ".npy", samples,
                allow_pickle=True)
        i += 1

import concurrent.futures

def batch(n_params, n_samples=5000, comparator=np.greater):
    """
    Parameters
    ----------
    @comparator
        To see if hypercube A is better than hypercube B, we use the if
        statement
        if comparator(A's minimum separation, B's minimum separation)
        That is to say, if this if-condition evaluates to True, we consider A
        as better than B.
        The two obvious options are np.greater and np.less. np.greater is the
        default and should be used if you are unsure.
    """
    samples = lhs(n=n_params, samples=n_samples, criterion='center')
    # We want to get a good configuration for the LHC, that maximises
    # the minimum distance among points.
    dist = cdist(samples, samples, metric='euclidean')
    min_dist = np.amin(dist[dist > 0])
    # Iterating this for n_trials times
    for i in range(49):
        samples_new = lhs(n=n_params, samples=n_samples, criterion='center')
        dist = cdist(samples_new, samples_new, metric='euclidean')
        min_dist_new = np.amin(dist[dist > 0])
        # If the new minimum distance is larger than the one we stored,
        # then we pick this configuration and save it, broadcasting the
        # update
        if comparator(min_dist_new, min_dist):
            min_dist = min_dist_new
            samples = samples_new

    return samples, min_dist

def multithread_LHC_builder(param_ranges, n_samples, label="unlabeled",
    previous_record=0, comparator=np.greater):
    """
    Use more CPU to compute more random LHCs, periodically saving the one with
    the greatest minimum separation. In other words, our LHC generation
    approach is still fundamentally inefficient (we're hoping to simply get
    lucky with the minimum separations).

    As of 19.06.23 @ 11:40 am, the records are as follows:
    * COMET priors
        * massive case: 0.0802202670064637
            (saved under "best_lhc_multi2.npy")

    Parameters
    ----------
    @comparator
        To see if hypercube A is better than hypercube B, we use the if
        statement:
        if comparator(A's minimum separation, B's minimum separation)
        That is to say, if this if-condition evaluates to True, we consider A
        as better than B.
        The two obvious options are np.greater and np.less. np.greater is the
        default and should be used if you are unsure.
    """
    n_params = len(param_ranges)
    total_num_cubes = 0

    '''
    This is the external function, 'lhs.' It needs to know the size
    of the parameter space, the number of points you want, and the
    criterion to select points. 'center' means that it first divides
    each dimension of the parameter space into n_samples equispaced
    intervals and it picks the middle point of the chosen interval
    '''
    overall_min_dist = previous_record
    overall_best_lhc = None
    num_workers = 12

    while True:
        executor = concurrent.futures.ProcessPoolExecutor(num_workers)
        futures = [executor.submit(batch, n_params) \
            for i in range(num_workers)]
        min_dists = []
        lhcs = []

        for future in futures:
            this_lhc, this_min_dist = future.result()
            if comparator(this_min_dist, overall_min_dist):
                overall_min_dist = this_min_dist
                overall_best_lhc = this_lhc

                for n, (par, param_range) in \
                    enumerate(iteritems(param_ranges)):
                    overall_best_lhc[:, n] = overall_best_lhc[:, n] * \
                    (param_range[1] - param_range[0]) + param_range[0]
                np.save("best_lhc_" + label + ".npy", overall_best_lhc,
                    allow_pickle=True)

                print("New best (" + str(overall_min_dist) + \
                      ") found! Saving...")

        #outs = concurrent.futures.wait(futures)

        '''
        The lhs function returns a LHC normalized within the range [0,1]
        for each dimension. We have to rescale the LHC to match the range
        we want for our cosmological parameters
        '''

def multithread_unit_LHC_builder(dim, n_samples, label="unlabeled",
    num_workers=12, previous_record=0, comparator=np.greater,
    track_performance=False):
    """
    Use more CPU to compute more random LHCs, periodically saving the one with
    the greatest minimum separation. In other words, our LHC generation
    approach is still fundamentally inefficient (we're hoping to simply get
    lucky with the minimum separations).

    As of 19.06.23 @ 11:40 am, the records are as follows:
    * COMET priors
        * massive case: 0.0802202670064637
            (saved under "best_lhc_multi2.npy")

    Parameters
    ----------
    @comparator
        To see if hypercube A is better than hypercube B, we use the if
        statement
        if comparator(A's minimum separation, B's minimum separation)
        That is to say, if this if-condition evaluates to True, we consider A
        as better than B.
        The two obvious options are np.greater and np.less. np.greater is the
        default and should be used if you are unsure.
    """
    total_calls_counter = 0
    start_time = time.time()
    
    function_calls_record = np.array([])
    bests = np.array([])
    wall_times = np.array([])

    overall_min_dist = previous_record
    overall_best_lhc = None

    while True:
        executor = concurrent.futures.ProcessPoolExecutor(num_workers)
        futures = [executor.submit(batch, dim, n_samples) \
            for i in range(num_workers)]
        min_dists = []
        lhcs = []

        # Now collapse everything
        for future in futures:
            total_calls_counter += 50
            this_lhc, this_min_dist = future.result()
            if comparator(this_min_dist, overall_min_dist):
                overall_min_dist = this_min_dist
                overall_best_lhc = this_lhc

                np.save("best_lhc_" + label + ".npy", overall_best_lhc,
                    allow_pickle=True)

                print("New best (" + str(overall_min_dist) + \
                      ") found! Saving...")
                
                if track_performance:
                    function_calls_record = np.append(function_calls_record,
                                                      total_calls_counter)
                    bests = np.append(bests, overall_min_dist)
                    wall_times = np.append(wall_times,
                                           time.time() - start_time)
                    
                    np.save("fn_calls.npy", function_calls_record)
                    np.save("bests.npy", bests)
                    np.save("wall_times.npy", wall_times)


def nearness_comparator(target):
    """
    This function returns a Boolean comparator function based on numerical
    proximity.

    The comparator works like this: comparator(a, b) returns True if a is
    closer to target than b, otherwise False.

    #! Should we implement error checking here?
    """
    comparator = lambda a, b: np.abs(a - target) < np.abs(b - target)
    return comparator

def minimum_separation(samples):
    dist = cdist(samples, samples, metric='euclidean')
    return np.amin(dist[dist > 0])

