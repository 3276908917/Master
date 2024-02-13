"""
Module for generating and dealing with
latin hypercubes.
"""

import numpy as np
from pyDOE2 import lhs
from scipy.spatial.distance import cdist
import concurrent.futures

import time


def batch(n_params, n_samples=5000, batch_size=50, comparator=np.greater):
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
    for i in range(batch_size - 1):
        print(i + 1)
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


def multithread_unit_LHC_builder(dim, n_samples, label="unlabeled",
    batch_size=50, num_workers=12, previous_record=0, comparator=np.greater,
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
        futures = [executor.submit(batch, dim, n_samples, batch_size, \
            comparator) for i in range(num_workers)]
        min_dists = []
        lhcs = []

        # Now collapse everything
        for future in futures:
            total_calls_counter += batch_size
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

