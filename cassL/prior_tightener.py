import numpy as np

from cassL import generate_emu_data as ged
from cassL import user_interface as ui

import copy as cp

# If a particular range of cells is already known to cause problems, it may be
# more efficient to change this value away from None.
special_cell_range = range(3247, 3250)

standard_k = np.load("data_sets/k/300k.npy", allow_pickle=True)

#hc = np.load("lhc_prior_tightener.npy", allow_pickle=True)
hc = np.load("data_sets/Hnu3/lhc_val_initial.npy", allow_pickle=True)

safe = ui.prior_file_to_array("COMET")
dangerous = ui.prior_file_to_array("CLASSIC")

# implement a binary search approach
highest_failed_conservatism = 0.0
lowest_successful_conservatism = 1.0

priors = lambda c: c * safe + (1 - c) * dangerous

# Keep testing values of conservatism until the discrepancy reaches the percent
# level.
while lowest_successful_conservatism - highest_failed_conservatism > 0.01:
    average_bound = (highest_failed_conservatism + \
                     lowest_successful_conservatism) / 2
    conservatism = np.around(average_bound, 2)

    if conservatism == highest_failed_conservatism or \
        conservatism == lowest_successful_conservatism:
        raise ValueError("For some reason, the loop is not adjusting the " + \
                          "conservatism bounds correctly, and we cannot " + \
                          "proceed.")

    priors_at = priors(conservatism)

    try:
        # We need to use a deep copy, because this function mutates the
        # sigma12 entry of the HC!!
        _, _ = ged.fill_hypercube(cp.deepcopy(hc), standard_k,
            priors=priors_at, crash_when_unsolvable=True,
            cell_range=special_cell_range)
    except ValueError:
        print("Conservatism level " + str(conservatism) + " has failed.")
        highest_failed_conservatism = conservatism
        print("Increasing...")
        continue

    print("Conservatism level " + str(conservatism) + " has succeeded.")
    lowest_successful_conservatism = conservatism
    print("Decreasing...")

print("Script has completed.")

