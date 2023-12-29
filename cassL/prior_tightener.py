import numpy as np

from cassL import generate_emu_data as ged
from cassL import user_interface as ui

standard_k = np.load("data_sets/k/300k.npy", allow_pickle=True)

hc = np.load("lhc_prior_tightener.npy", allow_pickle=True)

safe = ui.prior_file_to_array("COMET")
dangerous = ui.prior_file_to_array("CLASSIC")

conservatism = 0.5

# implement a binary search approach

highest_failed_conservatism = 0.0
lowest_successful_conservatism = 1.0

# Keep testing values of conservatism until the discrepancy reaches percent
while lowest_successful_conservatism - highest_failed_conservatism != 0.01:
	average_bound = (highest_failed_conservatism + \
                     lowest_successful_conservatism) / 2
	conservatism = np.around(average_bound, 2)

    if conservatism == highest_failed_conservatism or \
        conservatism == lowest_successful_conservatism:
        raise ValueError("For some reason, the loop is not adjusting the " + \
                          "conservatism bounds correctly, and we cannot " + \
                          "proceed.")

    priors = conservatism * safe + (1 - conservatism) * dangerous

    try:
        _, _ = ged.fill_hypercube(hc, standard_k, priors=priors,
                crash_when_unsolvable=True)
    except ValueError:
        print("Conservatism level " + str(conservatism) + " has failed.")
        highest_failed_conservatism = conservatism
        print("Increasing...")
        continue

    print("Conservatism level " + str(conservatism) + " has succeeded.")
    lowest_successful_conservatism = conservatism
    print("Decreasing...")

print("Script has completed.")

