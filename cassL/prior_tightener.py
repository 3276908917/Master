import numpy as np

from cassL import generate_emu_data as ged
from cassL import user_interface as ui

standard_k = np.load("data_sets/k/300k.npy", allow_pickle=True)

hc = np.load("lhc_prior_tightener.npy", allow_pickle=True)

safe = ui.prior_file_to_array("COMET")
dangerous = ui.prior_file_to_array("CLASSIC")

conservatism = 0

for i in range(100):
    conservatism += 0.01
    priors = conservatism * safe + (1 - conservatism) * dangerous

    try:
        _, _ = ged.fill_hypercube(hc, standard_k, priors=priors,
                crash_when_unsolvable=True)
    except ValueError:
        print("Conservatism level " + str(conservatism) + " has failed.")
        print("Increasing...")
        continue

    break

print("Script has completed.")

