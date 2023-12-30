import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

standard_k = np.load("data_sets/k/300k.npy")

safe = ui.prior_file_to_array("COMET")
dangerous = ui.prior_file_to_array("CLASSIC")

priors = lambda conservatism: conservatism * safe + (1 - conservatism) * \
         dangerous
priors_at = priors(0.39)

# Modify this to change the data set you're building
file_particle = "val"

lhc = np.load("data_sets/Hnu3/lhc_" + file_particle + "_initial.npy",
                  allow_pickle=True)
samples, rescalers = ged.fill_hypercube(lhc, standard_k, priors=priors_at,
        write_period=1000, save_label="Hnu3_" + file_particle,
        crash_when_unsolvable=True)

np.save("data_sets/Hnu3/lhc_" + file_particle + "_final.npy", lhc)
np.save("data_sets/Hnu3/samples_" + file_particle + ".npy", sample)
np.save("data_sets/Hnu3/rescalers_" + file_particle + ".npy", rescalers)

