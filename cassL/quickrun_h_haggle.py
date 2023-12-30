import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

standard_k = np.load("data_sets/k/300k.npy")
lhc_val = np.load("data_sets/Hnu3/lhc_val_initial.npy")

safe = ui.prior_file_to_array("COMET")
dangerous = ui.prior_file_to_array("CLASSIC")

priors = lambda conservatism: conservatism * safe + (1 - conservatism) * \
         dangerous
priors_at = priors(0.6)

samples_val, rescalers_val = ged.fill_hypercube(lhc_val, standard_k, priors_at,
        write_period=1000, save_label="Hnu3_val", crash_when_unsolvable=True,
        cell_range=range(3248, 5000))
