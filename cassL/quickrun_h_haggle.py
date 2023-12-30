import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

# If a particular range of cells is already known to cause problems, it may be
# more efficient to change this value away from None.
special_cell_range = range(593, 594)

standard_k = np.load("data_sets/k/300k.npy", allow_pickle=True)
hc = np.load("data_sets/Hnu3/lhc_val_initial.npy", allow_pickle=True)

safe = ui.prior_file_to_array("COMET")
dangerous = ui.prior_file_to_array("CLASSIC")

priors = lambda conservatism: conservatism * safe + (1 - conservatism) * \
         dangerous
priors_at = priors(0.45)

samples, _ = ged.fill_hypercube(hc, standard_k, priors=priors_at,
    crash_when_unsolvable=False, cell_range=special_cell_range)
