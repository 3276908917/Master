import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

hc = np.load("best_lhc_Hz1_val.npy", allow_pickle=True)
priors = ui.prior_file_to_array("COMET")
standard_k_axis = np.load("data_sets/k/300k.npy", allow_pickle=True)

samples, rescalers = ged.fill_hypercube(hc, standard_k_axis, priors,
	save_label="Hz1_val")