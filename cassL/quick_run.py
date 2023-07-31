import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

standard_k = np.load("current_data_files/standard_k.npy", allow_pickle=True)
hc_sigma12_pred = np.load("best_lhc_sig12sq_pred.npy", allow_pickle=True)
param_ranges = ui.get_param_ranges(massive_neutrinos=True)
#del param_ranges["sigma12"]
#param_ranges["sigma12_2"] = [0.04, 1]
samples_H2, rescalers_H2 = ged.fill_hypercube(
    hc_sigma12_pred, standard_k, param_ranges,
    massive_neutrinos=True, write_period=50, save_label="H3_unit_train"
)

