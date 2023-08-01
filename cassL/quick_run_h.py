import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

massive_neutrinos=False

standard_k = np.load("current_data_files/standard_k.npy", allow_pickle=True)
hc_sigma12_pred = np.load("../best_lhc_unit_G8_massless.npy", allow_pickle=True)
param_ranges = ui.get_param_ranges(massive_neutrinos=massive_neutrinos)
# del param_ranges["sigma12"]
# param_ranges["sigma12_2"] = [0.04, 1]
samples_H2, rescalers_H2 = ged.fill_hypercube(
    hc_sigma12_pred, standard_k, param_ranges,
    massive_neutrinos=massive_neutrinos, write_period=50,
    save_label="h1_unit_train"
)

