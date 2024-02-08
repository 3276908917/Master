import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

massive_neutrinos=False

emu_stage = "test"
# Technically, this variable is not very user-friendly anymore, because we often
# use hypercubes from older versions...
emu_vlabel = "Hz1"

file_suffix = emu_vlabel + "_" + emu_stage

standard_k = np.load("current_data_files/standard_k.npy", allow_pickle=True)
hc = np.load("../best_lhc_unit_" + file_suffix + ".npy",
    allow_pickle=True)
param_ranges = ui.get_param_ranges(massive_neutrinos=massive_neutrinos)
# del param_ranges["sigma12"]
# param_ranges["sigma12_2"] = [0.04, 1]
samples, rescalers = ged.fill_hypercube(
    hc, standard_k, param_ranges,
    massive_neutrinos=massive_neutrinos, write_period=100,
    save_label="unit_" + file_suffix
)
np.save("samples_unit_" + file_suffix + ".npy", samples, allow_pickle=True)
np.save("rescalers_unit_" + file_suffix + ".npy", rescalers, allow_pickle=True)
np.save("lhc_unit_" + file_suffix + ".npy", hc, allow_pickle=True)

