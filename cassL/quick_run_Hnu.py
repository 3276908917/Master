import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

massive_neutrinos=True

emu_stage = "train"
# Technically, this variable is not very user-friendly anymore, because we often
# use hypercubes from older versions...
emu_vlabel = "Hnu2direct"
hc_vlabel = "10k_massive"

file_suffix = emu_vlabel + "_" + emu_stage

standard_k = np.load("current_data_files/standard_k.npy", allow_pickle=True)
hc = np.load("../best_lhc_unit_" + hc_vlabel + "_" + emu_stage + ".npy",
    allow_pickle=True)
param_ranges = ui.get_param_ranges(priors="COMET",
    massive_neutrinos=massive_neutrinos)
# del param_ranges["sigma12"]
#param_ranges["sigma12_root"] = [0.4472135954999579, 1]

completed_index = 3999
samples = None

if completed_index is not None:
    file_ending = str(completed_index) + "_unit_" + file_suffix + ".npy"
    samples = np.load("samples_backup_i" + file_ending, allow_pickle=True)
    hc = np.load("hc_backup_i" + file_ending, allow_pickle=True)

samples, rescalers = ged.fill_hypercube(
    hc, standard_k, param_ranges, samples=samples,
    massive_neutrinos=massive_neutrinos, write_period=500,
    save_label="unit_" + file_suffix
)
np.save("samples_unit_" + file_suffix + ".npy", samples, allow_pickle=True)
np.save("rescalers_unit_" + file_suffix + ".npy", rescalers, allow_pickle=True)
np.save("lhc_unit_" + file_suffix + ".npy", hc, allow_pickle=True)

