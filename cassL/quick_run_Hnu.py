import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

massive_neutrinos=True

emu_stage = "train"
# Technically, this variable is not very user-friendly anymore, because we often
# use hypercubes from older versions...
emu_vlabel = "Hnu2_5k_100scales"
hc_vlabel = "5k_massive"
fending = ".npy"

file_suffix = emu_vlabel + "_" + emu_stage + fending

standard_k = np.load("data_sets/k/100k.npy", allow_pickle=True)
hc = np.load("../best_lhc_unit_" + hc_vlabel + "_" + emu_stage + fending,
    allow_pickle=True)
param_ranges = ui.prior_file_to_dict(prior_name="COMET_with_nu")
# del param_ranges["sigma12"]
#param_ranges["sigma12_root"] = [0.4472135954999579, 1]

completed_index = -1
samples = None

if completed_index > -1:
    try:
        file_ending = str(completed_index) + "_unit_" + file_suffix
        samples = np.load("samples_backup_i" + file_ending, allow_pickle=True)
        hc = np.load("hc_backup_i" + file_ending, allow_pickle=True)
    except FileNotFoundError:
        # Probably what happened is that we just didn't reach the end of the
        # first write interval. So, we have to start from the beginning.
        raise UserWarning("The given completed_index did not match any files!")
        completed_index = -1

samples, rescalers = ged.fill_hypercube(
    hc, standard_k, param_ranges, samples=samples,
    eval_func=ged.direct_eval_cell,
    massive_neutrinos=massive_neutrinos, write_period=500,
    cell_range=range(completed_index + 1, len(hc)),
    save_label="unit_" + file_suffix
)
np.save("samples_unit_" + file_suffix, samples, allow_pickle=True)
np.save("rescalers_unit_" + file_suffix, rescalers, allow_pickle=True)
np.save("lhc_unit_" + file_suffix, hc, allow_pickle=True)

### Wouldn't it be cool if this thing automatically cleaned up backups that
### aren't needed anymore??
### Like, we could maintain a last_backup_handle label. Every time we complete a
### new backup, we delete the file located at last_backup_handle and update the
### label to the new backup.
