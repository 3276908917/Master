from cassL import generate_emu_data as ged
import numpy as np
from cassL import user_interface as ui
import matplotlib.pyplot as plt

lil_k = np.load("data_sets/k/300k2.npy")
big_k = np.load("data_sets/k/65k_k.npy")
priors = ui.prior_file_to_array("COMET_PLUS")

def bundle(row):
    denormalized = ged.denormalize_row(row, priors)
    return ged.build_cosmology(denormalized)

lhc_train = np.load("data_sets/Hnu4c/lhc_train_final.npy")

p_good_intrp, _, _ = ged.interpolate_cell(bundle(lhc_train[0]), lil_k)
p_good, _, _ = ged.direct_eval_cell(bundle(lhc_train[0]), lil_k)

plt.loglog(lil_k, p_good)
# Why do we need to de-nest here?...
plt.loglog(lil_k, p_good_intrp[0], linestyle="dashed")
plt.show()
