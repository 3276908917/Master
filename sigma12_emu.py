from cassL import generate_emu_data as ged
from cassL import user_interface as ui
from cassL import camb_interface as ci
import numpy as np

priors = ui.prior_file_to_array("COMET")

def eval_func(cosmology):
    return ci.evaluate_sigma12(cosmology, [1])

sigma12_train_lhs = np.load("lhc_sigma12_train.npy")

sigma12_train_samples, _ = ged.fill_hypercube(sigma12_train_lhs, None, priors,
    eval_func, save_label = "sigma12_train")

np.save("sigma12_samples_train.npy", sigma12_train_samples)

sigma12_test_lhs = np.load("lhc_sigma12_test.npy")

sigma12_test_samples, _ = ged.fill_hypercube(sigma12_test_lhs, None, priors,
    eval_func, save_label = "sigma12_test")
    
np.save("sigma12_samples_test.npy", sigma12_test_samples)
    
