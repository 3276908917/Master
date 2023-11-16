from cassL import generate_emu_data as ged
from cassL import user_interface as ui
from cassL import camb_interface as ci
from cassL import train_emu as te
import numpy as np

priors = ui.prior_file_to_array("COMET")

def eval_func(cosmology):
    return ci.evaluate_sigma12(cosmology, [1])
    
if False: # skip already-computed stuff
    sigma12_train_lhs = np.load("lhc_sigma12_train_initial.npy")

    sigma12_train_samples, _ = ged.fill_hypercube(sigma12_train_lhs, None, priors,
        eval_func, save_label = "sigma12_train")

    np.save("sigma12_samples_train.npy", sigma12_train_samples)
    np.save("lhc_sigma12_train_final.npy", sigma12_train_lhs)

sigma12_test_lhs = np.load("lhc_sigma12_test_initial.npy")

sigma12_test_samples, _ = ged.fill_hypercube(sigma12_test_lhs, None, priors,
    eval_func, save_label = "sigma12_test")
    
np.save("sigma12_samples_test.npy", sigma12_test_samples)
np.save("lhc_sigma12_test_final.npy", sigma12_test_lhs)

lhs_train = np.load("lhc_sigma12_train.npy")
lhs_test = np.load("lhc_test_final.npy")

samples_train = np.load("samples_train.npy")
samples_test = np.load("samples_test.npy")

priors = ui.prior_file_to_array("COMET")

X_train, Y_train = te.eliminate_unusable_entries(lhs_train, samples_train)
X_test, Y_test = te.eliminate_unusable_entries(lhs_test, samples_test)

trainer = te.Emulator_Trainer("sigma12_v1", X_train, Y_train, priors)
trainer.train_p_emu()
trainer.save()

trainer.test(X_test, Y_test)
trainer.save()

trainer.error_hist()

trainer.error_hist(metric="delta_relative")
trainer.error_hist(metric="delta_deltas")
