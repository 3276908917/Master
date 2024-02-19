from cassL import generate_emu_data as ged
from cassL import user_interface as ui
from cassL import camb_interface as ci
from cassL import train_emu as te
import numpy as np

PATH_BASE = ""

# Emulator version.
version = "99"
priors = ui.prior_file_to_array("MASSLESS")

def eval_func(cosmology):
    return ci.evaluate_sigma12(cosmology, [1])

train_lhs = np.load(PATH_BASE + "lhc_train_initial.npy")
test_lhs = np.load(PATH_BASE + "lhc_test_initial.npy")
    
checkpoint = 2 # which phase of this script we're on    
    
if checkpoint < 1: # skip already-computed stuff
    sigma12_train_samples, _ = ged.fill_hypercube(train_lhs, None,
        priors, eval_func, save_label = "sigma12_train")

    np.save(PATH_BASE + "samples_train.npy", sigma12_train_samples)

if checkpoint < 2:
    sigma12_test_samples, _ = ged.fill_hypercube(test_lhs, None,
        priors, eval_func, save_label = "sigma12_test")
        
    np.save(PATH_BASE + "samples_test.npy", sigma12_test_samples)

if checkpoint < 3:
    samples_train = np.load(PATH_BASE + "samples_train.npy")

    X_train, Y_train = te.eliminate_unusable_entries(train_lhs, samples_train)
    Y_train2 = Y_train.reshape((len(Y_train), 1))

    trainer = te.Emulator_Trainer("sigma12_v" + version, X_train, Y_train2,
                                  priors, False)
    trainer.train_p_emu()
    trainer.save()
    
else:
    trainer = np.load(te.path_to_emus + "sigma12_v" + version + ".cle",
                      allow_pickle=True)

if checkpoint < 4:
    samples_test = np.load(PATH_BASE + "samples_test.npy")
    X_test, Y_test = te.eliminate_unusable_entries(test_lhs, samples_test)

    Y_test2 = Y_test.reshape((len(Y_test), 1))
    trainer.test(X_test, Y_test2)
    trainer.save()

    print("step 4 complete!")

trainer.error_hist()

trainer.error_hist(metric="relative")
trainer.error_hist(metric="deltas")
