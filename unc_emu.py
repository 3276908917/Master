from cassL import train_emu as te
from cassL import user_interface as ui
import numpy as np

lhs_train = np.load("lhc_train_final.npy")
lhs_val = np.load("lhc_val_final.npy")
lhs_test = np.load("lhc_test_final.npy")

samples_train = np.load("samples_train.npy")
samples_val = np.load("samples_val.npy")
samples_test = np.load("samples_test.npy")

priors = ui.prior_file_to_array("COMET")

X_train, Y_train = te.eliminate_unusable_entries(lhs_train, samples_train)
X_val, Y_val = te.eliminate_unusable_entries(lhs_val, samples_val)
X_test, Y_test = te.eliminate_unusable_entries(lhs_test, samples_test)

if False: # False = skip main emulator training (if we've already completed it).
    trainer = te.Emulator_Trainer("Hnu2unc", X_train, Y_train, priors, True)
    trainer.train_p_emu()
    trainer.save()
    print("Step 1 complete!")
else:
    trainer = np.load(te.path_to_emus + "Hnu2unc.cle", allow_pickle=True)

if True:
    trainer.validate(X_val, Y_val)
    trainer.save()
    print("Step 2 complete!")

if True:
    trainer.test(X_test, Y_test)
    trainer.save()
    print("Step 3 complete!")

trainer.error_hist()

trainer.error_hist(metric="val_percent")
trainer.error_hist(metric="val_deltas")
trainer.error_hist(metric="unc_percent")
trainer.error_hist(metric="unc_deltas")
