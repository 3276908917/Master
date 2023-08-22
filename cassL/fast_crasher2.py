from cassL import train_emu as te
from cassL import user_interface as ui

data_dict = ui.get_data_dict("Hnu2_5k_knockoff", "COMET_with_nu")
trainer = te.Emulator_Trainer("emulators/Hnu2_5k_knockoff.cle")

X_test_clean, Y_test_clean = \
    te.eliminate_unusable_entries(data_dict["X_test"], data_dict["Y_test"])

trainer.test(X_test_clean, Y_test_clean)
