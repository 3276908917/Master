from cassL import user_interface as ui
dd = ui.get_data_dict("Hnu2_5k_knockoff", "COMET_with_nu")
emu_trainer = ui.build_and_test_emulator(dd)
