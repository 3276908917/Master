import numpy as np
standard_k = np.load("data_sets/k/300k.npy")
emu = np.load("emulators/Hnu2unc.cle", allow_pickle=True)
emu.set_scales(standard_k)
emu.error_curves(param_index=-2, param_label="max. extr.")
