# coding: utf-8
from cassL import generate_emu_data as ged
from cassL import user_interface as ui
import numpy as np

lhc = np.load("lhc_fp_massive.npy")
k_axis = np.load("../k/65k_k.npy")
priors = ui.prior_file_to_array("COMET_PLUS_FP")

ged.fill_hypercube(lhc[:10000], k_axis, priors=priors, save_label="fp",
                   write_period=100, eval_func=ged.interpolate_nosigma12)
