# coding: utf-8
from cassL import generate_emu_data as ged
from cassL import user_interface as ui
import numpy as np

lhc = np.load("lhc_full_pipeline.npy")
k_axis = np.load("../k/65k_k.npy")
priors = ui.prior_file_to_array("COMET_FP")

ged.fill_hypercube(lhc, k_axis, priors=priors, save_label="fp",
                   write_period=100, eval_func=ged.interpolate_nosigma12)
