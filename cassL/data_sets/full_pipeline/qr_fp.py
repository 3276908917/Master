# coding: utf-8
from cassL import generate_emu_data as ged
from cassL import user_interface as ui
from cassL import camb_interface as ci

import numpy as np

lhc = np.load("lhc_massive_fp.npy")
k_axis = np.load("../k/65k_k.npy")
priors = ui.prior_file_to_array("COMET_PLUS_2PERCENT")

def model_0_eval_func(input_cosmology, standard_k_axis):
    alt_cosm = ci.default_cosmology()
    alt_cosm["ombh2"] = input_cosmology["ombh2"]
    alt_cosm["omch2"] = input_cosmology["omch2"]
    alt_cosm["n_s"] = input_cosmology["n_s"]
    alt_cosm["A_s"] = input_cosmology["A_s"]
    alt_cosm["z"] = 0
    alt_cosm = ci.specify_neutrino_mass(alt_cosm, input_cosmology["omnuh2"], 1)
    return ged.interpolate_nosigma12(alt_cosm, standard_k_axis)

samples, rescalers = ged.fill_hypercube_with_Pk(lhc, k_axis, priors=priors,
                                        save_label="fp_mass2p",
                                        write_period=100,
                                        eval_func=ged.interpolate_nosigma12)
