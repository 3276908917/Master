import numpy as np
import camb_interface as ci
import copy as cp
import generate_training_data

bad_cosmo = cp.deepcopy(ci.cosm.iloc[0])
bad_cosmo["ombh2"] = 0.022148
bad_cosmo["omch2"] = 0.112054
bad_cosmo["n_s"] = 0.840494
bad_cosmo["sigma12"] = 0.83016
bad_cosmo["nnu_massive"] = 1
bad_cosmo["mnu"] = 0.146852
bad_cosmo["A_s"] = 1.5203351579894607e-09
bad_cosmo["h"] = 0.01

h = bad_cosmo["h"]

bad_cosmo["OmB"] = bad_cosmo["ombh2"] / h ** 2
bad_cosmo["OmC"] = bad_cosmo["omch2"] / h ** 2
bad_cosmo["OmM"] = bad_cosmo["OmB"] + bad_cosmo["OmC"]
bad_cosmo["OmL"] = 1 - bad_cosmo["OmM"] - bad_cosmo["OmK"]

print(bad_cosmo)

_, _, _, list_sigma12 = ci.kzps(bad_cosmo, np.flip(np.linspace(0, 1100, 150)),
    fancy_neutrinos=False, k_points=generate_training_data.NPOINTS)
