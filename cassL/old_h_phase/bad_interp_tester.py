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
bad_cosmo["h"] = 0.67

h = bad_cosmo["h"]

if False:
    bad_cosmo["OmB"] = bad_cosmo["ombh2"] / h ** 2
    bad_cosmo["OmC"] = bad_cosmo["omch2"] / h ** 2
    bad_cosmo["OmM"] = bad_cosmo["OmB"] + bad_cosmo["OmC"]
    bad_cosmo["OmL"] = 1 - bad_cosmo["OmM"] - bad_cosmo["OmK"]

#print(bad_cosmo)

#_redshifts = np.flip(np.linspace(0, 0.0645990428900792, 150))
_redshifts = np.flip(np.linspace(0, 20, 150))

standard_k_axis = np.load("standard_k.npy", allow_pickle=True)

good_cosmo = cp.deepcopy(ci.cosm.iloc[0])
good_cosmo = ci.specify_neutrino_mass(good_cosmo, 0, 0)

#print(good_cosmo)

PKnu = ci.kzps_interpolator(good_cosmo, redshifts=_redshifts,
    fancy_neutrinos=False,
    z_points=150, kmax=max(standard_k_axis), hubble_units=False)

print(PKnu.P(0., 0.001))
