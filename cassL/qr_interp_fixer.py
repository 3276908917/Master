import numpy as np
from cassL import camb_interface as ci
import matplotlib.pyplot as plt

redshift = 0.

lil_k = np.load("data_sets/k/300k.npy")
m0 = ci.default_cosmology()

if False: # This is an example of a real model we would have to evaluate
    m0["ombh2"] = 2.112068e-02
    m0["omch2"] = 1.892713e-01
    m0["n_s"] = 9.757690e-01
    m0["sigma12"] = 6.981600e-01
    m0["A_s"] = 4.373864e-09
    m0["omnuh2"] = 6.219000e-03

intrp = ci.cosmology_to_PK_interpolator(m0, redshifts=[redshift],#np.linspace(0, 20, 100),
                                        kmax=7, hubble_units=False)

kd, zd, pd, sd = ci.evaluate_cosmology(m0, redshifts=[redshift], k_points=300)

plt.loglog(kd, pd)
plt.loglog(kd, intrp.P(redshift, kd), linestyle="dashed")
plt.show()

# Testing stages
# $ wrapper on model0 is fine
# ! ged is not fine
# $ is wrapper fine on a realistic model?
# * is wrapper fine at a realistic redshift?
