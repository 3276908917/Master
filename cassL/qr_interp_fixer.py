import numpy as np
from cassL import camb_interface as ci
import matplotlib.pyplot as plt

redshift = 0.

lil_k = np.load("data_sets/k/300k.npy")
m0 = ci.default_cosmology()
intrp = ci.cosmology_to_PK_interpolator(m0, redshifts=[redshift],#np.linspace(0, 20, 100),
                                        kmax=7, hubble_units=False)

kd, zd, pd, sd = ci.evaluate_cosmology(m0, redshifts=[redshift], k_points=300)

plt.loglog(kd, pd)
plt.loglog(kd, intrp.P(redshift, kd), linestyle="dashed")
plt.show()
