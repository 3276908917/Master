import numpy as np
from cassL import camb_interface as ci
import matplotlib.pyplot as plt

lil_k = np.load("data_sets/k/300k.npy")
m0 = ci.default_cosmology()
intrp = ci.cosmology_to_PK_interpolator(m0, redshifts=[0],#np.linspace(0, 20, 100),
                                        kmax=7)

kd, zd, pd, sd = ci.evaluate_cosmology(m0, k_points=300)

plt.loglog(kd, pd)
plt.loglog(kd, intrp.P(0, kd), linestyle="dashed")
plt.show()