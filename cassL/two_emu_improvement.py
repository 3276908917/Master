# The point of this script is to iterate through all scenario files, load that
# emulator, regenerate the accuracies, then store them to files.

from cassL import user_interface as ui
from cassL import train_emu as te
from cassL import utils
from cassL import camb_interface as ci
import matplotlib.pyplot as plt
import os
import numpy as np

import copy as cp

def acquire_colors(X_test):
    sigma12_vals = X_test[:, 3]
    normalized_vals = utils.normalize(sigma12_vals)
    return plt.cm.plasma(sigma12_vals)

ignore = ["template", "Hnu2MEGA", "Hnu2CLASSIC",
    "Hnu2direct", "Hnu2_100scales", "Hnu2_200scales",
    "Hnu2_3000samples", "Hnu2_4000samples", "Hnu2_400scales",
    "Hnu2_500scales"]

    
data_dict = ui.get_data_dict("Hz1direct")
X_test = np.zeros((5000, 6))

A_s = ci.cosm.iloc[0]["A_s"]

X_test[:, 0:4] = data_dict["X_test"]
X_test[:, 4] = A_s
X_test[:, 5] = 0

print(X_test[0:4])

emu_file = "emulators/Hnu2direct.cle"
trainer = te.Emulator_Trainer(emu_file)
trainer.test(X_test, data_dict["Y_test"])

deltas = trainer.get_errors("deltas")
percents = trainer.get_errors("percent")
colors = acquire_colors(X_test)

np.save("thesis_deltas/two_emu.npy", deltas)
np.save("thesis_percents/two_emu.npy", percents)
np.save("thesis_colors/two_emu.npy", colors)
    
try:
    print("Done!")
except Exception:
    print("There is something wrong with this scenario file!")
    
    norm = mpl.colors.Normalize(vmin=0.2, vmax=1.0)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm))