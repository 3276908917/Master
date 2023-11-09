# The point of this script is to iterate through all scenario files, load that
# emulator, regenerate the accuracies, then store them to files.

from cassL import user_interface as ui
from cassL import train_emu as te
from cassL import utils
import matplotlib.pyplot as plt
import os
import numpy as np

ignore = ["template", "Hnu2MEGA", "Hnu2CLASSIC",
    "Hnu2direct", "Hnu2_100scales", "Hnu2_200scales",
    "Hnu2_3000samples", "Hnu2_4000samples", "Hnu2_400scales",
    "Hnu2_500scales"]

def acquire_colors(X_test):
    sigma12_vals = X_test[:, 3]
    normalized_vals = utils.normalize(sigma12_vals)
    return plt.cm.plasma(sigma12_vals)

for file in os.listdir("scenarios"):
    # The name, as user_interface expects, lacks the file ending
    scenario_name = file[:-4]
    
    if scenario_name in ignore:
        continue
    
    print(scenario_name)
    
    data_dict = ui.get_data_dict(scenario_name)
    emu_file = "emulators/" + scenario_name + ".cle"
    trainer = te.Emulator_Trainer(emu_file)
    trainer.test(data_dict["X_test"], data_dict["Y_test"])
    
    deltas = trainer.get_errors("deltas")
    percents = trainer.get_errors("percent")
    colors = acquire_colors(data_dict["X_test"])
    
    np.save("thesis_deltas/" + scenario_name + ".npy", deltas)
    np.save("thesis_percents/" + scenario_name + ".npy", percents)
    np.save("thesis_colors/" + scenario_name + ".npy", colors)
    
try:
    print("Done!")
except Exception:
    print("There is something wrong with this scenario file!")
    
    norm = mpl.colors.Normalize(vmin=0.2, vmax=1.0)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm))