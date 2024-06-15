import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

standard_k = np.load("../k/65k_k.npy")

priors_at = ui.prior_file_to_array("COMET_PLUS_2PERCENT")

# Modify this to change the data set you're building
file_particle = "test"

lhc = np.load("lhc_" + file_particle + "_initial.npy",
                  allow_pickle=True)
samples, rescalers = ged.fill_hypercube(lhc, standard_k, priors=priors_at,
        write_period=1000, eval_func=ged.interpolate_cell,
        save_label="2p_Hnu4c_" + file_particle, crash_when_unsolvable=True)

np.save("2p_lhc_" + file_particle + "_final.npy", lhc)
np.save("2p_samples_" + file_particle + ".npy", samples)
np.save("2p_rescalers_" + file_particle + ".npy", rescalers)

