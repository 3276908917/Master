import numpy as np
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

standard_k = np.load("../k/65k_k.npy")

priors = ui.prior_file_to_array("MASSLESS")

# Modify this to change the data set you're building
file_particle = "test"

lhc = np.load("lhc_" + file_particle + "_initial.npy",
                  allow_pickle=True)
samples, rescalers = ged.fill_hypercube(lhc, standard_k, priors=priors,
        write_period=1000, save_label="Hz3_" + file_particle,
        crash_when_unsolvable=True)

np.save("../Hz3/lhc_" + file_particle + "_final.npy", lhc)
np.save("../Hz3/samples_" + file_particle + ".npy", samples)
np.save("../Hz3/rescalers_" + file_particle + ".npy", rescalers)

