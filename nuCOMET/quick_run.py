import numpy as np
import generate_training_data
standard_k = np.load("standard_k.npy", allow_pickle=True)
hc_demo3 = np.load("hc_demo3.npy", allow_pickle=True)
samples_demo3 = generate_training_data.fill_hypercube(
    hc_demo3, standard_k, write_period=50
)

