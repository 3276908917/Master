#constants
ombh2 = 0.022445
n_s = 0.96

#baselines
sig8_b = 0.82755
# Lbox_b = 1000 # I don't think CAMB has this
wa_b = 0.00
w0_b = -1.00
OmK_b = 0
OmL_b = 0.681415906
omch2_b = 0.120567
mnu_b = 0.00
h_b = 0.67
# derived quantities
H0_b = 100 * h_b

omega_nu = [0.0006356, 0.002, 0.006356]

baseline = {
    "H0": H0_b,
    "ombh2": ombh2,
    "omch2": omch2_b,
    "omk": OmK_b,
    "tau": 0.06, # not specified by provided data table
    "w": w0_b,
    "wa": wa_b,
    "ns": 0.96
}
As = [
    2.12723788013000E-09,
    1.78568440085517E-09,
    2.48485942677850E-09,
    2.32071013846548E-09,
    1.99553701204688E-09,
    2.07077004294502E-09,
    2.20196413682945E-09,
    1.92961581654148E-09,
    2.29291725000000E-09
]

cosm = [] # cosmologies
# This order ensures that the indices are the same
# before and after applying CAMB.

for i in range(9):
    cosm.append(baseline.copy())
    cosm[i]["As"] = As[i]
    
cosm[1]["H0"] = 55
cosm[2]["H0"] = 79

