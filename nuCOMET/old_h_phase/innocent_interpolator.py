import numpy as np
import camb
import camb_interface as ci
import copy as cp
from camb import model

m0 = cp.deepcopy(ci.cosm.iloc[0])

my_way = True

pars = None
if my_way:
    m0 = ci.specify_neutrino_mass(m0, 0, 0)
    pars = ci.input_cosmology(m0)
else:
    pars = camb.set_params(H0=m0["h"] * 100, ombh2=m0["ombh2"],
        omch2=m0["omch2"], omnuh2=0, omk=m0["OmK"])

    pars.InitPower.set_params(ns=m0["n_s"], As=m0["A_s"])

    de_model = "ppf"
    m0["wa"] = float(m0["wa"])
    if m0["w0"] == -1 and m0["wa"] == 0:
        de_model = "fluid"

    pars.set_dark_energy(w=m0["w0"], wa=m0["wa"], dark_energy_model=de_model)

if my_way:
    ci.apply_universal_output_settings(pars)
else:
    pars.NonLinear = model.NonLinear_none
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False
    pars.Transfer.kmax = 20

_redshifts = np.flip(np.linspace(0, 20, 150))

pars.set_matter_power(redshifts=_redshifts, kmax=10.0, nonlinear=False)

print(pars)

PKnu = camb.get_matter_power_interpolator(pars, zmin=min(_redshifts),
    zmax=max(_redshifts), k_hunit=False, kmax=10.0, nonlinear=False,
    var1='delta_nonu', var2='delta_nonu', hubble_units=False)

print(PKnu.P(0., 0.001))
