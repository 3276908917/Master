import camb
from scipy.integrate import quad
import numpy as np
from camb import initialpower, model

def get_PK(ombh2, omch2, ns, mnu, H0, As, w0=-1.0, wa=0.0, omk=0.0, de_model='fluid', w_mzero=True): 
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    #pars.num_nu_massive = 1
    omnuh2 = np.copy(pars.omnuh2)
    print (omnuh2)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
    pars.NonLinear = model.NonLinear_none
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False
    pars.Transfer.kmax = 10.0
    pars.set_matter_power(redshifts=[0.0], kmax=100.0)
    
    print (pars.num_nu_massive)
    
    PKnu = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                              hubble_units=False, k_hunit=False,
                                              kmax=100.0, zmax=20.0,
                                              var1='delta_nonu', var2='delta_nonu')
    
    #print (camb.get_results(pars))
    
    if w_mzero:
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
        pars.InitPower.set_params(ns=ns, As=As)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
        pars.NonLinear = model.NonLinear_none
        pars.Accuracy.AccuracyBoost = 3
        pars.Accuracy.lAccuracyBoost = 3
        pars.Accuracy.AccuratePolarization = False
        pars.Transfer.kmax = 100.0
        pars.set_matter_power(redshifts=[0.0], kmax=100.0)
        PK = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                                hubble_units=False, k_hunit=False,
                                                kmax=100.0, zmax=20.0,
                                                var1='delta_tot', var2='delta_tot')
    
    if w_mzero:
        out = {}
        out['mzero'] = PK
        out['mnu'] = PKnu
    else:
        out = PKnu
    
    return out


def get_s12(PK, z):
    
    def W(x):
        return 3.0 * (np.sin(x) - x*np.cos(x)) / x**3
    def integrand(x):
        return x**2 * PK.P(z,x) * W(x*12)**2
    
    s12 = quad(integrand, 1e-4, 5)[0]
    
    return np.sqrt(s12/(2*np.pi**2))
    
def get_s12_fixedz(PK, z):
    
    def W(x):
        return 3.0 * (np.sin(x) - x*np.cos(x)) / x**3
    def integrand(x):
        return x**2 * PK(x) * W(x*12)**2
    
    s12 = quad(integrand, 1e-4, 5)[0]
    
    return np.sqrt(s12/(2*np.pi**2))
