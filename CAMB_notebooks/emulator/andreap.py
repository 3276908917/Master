import camb
from scipy.integrate import quad
import numpy as np
from camb import initialpower, model

def cassL_to_andrea_cosmology(cosm, hybrid=False):
    """
    The 'hybrid' flag indicates whether we should use the original code as
        directly given us by Andrea (the function "get_PK", or if we should
        use thy hybrid function "get_PK_hybrid." As a reminder, the purpose
        of the hybrid function is to bring my code
        ("cosmology_to_PK_interpolator") and Andrea's ("get_PK") closer and
        closer together until this strange discrepancy (as chronicled on the
        "interpolator_CAMB_agreement" notebook) vanishes.
    """
    ombh2 = cosm['ombh2']
    omch2 = cosm['omch2']
    ns = cosm['n_s']
    H0 = cosm['h'] * 100
    omnuh2 = cosm['omnuh2']
    As = cosm['A_s']
    w0 = cosm['w0']
    wa = float(cosm['wa'])
    OmK = cosm['OmK']

    de_model = 'fluid'
    if w0 != -1 or wa != 0:
        de_model='ppf'

    if hybrid:
        return get_PK(ombh2, omch2, ns, omnuh2, H0, As, w0, wa, OmK, de_model,
            w_mzero=True)
    else:
        return get_PK_hybrid(ombh2, omch2, ns, omnuh2, H0, As, w0, wa, OmK,
            de_model, w_mzero=True)

def get_PK_hybrid(ombh2, omch2, ns, omnuh2, H0, As, w0=-1.0, wa=0.0, omk=0.0,
    de_model='fluid', w_mzero=True): 
    """
    The point of this function is to begin with the code "get_PK" given us by
        Andrea, and to incrementally bring it more and more in line with our
        current interpolator approach ("cosmology_to_PK_interpolator", as
        written in 
    """

    #pars = camb.CAMBparams()
    #pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, omnuh2=omnuh2,
        omk=omk)
    pars.num_nu_massive = 1
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
    pars.Transfer.kmax = 20.0

    #omnuh2 = np.copy(pars.omnuh2)
    #print (omnuh2)
    pars.NonLinear = model.NonLinear_none
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)
    
    #print (pars.num_nu_massive)
    
    PKnu = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                              hubble_units=False, k_hunit=False,
                                              kmax=20.0, zmax=20.0,
                                              var1='delta_nonu', var2='delta_nonu')
    
    #print (camb.get_results(pars))
    
    if w_mzero:
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2+omnuh2, mnu=0.0, omk=omk)
        pars.InitPower.set_params(ns=ns, As=As)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
        pars.NonLinear = model.NonLinear_none
        pars.Accuracy.AccuracyBoost = 3
        pars.Accuracy.lAccuracyBoost = 3
        pars.Accuracy.AccuratePolarization = False
        pars.Transfer.kmax = 20.0
        pars.set_matter_power(redshifts=[0.0], kmax=20.0)
        PK = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                                hubble_units=False, k_hunit=False,
                                                kmax=20.0, zmax=20.0,
                                                var1='delta_tot', var2='delta_tot')
    
    if w_mzero:
        out = {}
        out['mzero'] = PK
        out['mnu'] = PKnu
    else:
        out = PKnu
    
    return out

def get_PK(ombh2, omch2, ns, omnuh2, H0, As, w0=-1.0, wa=0.0, omk=0.0,
    de_model='fluid', w_mzero=True): 
    """
    An updated version of "get_PK_doubtful." We should remove the doubtful
        version relatively soon, but it may be helpful in identifying the
        precise source of discrepancies currently recorded by our Jupyter
        notebooks.
    """

    #pars = camb.CAMBparams()
    #pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, omnuh2=omnuh2, omk=omk)
    pars.num_nu_massive = 1
    #omnuh2 = np.copy(pars.omnuh2)
    #print (omnuh2)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
    pars.NonLinear = model.NonLinear_none
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False
    pars.Transfer.kmax = 20.0
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)
    
    #print (pars.num_nu_massive)
    
    PKnu = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                              hubble_units=False, k_hunit=False,
                                              kmax=20.0, zmax=20.0,
                                              var1='delta_nonu', var2='delta_nonu')
    
    #print (camb.get_results(pars))
    
    if w_mzero:
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2+omnuh2, mnu=0.0, omk=omk)
        pars.InitPower.set_params(ns=ns, As=As)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
        pars.NonLinear = model.NonLinear_none
        pars.Accuracy.AccuracyBoost = 3
        pars.Accuracy.lAccuracyBoost = 3
        pars.Accuracy.AccuratePolarization = False
        pars.Transfer.kmax = 20.0
        pars.set_matter_power(redshifts=[0.0], kmax=20.0)
        PK = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                                hubble_units=False, k_hunit=False,
                                                kmax=20.0, zmax=20.0,
                                                var1='delta_tot', var2='delta_tot')
    
    if w_mzero:
        out = {}
        out['mzero'] = PK
        out['mnu'] = PKnu
    else:
        out = PKnu
    
    return out


def get_PK_doubtful(ombh2, omch2, ns, mnu, H0, As, w0=-1.0, wa=0.0, omk=0.0, de_model='fluid', w_mzero=True): 
    
    h = H0 / 100
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    #pars.num_nu_massive = 1
    omnuh2 = np.copy(pars.omnuh2)
    print("Physical density in massive neutrinos", omnuh2)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
    pars.NonLinear = model.NonLinear_none
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False
    pars.Transfer.kmax = 10.0 / h
    pars.set_matter_power(redshifts=[0.0], kmax=10.0 / h)
    
    print("Number of massive neutrinos:", pars.num_nu_massive)
    print("Massive-neutrino sigma12:",
            camb.get_results(pars).get_sigmaR(12, hubble_units=False))
    PKnu = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                              hubble_units=False, k_hunit=False,
                                              kmax=10.0 / h, zmax=10.0,
                                              var1='delta_nonu', var2='delta_nonu')
    
    #print (camb.get_results(pars))
    
    if w_mzero:
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2+omnuh2,
            mnu=0.0, omk=omk)
        pars.InitPower.set_params(ns=ns, As=As)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=de_model)
        pars.NonLinear = model.NonLinear_none
        pars.Accuracy.AccuracyBoost = 3
        pars.Accuracy.lAccuracyBoost = 3
        pars.Accuracy.AccuratePolarization = False
        pars.Transfer.kmax = 10.0 / h
        pars.set_matter_power(redshifts=[0.0], kmax=10.0 / h)
        print("Massless-neutrino sigma12:",
            camb.get_results(pars).get_sigmaR(12, hubble_units=False))
        PK = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                                hubble_units=False, k_hunit=False,
                                                kmax=10.0 / h, zmax=10.0,
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
    
def get_s12_fixedz(PK):
    
    def W(x):
        return 3.0 * (np.sin(x) - x*np.cos(x)) / x**3
    def integrand(x):
        return x**2 * PK(x) * W(x*12)**2
    
    s12 = quad(integrand, 1e-4, 5)[0]
    
    return np.sqrt(s12/(2*np.pi**2))
