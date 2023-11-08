from cassL import camb_interface as ci
import copy as cp

def test_specify_neutrino_mass():
    m0_massless = ci.specify_neutrino_mass(ci.cosm.iloc[0], 0, 0)

    assert m0_massless["nnu_massive"] == 0, "There should be no massive " + \
        "species in the massless case."
    assert m0_massless["omnuh2"] == 0, "There should be no physical " + \
        "density in neutrinos in the massless case."
    #assert m0_massless["mnu"] == 0, "The neutrino masses should sum to " + \
    #    "zero in the massless case."

    m0_massive = ci.specify_neutrino_mass(ci.cosm.iloc[0], 0.02)

    assert m0_massive["nnu_massive"] == 1, "There should be one massive " + \
        "species in the unspecified case."
    assert m0_massive["omnuh2"] == 0.02, "The input omnuh2 does not " + \
        "match the recorded omnuh2."
    #assert m0_massive["mnu"] == 0, "The sum of the neutrino masses was " + \
    #    "incorrectly computed."

    #m0_triMassive = ci.specify

def test_input_cosmology_direct_values():
    m0 = ci.specify_neutrino_mass(ci.cosm.iloc[0], 0)
    pars = ci.input_cosmology(m0)

    assert pars.H0 == 67, "incorrect Hubble constant"
    assert pars.ombh2 == 0.022445, "incorrect physical density in b"

    # Now check to make sure mnu was computed correctly.

def test_input_cosmology_indirect_vals():
    m0 = ci.specify_neutrino_mass(ci.cosm.iloc[0], 0)
    pars = ci.input_cosmology(m0)

    try:
        pars.Omb
        pars.Omc
        assert False, "fractional densities should not be calculated."
    except Exception:
        pass

    try:
        pars.mnu
        assert False, "the sum of neutrino masses should not be calculated."
    except Exception:
        pass

def test_input_cosmology_err_handling():
    mA = ci.specify_neutrino_mass(ci.cosm.iloc[0], 0)
    del mA["h"]
    try:
        pars = ci.input_cosmology(mA)
    except Exception as err:
        assert isinstance(err, ValueError), "When the value of h is " + \
            "missing, input_cosmology crashes for the wrong reason."


    # Columbus_0    0.022445  0.120567  0.96  2.12723788013000E-09  0.050000000  0.268584094  0.318584094  0.000  0.681415906  0.67  -1.00  0.00    -    2.000000  1.000000  0.570000  0.300000  0.000000  1000.00000  0.82755
