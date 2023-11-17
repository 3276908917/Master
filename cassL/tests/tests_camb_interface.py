from cassL import camb_interface as ci

m0 = ci.specify_neutrino_mass(ci.default_cosmology(), 0)

# Without these keys, a dictionary cannot qualify as a cosmology.
essential_keys = ["h", "ombh2", "omch2", "OmK", "omnuh2", "A_s", "n_s", "w0",
                  "wa"]

def test_parsing():
    """
    Let's FINALLY kill that bug that keeps wa a string...
    """
    for essential_key in essential_keys:
        if essential_key == "omnuh2": # not specified by the cosm table
            continue
        for value in ci.cosm[essential_key]:
            assert isinstance(value, float), "At least one value in the '" + \
                essential_key + "' column of the 'cosm' table is not a float!"
                
    
def test_default_cosmology():
    # First, check that it has all of the basic fields:
    default_cosmology = ci.default_cosmology()
    
    for essential_key in essential_keys:
        assert essential_key in default_cosmology, "The default cosmology " + \
            "does not specify a value for the necessary " + essential_key + \
            "field."
    
    assert "nnu_massive" in default_cosmology, "The default cosmology " + \
            "does not specify a value for the necessary nnu_massive field."

    # make sure that modifying the result of this function does not impact
    # future calls to the function
    del default_cosmology["omnuh2"]
    assert "omnuh2" in ci.default_cosmology(), "default_cosmology is not " + \
        "returning a copy of the reference model, but the reference model " + \
        "itself!"
    
    default_cosmology2 = ci.default_cosmology()
    default_cosmology2["wa"] = default_cosmology["wa"] - 1
    assert default_cosmology2["wa"] == ci.default_cosmology()["wa"], \
        "default_cosmology is not returning a copy of the reference " + \
        "model, but the reference model itself!"


def test_specify_neutrino_mass():
    m0_massless = ci.specify_neutrino_mass(m0, 0, 0)

    assert m0_massless["nnu_massive"] == 0, "There should be no massive " + \
        "species in the massless case."
    assert m0_massless["omnuh2"] == 0, "There should be no physical " + \
        "density in neutrinos in the massless case."

    m0_massive = ci.specify_neutrino_mass(m0, 0.02)

    assert m0_massive["nnu_massive"] == 1, "There should be one massive " + \
        "species in the unspecified case."
    assert m0_massive["omnuh2"] == 0.02, "The input omnuh2 does not " + \
        "match the recorded omnuh2."


def test_input_cosmology_direct_values():
    """
    Unfinished: make sure that ALL values were correctly transcribed.
    """
    m0_copy = ci.specify_neutrino_mass(m0, 0)
    pars = ci.input_cosmology(m0_copy)

    assert pars.H0 == 67, "incorrect Hubble constant"
    assert pars.ombh2 == 0.022445, "incorrect physical density in b"
    
    # Columbus_0    0.022445  0.120567  0.96  2.12723788013000E-09  0.050000000  0.268584094  0.318584094  0.000  0.681415906  0.67  -1.00  0.00    -    2.000000  1.000000  0.570000  0.300000  0.000000  1000.00000  0.82755


    # Now check to make sure mnu was computed correctly.
    raise NotImplementedError

def test_input_cosmology_indirect_vals():
    m0_copy = ci.specify_neutrino_mass(m0, 0)
    pars = ci.input_cosmology(m0_copy)

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
    """
    Unfinished: make sure that input_cosmology throws an error when ANY of the
        necessary values is missing.
    """
    for essential_key in essential_keys:
        dummy_model = ci.default_cosmology()
        del dummy_model[essential_key]
        try:
            pars = ci.input_cosmology(dummy_model)
        except Exception as err:
            assert isinstance(err, ValueError), "When the value of " + \
                essential_key + " is missing, input_cosmology crashes for " + \
                "the wrong reason."

