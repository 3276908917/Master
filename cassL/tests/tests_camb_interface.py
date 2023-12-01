from cassL import camb_interface as ci

# Without these keys, a dictionary cannot qualify as a cosmology.
essential_keys = ["h", "ombh2", "omch2", "OmK", "omnuh2", "A_s", "n_s", "w0",
                  "wa"]

def test_parsing():
    """
    This test confirms that all entries in the cosmologies table, which need
    to be numerical entries, are indeed numerical.
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
    assert default_cosmology2["wa"] != ci.default_cosmology()["wa"], \
        "default_cosmology is not returning a copy of the reference " + \
        "model, but the reference model itself!"


def test_specify_neutrino_mass():
    m0_massless = ci.specify_neutrino_mass(ci.default_cosmology(), 0, 0)

    assert m0_massless["nnu_massive"] == 0, "There should be no massive " + \
        "species in the massless case."
    assert m0_massless["omnuh2"] == 0, "There should be no physical " + \
        "density in neutrinos in the massless case."

    m0_massive = ci.specify_neutrino_mass(ci.default_cosmology(), 0.02)

    assert m0_massive["nnu_massive"] == 1, "There should be one massive " + \
        "species in the unspecified case."
    assert m0_massive["omnuh2"] == 0.02, "The input omnuh2 does not " + \
        "match the recorded omnuh2."


def test_input_cosmology_direct_values():
    """
    Unfinished: make sure that ALL values were correctly transcribed.
    """
    default_cosmology = ci.default_cosmology()
    pars = ci.input_cosmology(default_cosmology())

    assert pars.H0 == 100 * default_cosmology["h"], "incorrect Hubble constant"
    assert pars.ombh2 == default_cosmology["ombh2"], "incorrect physical " + \
        "density in baryons transcribed."
    assert pars.omch2 == default_cosmology["omch2"], "incorrect physical " + \
        "density in cdm transcribed."
    assert pars.omnuh2 == default_cosmology["omnuh2"], "incorrect " + \
        "physical density in neutrinos transcribed."
    assert pars.nnu_massive == default_cosmology["nnu_massive"], \
        "incorrect number of massive neutrinos transcribed."
        
    assert pars.OmK == default_cosmology["OmK"], "incorrect fractional " + \
        "density in curvature transcribed."
    assert pars.InitPower.ns == default_cosmology["n_s"], "incorrect " + \
        "spectral index transcribed."
    assert pars.InitPower.ns == default_cosmology["A_s"], "incorrect " + \
        "scalar mode amplitude transcribed."


def test_input_dark_energy():
    """
    These tests make sure that input_dark_energy() correctly transcribes
    input dark energy values. Since the function accepts CAMBParams objects
    rather than cosmologies, we use input_cosmology first and then call
    input_dark_energy() again.
    """
    default_cosmology = ci.default_cosmology()
    pars = ci.input_cosmology(default_cosmology)
    
    new_wa = default_cosmology["wa"] - 2
    new_w0 = default_cosmology["w0"] - 2
    ci.input_dark_energy(pars, new_w0, new_wa)
    
    assert pars.DarkEnergy.w == new_w0, "incorrect w0 transcribed."
    assert pars.DarkEnergy.wa == new_wa, "incorrect wa transcribed."


def test_input_cosmology_indirect_vals():
    default_cosmology = ci.default_cosmology()
    pars = ci.input_cosmology(default_cosmology)

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

