import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import camb
# This line should be removed as soon as Andrea's code is fully integrated
from camb import model
import re
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import quad
import copy as cp

import os
data_prefix = os.path.dirname(os.path.abspath(__file__)) + "/"

from cassL import utils

# This isn't inside a function because it's annoying to edit global
# variables...
try:
    # Keep in mind that 'cosmologies.dat' is NOT the same file as the original
    # 'cosmology_Aletheia.dat' that Ariel provided. In order to facilitate the
    # reading-in of the file, we make some minor formatting adjustments such as
    # the removal of number signs. Use the unaltered version will cause a
    # segfault.
    path_to_cosms = data_prefix + "cosmologies.dat"
    full_cosm = pd.read_csv(path_to_cosms, sep=r'\s+')
    
    cosm = full_cosm[:8] # exclude model 8 always 
    
    # Eliminate unused columns which represent clutter in the pipeline that
    # obstructs debugging.
    cosm = cosm.drop("Name", axis=1)
    cosm = cosm.drop("OmC", axis=1)
    cosm = cosm.drop("OmB", axis=1)
    cosm = cosm.drop("OmM", axis=1)
    cosm = cosm.drop("OmL", axis=1)
    cosm = cosm.drop("EOmDE", axis=1)
    cosm = cosm.drop("Lbox", axis=1)
    cosm = cosm.drop("sigma8", axis=1)  
    
    for series_name, series in cosm.items():
        if series.dtype != float:
            if series_name == "EOmDE" or series_name == "Name":
                continue
            else:
                cosm[series_name] = pd.to_numeric(series)
                
except FileNotFoundError:
    print("Failure to load table of Aletheia cosmologies.")

# ! If there is a justification for using these specific values, one would
# have to ask Ariel for it. I use these values because he used them. Anyway,
# the values used here have nothing to do with the emulator, we are only using
# them as reference values at which to compare code implementations.
OMNUH2_FLOATS = np.array([0.0006356, 0.002148659574468, 0.006356, 0.01])

# Add corresponding file accessors, to check our work later. These strings,
# which contain approximations of the values in OMNUH2_FLOATS, are used to
# access the power spectra save files provided to us by Ariel.
OMNUH2_STRS = np.array(["0.0006", "0.002", "0.006", "0.01"])

ALETHEIA_SNAPS = np.array([0, 1, 2, 3, 4])

# ! Just some standard colors and styles for when I plot several models
# together. We should figure out a way to get rid of this.
colors = ["green", "blue", "brown", "red", "black", "orange", "purple",
          "magenta", "cyan"] * 200
styles = ["solid"] * 200

# This regex expression powers the parse_redshifts function.
redshift_column = re.compile("z.+")

def default_cosmology():
    r"""
    Return a deep copy of a benchmark cosmology (in this case, Aletheia model
    0, based on the Planck best fit for a flat LambdaCDM cosmology) such that
    the user is free to modify the result without having to worry about making
    a copy.

    Returns: dictionary (a copy of the default cosmology)
    """
    return specify_neutrino_mass(cosm.iloc[0], 0)
    

def parse_redshifts(model_num):
    r"""
    Return the list of redshifts given for a particular model in the Aletheia
    dat file. The models are equal in sigma12 for each index of this list.
    For example, sigma12(model a evaluated at parse_reshifts(a)[j]) is equal to
    sigma12(model b evaluated at parse_redshifts(b)[j]).

    This function is intended to return the redshifts in order from high (old)
    to low (recent), since this is the order that CAMB will impose unless
    already used.

    Parameters
    ----------
    model_num: int
        Index corresponding to the Aletheia model. For example, model 0
        corresponds to the Planck best fit configuration.

    Returns
    -------
    z: numpy.ndarray of float64
        List of redshifts at which to evaluate the model so that the sigma12
        values of the different models match.
    """
    z = []
    model = cosm.loc[model_num]

    for column in cosm.columns:
        # In the file, all of the columns containing redshift information begin
        # with the letter 'z.'
        if redshift_column.match(column):
            z.append(model[column])

    return np.flip(np.sort(np.array(z)))


def balance_neutrinos_with_CDM(cosmology, new_omnuh2):
    r"""
    Return a modified cosmology dictionary with a physical density in neutrinos
    specified by new_omnuh2, but with an equivalent total physical density in
    matter. This is achieved by drawing from or adding to the physical density
    in cold dark matter. Therefore, the returned cosmology will be equivalent
    to the original cosmology for all parameters except for omnuh2 and omch2.

    This function reduces to a sort of get_MEMNeC function when new_omnuh2 is
    0.

    Parameters
    ----------
    cosmology: dict
        A dictionary of settings for cosmological parameters. The precise
        format is specified in the file "standards.txt". In particular, keep in
        mind that the cosmology will be assumed to completely lack massive
        neutrinos in the event that both of the keys 'mnu' and 'omnuh2' are
        missing.

    new_omnuh2: float
        The new desired value for the physical density in neutrinos. The
        returned cosmology dictionary will have
        new_cosmology["omnuh2"] == new_omnuh2
        In other words, new_omnuh2 is not a delta, but the new setting, and
        the function will behave appropriately even if
        cosmology["omnuh2"] != 0.

    Returns
    -------
    (dict)
        A dictionary of settings for cosmological parameters. The precise
        format is specified in the file "standards.txt".
    """
    new_cosmology = cp.deepcopy(cosmology)

    # If omnuh2 has not yet been specified, assume it is zero by default.
    old_omnuh2 = 0
    if "omnuh2" in cosmology:
        old_omnuh2 = cosmology["omnuh2"]

    delta = new_omnuh2 - old_omnuh2
    assert delta < cosmology["omch2"], "Not enough density in CDM to " + \
        "complete the desired transfer!"
    new_cosmology["omch2"] -= delta

    #! It might be dangerous for us to keep assuming '1' when the neutrinos are
    # massive...
    new_nnu_massive = 0 if new_omnuh2 == 0 else 1
    return specify_neutrino_mass(new_cosmology, new_omnuh2, new_nnu_massive)

def load_benchmark(relative_path, omnuh2_strs=None):
    r"""
    Return a nested dictionary containing the power spectra that Ariel computed
    using CAMB in native Fortran. These power spectra were stored in a series
    of files with the common prefix "powernu."

    Parameters
    ----------
    relative_path: str
       The relative file path of the particular save file containing the
       desired benchmark spectra. The file path is relative to the 'shared'
       folder in the 'CAMB_notebooks' directory.

    Returns
    -------
    benchmark: dict
        Power spectra nested in a series of arrays and dictionaries. The order
        of layers from outer to inner is as follows:
            1. A dictionary whose keys are strings representing approximations
                of the true omega_nu_h2 values to which they correspond. For
                specific examples of keys and the true vales behind them,
                compare the global constant arrays OMNU_FLOATS and OMNU_STRS
                defined at the top of camb_interface.py.
            2. An array whose indices correspond to Aletheia model indices.
            3. An array whose indices correspond to snapshot indices. Lower
                shapshot indices correspond to higher values of redshift.
            4. A dictionary whose key-value pairs are as follows:
                "P_nu": np.ndarray of float64
                    the power spectrum of the cosmology defined by that
                        particular Aletheia model
                "P_no": np.ndarray of float64
                    the power spectrum of its MEMNeC
                "k": np.ndarray of float64
                    k[i] is the inverse of the physical scale associated with
                    each P_nu[i] and P_no[i]
    """

    def iterate_over_models_and_redshifts(accessor="0.002"):
        nested_spectra = []
        for i in range(0, 9):  # iterate over models
            nested_spectra.append([])
            for j in range(0, 5):  # iterate over snapshots
                next_file_name = relative_path + accessor + "_caso" + \
                            str(i) + "_000" + str(j) + ".dat"
                next_spectrum = pd.read_csv(next_file_name,
                                            names=["k", "P_no", "P_nu",
                                                   "ratio"],
                                            sep=r'\s+')
                nested_spectra[i].append(next_spectrum)

        return nested_spectra

    if omnuh2_strs is None:
        return iterate_over_models_and_redshifts()

    benchmark = {}

    for i in range(len(omnuh2_strs)):
        file_accessor = omnuh2_strs[i]
        benchmark_key = OMNUH2_STRS[i]

        benchmark[benchmark_key] = \
            iterate_over_models_and_redshifts(file_accessor)

    return benchmark


def is_matchable(target, cosmology):
    r"""
    !
    Delete this function? When was the last time that we used it?
    """
    # I thought bigger sigma12 values were supposed to come with lower z,
    # but the recent matching results have got me confused.
    _, _, _, s12_big = evaluate_cosmology(cosmology, 0, nu_massive=False,
                                          redshifts=[0])


def match_sigma12(target, tolerance, cosmology,
                  _redshifts=np.flip(np.linspace(0, 1100, 150)), _min=0):
    """
    !
    Delete this function? When was the last time that we used it?
        Return a redshift at which to evaluate the power spectrum of cosmology
    @cosmology such that the sigma12_massless value of the power spectrum is
    within @tolerance (multiplicative discrepancy) of @target.

    @target this is the value of sigma12_massless at the assumed redshift
        (e.g. typically at z=2.0 for a standard Aletheia model-0 setup).
    @cosmology this is the cosmology for which we want to find a sigma12 value,
        so this will typically be an exotic or randomly generated cosmology
    @tolerance ABS((target - sigma12_found) / target) <= tolerance is the
        stopping condition for the binary search that this routine uses.
    @_z: the redshift to test at. This is part of the internal logic of the
        function and should not be referenced elsewhere.
    """
    # If speed is an issue, let's reduce the k samples to 300, or at least
    # add num_samples as a function parameter to kzps

    # First, let's probe the half-way point.
    # We're assuming a maximum allowed redshift of $z=2$ for now.
#!
    # print(_z)
    _, _, _, list_s12 = evaluate_cosmology(cosmology, _redshifts,
                                           k_points=1000)

    import matplotlib.pyplot as plt
    # print(list_s12)
    if False:
        plt.plot(_redshifts, list_s12)
        plt.axhline(sigma12_in)
        plt.show()

    # debug block
    if False:
        plt.plot(_redshifts, list_s12 - sigma12_in)
        plt.axhline(0)
        plt.show()

    list_s12 -= target  # now it's a zero-finding problem

    # !For some reason, flipping both arrays helps the interpolator
    # But I should come back and check this, I'm not sure if this was just a
    # patch for the Newton method
    interpolator = interp1d(np.flip(_redshifts), np.flip(list_s12),
                            kind='cubic')
    try:
        z_best = root_scalar(interpolator, bracket=(np.min(_redshifts),
                                                    np.max(_redshifts))).root
    except ValueError:
        print("No solution.")
        return None

    _, _, _, s12_out = evaluate_cosmology(cosmology, redshifts=[z_best],
                                          k_points=1000)
    discrepancy = (s12_out[0] - target) / target
    if abs(discrepancy) <= tolerance:
        return z_best
    else:
        z_step = _redshifts[0] - _redshifts[1]
        new_floor = max(0, z_best - z_step)
        new_ceil = min(1100, z_best + z_step)
        new_redshifts = np.flip(np.linspace(new_floor, new_ceil, 150))
        return match_sigma12(target, tolerance, cosmology,
                         _redshifts=new_redshifts)


def get_As_matched_random_cosmology(omega_nu, A_s=2.12723788013000E-09):
    """
    !
    Return a cosmological configuration based on model0 but uniformly
    randomized in:
        h: [0.2, 1]
        OmegaK: [-.05, 0]
        w0: [-2, -.5]
        wa: [-.5, .5]

    Warning! You may have to throw out some of the cosmologies that you get
    from this routine because I am nowhere guaranteeing that the sigma12 you
    want actually corresponds to a positive redshift... of course, this
    wouldn't be a problem if CAMB allowed negative redshifts.

    ! This function is nearly the same as get_random_cosmology(). Can we
        collapse the two?

    Returns
    -------
    row: dict
        Configuration of cosmological parameters in the format of a row read in
        from the data table of Aletheia models.

    ! Unfortunately, all of these bounds are hard-coded. Maybe we can read in a
    table for this?
    """
    row = cp.deepcopy(cosm.iloc[0])
    
    omdeh2 = np.random.uniform(0.1, 0.5)
    omKh2 = np.random.uniform(-0.05, 0.05)

    row['h'] = np.sqrt(row["ombh2"] + row["omch2"] + omdeh2 + omKh2)
    
    # These fields are not used anywhere
    # Given h, the following are now fixed:
    # row['OmB'] = row['ombh2'] / row['h'] ** 2
    # row['OmC'] = row['omch2'] / row['h'] ** 2
    # row['OmM'] = row['OmB'] + row['OmC']

    row['OmK'] = omKh2 / row['h'] ** 2
        
    # This field is not used anywhere
    #row['OmL'] = 1 - row['OmM'] - row['OmK']

    # ~ Do we have any constraints on h besides Aletheia?
    # I ask because this seems like a pretty small window.
    # ditto
    row['w0'] = np.random.uniform(-2., -.5)
    # ditto
    row['wa'] = np.random.uniform(-0.5, 0.5)

    row['A_s'] = A_s
    
    nnu_massive = 1 if omega_nu != 0 else 0

    return specify_neutrino_mass(row, omega_nu)


#! This is really an awful approach, but it should make the code marginally
# faster.
A_min = np.exp(1.61) / 10 ** 10
A_max = np.exp(5) / 10 ** 10

def get_random_cosmology(omega_nu):
    r"""
    !
    Return a cosmological configuration based on model0 but uniformly
    randomized in:
        h: [0.2, 1]
        OmegaK: [-0.05, 0.05]
        A_s: about [5.0028e-10, 1.4841e-8]
        w0: [-2, -.5]
        wa: [-.5, .5]

    Returns
    -------
    row: dict
        Configuration of cosmological parameters in the format of a row read in
        from the data table of Aletheia models.

    ! Unfortunately, all of these bounds are hard-coded. Maybe we can read in a
    table for this?
    """
    row = get_As_matched_random_cosmology(omega_nu)
    
    row['A_s'] = np.random.uniform(A_min, A_max)

    return row
    

#! It's confusing to the user to have both omnuh2_floats and skips_omega as
# parameters. Let's just assume that omnuh2_floats already skips the
# uninteresting values!
def boltzmann_battery(omnuh2_floats, skips_omega=[0, 2], skips_model=[8],
                      skips_snapshot=[1, 2, 3], hubble_units=False,
                      models=cosm, fancy_neutrinos=False, k_points=100000):
    """
    !
    We should get rid of omnuh2_strs and make the outer layer a dictionary
    where omnuh2_floats are the keys.

    Return a nested dictionary containing power spectra computed by CAMB via
    the Python interface.

    Parameters
    ----------
    omnuh2_floats: list or np.ndarray of float64
        The values for omnuh2 at which to compute the power spectra for each
        model and snapshot.
    omnuh2_strs: list or np.ndarray of str
        The labels associated with each index of omnuh2_floats.

    Returns
    -------
    spectra: dict
        Power spectra nested in a series of arrays and dictionaries. The order
        of layers from outer to inner is as follows:
            1. A dictionary whose keys are floats representing corresponding to
                the value of omega_nu_h2 used.
                defined at the top of camb_interface.py.
            2. An array whose indices correspond to Aletheia model indices.
            3. An array whose indices correspond to snapshot indices. Lower
                shapshot indices correspond to higher values of redshift.
            4. A dictionary whose key-value pairs are as follows:
                "P_nu": np.ndarray of float64
                    the power spectrum of the cosmology defined by that
                        particular Aletheia model
                "P_no": np.ndarray of float64
                    the power spectrum of its MEMNeC
                "k": np.ndarray of float64
                    k[i] is the inverse of the physical scale associated with
                    each P_nu[i] and P_no[i]
                "s12_massless": the sigma12 of the massive-neutrino case
                "s12_massive": the sigma12 of the massless-neutrino case
    """
    assert isinstance(omnuh2_floats, list) or \
        isinstance(omnuh2_floats, np.ndarray), "if you want only one" + \
        " omega value, you must still nest it in a list."

    spectra = {}

    for this_omnuh2_index in range(len(omnuh2_floats)):
        #print(this_omnuh2_index % 10, end='')
        this_omnuh2_float = omnuh2_floats[this_omnuh2_index]
        if this_omnuh2_index in skips_omega:
            spectra[this_omnuh2_float] = None
            continue
        spectra[this_omnuh2_float] = []
        for mindex, row in models.iterrows():
            if mindex in skips_model:
                # For example, I don't yet understand how to implement model 8
                spectra[this_omnuh2_float].append(None)
                continue

            h = row["h"]
            spectra[this_omnuh2_float].append([])

            z_input = parse_redshifts(mindex)
            if None in z_input:
                spectra[this_omnuh2_float][mindex] = None
                continue

            for snap_index in range(len(z_input)):
                #  Since z_input is ordered from z large to z small, and since
                # snap indices run from z large to z small,
                # z_index = snap_index in this case and NOT in general.
                if snap_index in skips_snapshot:
                    spectra[this_omnuh2_float][mindex].append(None)
                    continue

                inner_dict = {}
                z = z_input[snap_index]

                massless_nu_cosmology = specify_neutrino_mass(
                    row, 0, nnu_massive_in=0)
                massless_tuple = \
                    evaluate_cosmology(massless_nu_cosmology, redshifts=[z],
                                       fancy_neutrinos=fancy_neutrinos,
                                       k_points=k_points,
                                       hubble_units=hubble_units)
                inner_dict["k"] = massless_tuple[0]
                inner_dict["P_no"] = massless_tuple[2]
                inner_dict["s12_massless"] = massless_tuple[3]

                massive_nu_cosmology = specify_neutrino_mass(
                    row, this_omnuh2_float, nnu_massive_in=1)

                # Adjust CDM density so that we have the same total matter
                # density as before:
                massive_nu_cosmology["omch2"] -= this_omnuh2_float

                massive_tuple = \
                    evaluate_cosmology(massive_nu_cosmology, redshifts=[z],
                                       fancy_neutrinos=fancy_neutrinos,
                                       k_points=k_points,
                                       hubble_units=hubble_units)
                inner_dict["P_nu"] = massive_tuple[2]
                inner_dict["s12_massive"] = massive_tuple[3]

                inner_dict["z"] = z_input[snap_index]

                assert np.array_equal(massless_tuple[0], massive_tuple[0]), \
                    "assumption of identical k axes not satisfied!"

                spectra[this_omnuh2_float][mindex].append(inner_dict)

    return spectra


def make_neutrinos_fancy(pars, nnu_massive):
    """
    This is a kzps helper function which enables the infamous fancy neutrino.
    In practice, this only overrides the effective number of massless neutrinos
    by a small amount, which nonetheless leads to irreconcilable discrepancies
    compared to observations.
    """
    pars.num_nu_massless = 3.046 - nnu_massive
    pars.nu_mass_eigenstates = nnu_massive
    stop_i = pars.nu_mass_eigenstates + 1
    pars.nu_mass_numbers[:stop_i] = \
        list(np.ones(len(pars.nu_mass_numbers[:stop_i]), int))
    pars.num_nu_massive = 0
    if nnu_massive != 0:
        pars.num_nu_massive = sum(pars.nu_mass_numbers[:stop_i])


def apply_universal_output_settings(pars):
    """
    This is a kzps helper function which modifies the desired accuracy of CAMB
    and which disables certain outputs in which we are not interested.
    """

    # The following lines are desperation settings
    # If we ever have extra time, we can more closely study what each line does
    pars.WantCls = False
    pars.Want_CMB = False
    pars.Accuracy.AccuratePolarization = False
    pars.DoLensing = False
    pars.WantScalars = False
    
    pars.NonLinear = camb.model.NonLinear_none
    
    pars.YHe = 0.24
    
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuracyBoost = 3
    
    # Transfer.kmax describes an absolute k
    pars.Transfer.kmax = 10.0

def input_dark_energy(pars, w0, wa):
    """
    Helper function for input_cosmology.
    Handles dark energy by using a PPF model (which allows a wide range of
        w0 and wa values) unless w0 is -1 and wa is zero, in which case we use
        the two-fluid approximation.
    """
    # Default is fluid, so we don't need an 'else'
    if w0 != -1 or wa != 0:
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')


def specify_neutrino_mass(cosmology, omnuh2_in, nnu_massive_in=None):
    """
    Helper function for input_cosmology.
    This returns modified copy (and therefore does not mutate the original) of
    the input dictionary object, which corresponds to a cosmology with massive
    neutrinos.
    """
    # Default behavior: if no nnu_massive_in specified, take it to be 1 in the
    # massive-neutrino case, 0 otherwise
    if nnu_massive_in is None:
        nnu_massive_in = 1
    if omnuh2_in == 0 and nnu_massive_in == 0:
        raise UserWarning("CAMB crashes with nnu_massive == 0, even if ",
            "omnuh2 is zero. We're overriding and setting nnu_massive = 0...")

    full_cosmology = cp.deepcopy(cosmology)

    full_cosmology["omnuh2"] = omnuh2_in

    '''This is a horrible workaround, and I would like to get rid of it
    ASAP. It destroys dependence on TCMB and
    neutrino_hierarchy, possibly more. But CAMB does not accept omnuh2 as
    an input, so I have to reverse-engineer it somehow.

    Also, should we replace default_nnu with something else in the
    following expression? Even if we're changing N_massive to 1,
    N_total_eff = 3.046 nonetheless, right?'''
    # full_cosmology["mnu"] = omnuh2_to_mnu(full_cosmology["omnuh2"])

    # print("The mnu value", mnu_in, "corresponds to the omnuh2 value",
    #    omnuh2_in)
    # full_cosmology["omch2"] -= omnuh2_in
    ''' The removal of the above line is a significant difference, but it
        allows us to transition more naturally into the emulator sample
        generating code.'''

    full_cosmology["nnu_massive"] = nnu_massive_in

    return full_cosmology

def verify_cosmology(cosmology):
    """
    Makes sure all of the essential keys are in place for CAMB usage.
    """
    essential_keys = ["h", "ombh2", "omch2", "OmK", "omnuh2", "A_s", "n_s", \
        "w0", "wa"]
    for essential_key in essential_keys:
        if essential_key not in cosmology:
            raise ValueError("The provided cosmology does not specify " + \
                essential_key + "!")


def input_cosmology(cosmology):
    """
    Helper function for kzps.
    Read entries from a dictionary representing a cosmological configuration.
    Then write these values to a CAMBparams object and return.
    """
    # If the cosmology is underspecified, serve the user a clear complaint
    verify_cosmology(cosmology)    

    h = cosmology["h"]

    # tau is a desperation argument
    # Why are we using the degenerate hierarchy? Isn't that wrong?
    pars = camb.set_params(
        H0=h * 100,
        ombh2=cosmology["ombh2"],
        omch2=cosmology["omch2"],
        omk=cosmology["OmK"],
        omnuh2=cosmology["omnuh2"],
        #num_massive_neutrinos=cosmology["nnu_massive"],
        tau=0.0952,  # for justification, ask Matteo
        neutrino_hierarchy="degenerate"  # 1 eigenstate approximation; our
        # neutrino setup (see below) is not valid for inverted/normal
        # hierarchies.
    )

    # the last three are desperation arguments
    pars.InitPower.set_params(As=cosmology["A_s"], ns=cosmology["n_s"],
                              r=0, nt=0.0, ntrun=0.0)

    #print(cosmology["A_s"], cosmology["mnu"])

    input_dark_energy(pars, cosmology["w0"], float(cosmology["wa"]))

    return pars

def get_CAMB_sigma12(pars, redshifts=[0]):
    """
    Helper function for evaluate_sigma12.
    Given a fully set-up pars function, return the sigma12 values.
    """

    pars.set_matter_power(redshifts=redshifts, kmax=10.0 / pars.h,
                          nonlinear=False, k_per_logint=20)
    results = camb.get_results(pars)
    return results.get_sigmaR(12, hubble_units=False)

def evaluate_sigma12(cosmology, redshifts=[0]):
    if not isinstance(redshifts, list) and \
        not isinstance(redshifts, np.ndarray):
        raise TypeError("If you want to use a single redshift, you must " + \
            "still nest it in an array.")

    pars = input_cosmology(cosmology)
    apply_universal_output_settings(pars)

    return get_CAMB_sigma12(pars, redshifts)

def get_CAMB_pspectrum(pars, redshifts=[0], k_points=100000,
                       hubble_units=False):
    """
    Helper function for evaluate_cosmology.
    Given a fully set-up pars function, return the following in this order:
        scale axis, redshifts used, power spectra, and sigma12 values.
    """

    # To change the the extent of the k-axis, change the following line as
    # well as the "get_matter_power_spectrum" call.
    pars.set_matter_power(redshifts=redshifts, kmax=10.0 / pars.h,
                          nonlinear=False)

    results = camb.get_results(pars)

    sigma12 = results.get_sigmaR(12, hubble_units=False)

    # In some cursory tests, the accurate_massive_neutrino_transfers
    # flag did not appear to significantly alter the outcome.

    k, z, p = results.get_matter_power_spectrum(
        minkh=1e-4 / pars.h, maxkh=10.0 / pars.h, npoints=k_points,
        var1='delta_nonu', var2='delta_nonu'
    )

    # De-nest for the single-redshift case:
    if len(p) == 1:
        p = p[0]

    if not hubble_units:
        k *= pars.h
        p /= pars.h ** 3

    return k, z, p, sigma12


def evaluate_cosmology(cosmology, redshifts=[0], fancy_neutrinos=False,
                       k_points=100000, hubble_units=False):
    """
    ISSUE! It does not seem like the current state of the code actually
        supports redshifts=None. We should test and make sure, then correct if
        necessary.

    Return the scale axis, redshifts, power spectrum, and sigma12 of a
        cosmological model specified by a dictionary of parameter values.

    Parameters:
    -----------
    cosmology: dict
        a dictionary of value for CAMBparams fields
    redshifts: list or np.ndarray
        collection of redshift values at which to evaluate the model
        If you would like to fix the sigma12 value, specify this in the mlc
        dictionary and set this parameter to None. If you would not like to
        fix the sigma12 value, make sure that the mlc dictionary does not
        contain a non-None sigma12 entry.
    fancy_neutrinos: Boolean
        Whether we attempt to impose a neutrino scheme on CAMB after we've
        already set the physical density. The results seem to be inconsistent
        with observation, but we need more
    """
    ''' Retire this code block until we figure out z dominance
    if redshifts is None:
        assert "sigma12" in cosmology.keys(), \
            "Redshift and sigma12 cannot be supplied simultaneously."
    else:
        assert "sigma12" not in cosmology.keys() or mlc["sigma12"] is None, \
            "Redshift and sigma12 cannot be supplied simultaneously."
    '''
    if not isinstance(redshifts, list) and \
        not isinstance(redshifts, np.ndarray):
        raise TypeError("If you want to use a single redshift, you must " + \
            "still nest it in an array.")

    pars = input_cosmology(cosmology)

    if fancy_neutrinos:
        make_neutrinos_fancy(pars, cosmology["nnu_massive"])

    apply_universal_output_settings(pars)

    return get_CAMB_pspectrum(pars, redshifts, k_points=k_points,
                              hubble_units=hubble_units)


def get_CAMB_interpolator(pars, redshifts=[0], kmax=1, hubble_units=False):
    """
    Helper function for cosmology_to_PK_interpolator.
    Given a fully set-up pars function, return a CAMB PK interpolator object.
    
    The input array @redshifts describes the space over which to build the
    interpolator. For more accuracy, more redshifts should be used. Keep in
    mind that CAMB only allows nz_step as large as 150, so the @redshifts
    array should not exceed 150 in length!
    """
    # To change the the extent of the k-axis, change the following line as well
    # as the "get_matter_power_spectrum" call.
    pars.set_matter_power(redshifts=redshifts, kmax=kmax, nonlinear=False)

    # In some cursory tests, the accurate_massive_neutrino_transfers
    # flag did not appear to significantly alter the outcome.

    gmpi = camb.get_matter_power_interpolator
    PK = gmpi(pars, zmin=min(redshifts), zmax=max(redshifts),
              nz_step=len(redshifts), k_hunit=hubble_units, kmax=kmax,
              nonlinear=False, var1='delta_nonu', var2='delta_nonu',
              hubble_units=hubble_units)

    return PK


def cosmology_to_PK_interpolator(cosmology, redshifts=[0],
    fancy_neutrinos=False, kmax=1, hubble_units=False):
    """
    This is a really rough function, I'm just trying to test out an idea.
    
    The input array @redshifts describes the space over which to build the
    interpolator. For more accuracy, more redshifts should be used. Keep in
    mind that CAMB only allows nz_step as large as 150, so the @redshifts
    array should not exceed 150 in length!
    """
    assert isinstance(redshifts, list) or isinstance(redshifts, np.ndarray), \
        "If you want to use a single redshift, you must still nest it in" + \
        " an array."

    pars = input_cosmology(cosmology)

    if fancy_neutrinos:
        make_neutrinos_fancy(pars, cosmology["nnu_massive"])

    apply_universal_output_settings(pars)

    return get_CAMB_interpolator(pars, redshifts, kmax, hubble_units)


def andrea_interpolator(cosmology):
    """
    It's dangerous to have this function floating around when we're also
    closely testing the andreap.py script in the emulator folder of the
    CAMB_notebooks dir.

    We should consider putting Andrea's code into a special script which
    will be temporarily included in the cassL code to be pip-installed.
    """
    H0 = cosmology['h'] * 100
    ombh2 = cosmology['ombh2']
    omch2 = cosmology['omch2']
    omnuh2 = cosmology['omnuh2']
    As = cosmology['A_s']
    ns = cosmology['n_s']
    omk = cosmology['OmK']
    w0 = cosmology["w0"]
    wa = float(cosmology["wa"])

    #pars = camb.CAMBparams()
    #pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, omnuh2=omnuh2,
        omk=omk)
    print("nnu_massive:", cosmology['nnu_massive'])
    pars.num_nu_massive = 1#cosmology['nnu_massive']
    #omnuh2 = np.copy(pars.omnuh2)
    #print (omnuh2)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='fluid')
    pars.NonLinear = model.NonLinear_none
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False
    pars.Transfer.kmax = 20.0
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)

    #print (pars.num_nu_massive)

    return camb.get_matter_power_interpolator(
        pars, nonlinear=False, hubble_units=False, k_hunit=False,
        kmax=20.0, zmax=20.0, var1='delta_nonu', var2='delta_nonu')


def s12_from_interpolator(PK, z):
    """
    This, too, is Andrea's code. It would be ideal, to remove this entirely.
    """

    def W(x):
        return 3.0 * (np.sin(x) - x*np.cos(x)) / x**3
    def integrand(x):
        return x**2 * PK.P(z,x) * W(x*12)**2

    s12 = quad(integrand, 1e-4, 5)[0]

    return np.sqrt(s12/(2*np.pi**2))


def model_ratios(snap_index, sims, canvas, massive=True, skips=[],
                 subplot_indices=None, active_labels=['x', 'y'],
                 title="Ground truth", omnuh2_str="0.002", models=cosm,
                 suppress_legend=False, linewidth=1):
    """
    There are a couple of annoying formatting differences with the power nu
    dictionary which add up to an unpleasant time trying to squeeze it into the
    existing function...

    Here, the baseline is always model 0,
    but theoretically it should be quite easy
    to generalize this function further.
    """
    P_accessor = None
    if massive is True:
        P_accessor = "P_nu"
    elif massive is False:
        P_accessor = "P_no"

    baseline_h = models.loc[0]["h"]
    baseline_k = sims[0][snap_index]["k"]

    baseline_p = sims[0][snap_index]["P_nu"] / \
        sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p = sims[0][snap_index][P_accessor]

    plot_area = canvas  # if subplot_indices is None
    if subplot_indices is not None:
        if type(subplot_indices) == int:
            plot_area = canvas[subplot_indices]
        else:  # we assume it's a 2d grid of plots
            plot_area = canvas[subplot_indices[0], subplot_indices[1]]
        # No need to add more if cases because an n-d canvas of n > 2 makes no
        # sense.

    k_list = []
    rat_list = []
    for i in range(1, len(sims)):
        if i in skips:
            continue  # Don't know what's going on with model 8
        this_h = models.loc[i]["h"]
        this_k = sims[i][snap_index]["k"]

        this_p = sims[i][snap_index]["P_nu"] / \
            sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p = sims[i][snap_index][P_accessor]

        truncated_k, truncated_p, aligned_p = \
            utils.truncator(baseline_k, baseline_p, this_k, this_p)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, aligned_p / truncated_p,
                       label=label_in, c=colors[i], linestyle=styles[i],
                       linewidth=linewidth)

        k_list.append(truncated_k)
        rat_list.append(aligned_p / truncated_p)

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]", fontsize=24)

    ylabel = r"и($k$)"
    if P_accessor is not None:
        if massive is True:
            ylabel = r"$P_\mathrm{massive} / P_\mathrm{massive, model \, 0}$"
        if massive is False:
            ylabel = r"$P_\mathrm{massless} / P_\mathrm{massless, model \, 0}$"

    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel, fontsize=24)

    if title != "" and title[-1] != ":":
        if len(title) < 2 or title[-2:] != "\n":
            title += ": "
        
    plot_area.set_title(title + r"$\omega_\nu$ = " + omnuh2_str +
                        "; Snapshot " + str(snap_index), fontsize=24)
    
    if not suppress_legend:
        plot_area.legend(fontsize=24, loc='center left',
                         bbox_to_anchor=(1, 0.5))
        
    plot_area.tick_params(labelsize=24)

    return k_list, rat_list


def compare_wrappers(k_list, p_list, sims, snap_index, canvas, massive,
                     subscript, title, skips=[], subplot_indices=None,
                     active_labels=['x', 'y']):
    """
    Python-wrapper (i.e. Lukas') simulation variables feature the _py ending
    Fortran (i.e. Ariel's) simulation variables feature the _for ending
    """

    P_accessor = None
    if massive is True:
        P_accessor = "P_nu"
    elif massive is False:
        P_accessor = "P_no"
    x_mode = P_accessor is None

    # Remember, the returned redshifts are in increasing order
    # Whereas snapshot indices run from older to newer
    z_index = 4 - snap_index

    baseline_h = cosm.loc[0]["h"]

    baseline_k_py = k_list[0] * baseline_h

    baseline_p_py = None
    if x_mode:
        baseline_p_py = p_list[0][z_index]
    else:
        baseline_p_py = p_list[0][z_index] / baseline_h ** 3

    baseline_k_for = sims[0][snap_index]["k"]

    baseline_p_for = sims[0][snap_index]["P_nu"] / \
        sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p_for = sims[0][snap_index][P_accessor]

    plot_area = None
    if subplot_indices is None:
        plot_area = canvas
    elif type(subplot_indices) == int:
        plot_area = canvas[subplot_indices]
    else:
        plot_area = canvas[subplot_indices[0], subplot_indices[1]]

    # k_list is the LCD because Ariel has more working models than I do
    for i in range(1, len(k_list)):
        if i in skips:
            continue
        this_h = cosm.loc[i]["h"]

        this_k_py = k_list[i] * this_h
        this_p_py = None
        if x_mode is False:
            this_p_py = p_list[i][z_index] / this_h ** 3
        else:
            this_p_py = p_list[i][z_index]

        this_k_for = sims[i][snap_index]["k"]

        this_p_for = sims[i][snap_index]["P_nu"] / \
            sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p_for = sims[i][snap_index][P_accessor]

        truncated_k_py, truncated_p_py, aligned_p_py = \
            utils.truncator(baseline_k_py, baseline_p_py, this_k_py,
                      this_p_py, interpolation=this_h != baseline_h)
        y_py = aligned_p_py / truncated_p_py

        truncated_k_for, truncated_p_for, aligned_p_for = \
            utils.truncator(baseline_k_for, baseline_p_for, this_k_for,
                      this_p_for, interpolation=this_h != baseline_h)
        y_for = aligned_p_for / truncated_p_for

        truncated_k, truncated_y_py, aligned_p_for = \
            utils.truncator_neutral(truncated_k_py, y_py, truncated_k_for, y_for)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, truncated_y_py / aligned_p_for,
                       label=label_in, c=colors[i], linestyle=styles[i],
                       linewidth=5)

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")

    ylabel = None
    if x_mode:
        ylabel = r"$ж_i/ж_0$"
    else:
        ylabel = r"$y_\mathrm{py} / y_\mathrm{fortran}$"

    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)

    plot_area.set_title(title)
    plot_area.legend()

    plot_area.set_title(title)
    plot_area.legend()

default_sigma12 = evaluate_sigma12(default_cosmology())