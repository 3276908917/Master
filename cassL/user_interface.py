import numpy as np
from cassL import lhc
from cassL import train_emu as te

# In keeping with the format of values typically quoted in the literature for
# the scalar mode amplitude (see, for example, Spurio Mancini et al, 2021 ),
# we here define our prior ranges with the use of exponentiation.
A_MEGA_MIN = np.exp(1.61) / 10 ** 10
A_MEGA_MAX = np.exp(5) / 10 ** 10

# The MINI qualifier indicates that these belong to the "classic" prior ranges
# (see the function get_param_ranges).
A_CLASSIC_MIN = np.exp(2.35) / 10 ** 10
A_CLASSIC_MAX = np.exp(3.91) / 10 ** 10

def get_param_ranges(priors="COMET", massive_neutrinos=True):
    """
    !
    Return a dictionary of arrays where each key is a cosmological parameter
    over which the emulator will be trained. The first entry of each array is
    the parameter's lower bound, the second entry is the parameter's upper
    bound.

    @priors string indicating which set of parameters to use.
        "MEGA": the original goal for this project, which unfortunately
            suffered from too many empty cells. The code has gone through
            several crucial bug fixes since switching to a different set of
            priors, so we need to test this prior suite again and re-assess the
            rate of empty cells.
        "classic": a prior range with widths in between those of "COMET" and
            "MEGA." We need to test this prior suite again to see if it still
            suffers from a large number of empty cells.
        "COMET" as of 19.06.23, this is the default for the emulator. It is
            the most restrictive of the three options and is intended to
            totally eliminate the problem of empty cells, so that a complete
            LHC can be used to train a demonstration emulator. The hope is for
            the demonstration emulator trained over such priors to be extremely
            accurate due to the very narrow permissible parameter values.

    @massive_neutrinos should be set to False when one is training the emulator
        for massless neutrinos. This is because the massless neutrino emulator
        should be using two fewer parameters--A_s and omega_nu_h2 are no longer
        appropriate.
    """
    param_ranges = {}

    if priors == "MEGA":
        param_ranges = {
            'ombh2': [0.005, 0.28],
            'omch2': [0.001, 0.99], # max 0.3?
            'n_s': [0.7, 1.3], # expand?
            'sigma12': [0.2, 1], # based on Sanchez et al 21 and
                # Sanchez 20, fig 2 
        }
    elif priors == "classic": # This is useful for a demo run. 
        param_ranges = {
            'ombh2': [0.01875, 0.02625],
            'omch2': [0.05, 0.255],
            'n_s': [0.84, 1.1],
            'sigma12': [0.2, 1], # based on Sanchez et al 21; Sanchez 20 fig 2 
        }
    elif priors=="COMET":
        param_ranges = {
            'ombh2': [0.0205, 0.02415],
            'omch2': [0.085, 0.155],
            'n_s': [0.92, 1.01],
            'sigma12': [0.2, 1], # based on Sanchez et al 21; Sanchez 20 fig 2 
        }

    if massive_neutrinos:
        if priors == "MEGA":
            param_ranges['A_s'] = [A_MEGA_MIN, A_MEGA_MAX]
        elif priors == "classic":
            param_ranges['A_s'] = [A_CLASSIC_MIN, A_CLASSIC_MAX]
        elif priors == "COMET": 
            param_ranges['A_s'] = [1.15e-9, A_MINI_MAX]

        param_ranges['omnuh2'] = [0., 0.01]

    return param_ranges

def build_train_and_test_sets(scenario_file_handle):
    """
    Options in the scenario file:
    * A header with a comment explaining, in English, the basic gist of the
        scenario.
    * A totally-unique scenario name.
    * Flag: do we need to also create a corresponding new test hypercube?
    * Number of k points associated with each spectrum.
    * Number of samples associated with the hypercube, default 5k.
    
    THIS FUNCTION SHOULD ISSUE A VERY LOUD WARNING IF ANY VALUES ARE IMPLIED
    FROM DEFAULT VALUES!!!
    
    (have a very specific directory with the following subdirectories: \
        1. Initial LHCs (i.e. stuff spat out by lhs.py functions)
            New directories not necessary, because some scenarios will only
            generate one file anyway.
        2. Backups
            At its peak, this folder will contain six files from the current
            run: three for the most recent backups and three for the backups
            before that. Then the old backups will be promptly deleted.
        3. Final products
            * Create a new directory for each scenario!!
    """
    return 23

def build_and_test_emulator(X_train, Y_train, X_test, Y_test, priors,
    emu_label):
    """
    Build a new Gaussian process regression over X_train and Y_train, then
    test its accuracy using X_test and Y_test.
    This function automatically throws out bad entries in all X and Y inputs.
    
    Just like the code in train_emu.py, this cannot yet handle the case of
    different sampling distributions, e.g. root- or square-sampling in sigma12.
    """
    X_train_clean, Y_train_clean = \
        te.eliminate_unusable_entries(X_train, Y_train)
    trainer = te.Emulator_Trainer(X_train_clean, Y_train_clean, priors)
    trainer.train()
    
    X_test_clean, Y_test_clean = \
        te.eliminate_unusable_entries(X_test, Y_test)
    trainer.test(X_test_clean, Y_test_clean)
    
    # trainer.test(X
