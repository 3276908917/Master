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

def get_param_ranges(prior_name="COMET_with_nu"):
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

    prior_file = "priors/" + prior_name + ".txt"
    
    with open(prior_file, 'r') as file:
        lines = file.readlines()
        key = None
        for line in lines:
            if line[0] == "$":
                key = line[1:].strip()
            else:
                bounds = line.split(",")
                param_ranges[key] = [float(bounds[0]), float(bounds[1])]

    return param_ranges

def build_train_and_test_sets(scenario_name):
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

    # Step 1: read and error-check scenario file
    
    # Keep track of the scenario folder, so that we can save things there...
    
    scenario = {}
    file_handle = "scenarios/" + scenario_name + "/" + scenario_name + ".txt"
    
    with open(file_handle, 'r') as file:
        lines = file.readlines()
        key = None
        for line in lines:
            if line[0] == "$":
                key = line[1:].strip()
            else:
                val = line
                if line == "None":
                    val = None
                elif line.isnumeric():
                    val = float(line)
                scenario[key] = val
    
    # Step 2: build train LHC
    
    train_lhc = lhc.multithread_unit_LHC_builder(dim, n_samples, label="unlabeled",
        num_workers=12, previous_record=0):
    
    # Step 3: fill-in train LHC
    
    # Step 4: build test LHC
    
    # Step 5: fill-in test LHC
    
    # Step 6: call build_and_test_emulator



    # The function needs to auto-detect how much progress we've already made
    # on each particular scenario.

    # The function needs to store results in carefully organized directories

    # The function needs to automatically delete backups that are no longer
    # needed.

    return 23

def get_data_dict(emu_name, prior_name="COMET"):
    #! WATCH OUT! THIS FUNCTION ASSUMES MASSIVE NEUTRINOS ALWAYS

    # This will return a dictionary which the new iteration of
    # build_and_test_emulator will be able to expand into all of the info
    # necessary to build an emulator.
    
    # e.g. emu_name is Hnu2_5k_knockoff
    
    # This function will have to be expanded dramatically once we implement
    # the scenario structure
    directory = "data_sets/" + emu_name + "/"
    data_dict = {"emu_name": emu_name}

    X_train = np.load(directory + "lhc_train_final.npy", allow_pickle=False)
    data_dict["X_train"] = X_train

    Y_train = np.load(directory + "samples_train.npy", allow_pickle=False)
    data_dict["Y_train"] = Y_train

    X_test = np.load(directory + "lhc_test_final.npy", allow_pickle=False)
    data_dict["X_test"] = X_test
    
    Y_test = np.load(directory + "samples_test.npy", allow_pickle=False)
    data_dict["Y_test"] = Y_test
    
    priors = get_param_ranges(prior_name=prior_name)
    data_dict["priors"] = priors

    return data_dict

def build_and_test_emulator(data_dict):
    """
    Build a new Gaussian process regression over X_train and Y_train, then
    test its accuracy using X_test and Y_test.
    This function automatically throws out bad entries in all X and Y inputs.
    
    Just like the code in train_emu.py, this cannot yet handle the case of
    different sampling distributions, e.g. root- or square-sampling in sigma12.
    """
    X_train_clean, Y_train_clean = \
        te.eliminate_unusable_entries(data_dict["X_train"],
                                      data_dict["Y_train"])
    trainer = te.Emulator_Trainer(data_dict["emu_name"], X_train_clean,
                                  Y_train_clean, data_dict["priors"])
    trainer.train()
    
    trainer.save()
    
    X_test_clean, Y_test_clean = \
        te.eliminate_unusable_entries(data_dict["X_test"], data_dict["Y_test"])
    trainer.test(X_test_clean, Y_test_clean)

    trainer.error_hist()

    return trainer

