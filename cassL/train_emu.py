import numpy as np
import pylab as pb
import GPy
import copy as cp
import pickle

from cassL import camb_interface as ci
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

constructor_complaint = "emu objects require either two NumPy data sets or" + \
    " an emu file handle."

### We should create an emulator object

def eliminate_unusable_entries(X_raw, Y_raw,
    bad_X_vals = [float('-inf'), float('inf'), None, True, np.nan],
    bad_Y_vals = [float('-inf'), float('inf'), None, True, np.nan, 0]):
    """
    This function looks for 'bad' rows in X
    and Y. If a row is bad in X or Y, the row
    is dropped from both data sets.
    
    Returns: cleaned-up X and Y without
    unusable rows.
    """
    assert len(X) == len(Y), "X and Y must have the same length"
    
    def unusable(row, bad_vals):
        for bad_val in bad_vals:
            if bad_val in row:
                return True
            if np.isnan(bad_val) and True in np.isnan(row):
                return True
            return False
    
    bad_row_numbers = np.array([])
    
    for i in range(len(X)):
        row_x = X[i]
        row_y = Y[i]
        if unusable(row_x, bad_X_vals) or unusable(row_y, bad_Y_vals):
            bad_row_numbers = np.append(bad_row_numbers, i)
    
    cleaned_X = np.delete(X_raw, bad_row_numbers)
    cleaned_Y = np.delete(Y_raw, bad_row_numbers)
    
    return cleaned_X, cleaned_Y
    
def normalize_spectra(Y):
    Ylog = np.log(Y)
    ymu = np.mean(Ylog, axis=0)
    Y_shifted = np.substract(Ylog, ymu)
    ystdev = np.std(Ylog, axis=0)
    Y_normalized = np.divide(Y_shifted, ystdev)
    return Y_normalized, ymu, ystdev

class Emulator_Trainer:

    ### To-do:
    # this should automatically handle different kinds of sampling, i.e. in
    # sigma_{12}: square linear or root linear stuff. Right now the handling is
    # incomplete.

    class Emulator:

        def __init__(self, xmin, xrange, ymu, ystdev):
            # We're requiring all emulators to follow the same approach:
            # all mappings can be represented with just the two variables,
            # xmin and xrange
            self.xmin = xmin
            self.xrange = xrange
            self.ymu = ymu
            self.ystdev = ystdev

        def convert_to_normalized_params(self, config):
            """
            Using the originally input prior, convert a particular cosmological
            configuration into a set of unit-LHC coordinates.
            """
            assert len(x) == self.dim, "This is a " + str(self.dim) + \
                "-dimensional emulator. Input vector had only " + len(x) + \
                " dimensions."
               
            return (config - self.xmin) / self.xrange
            
        def _predict_normalized_spectrum(self, x):
            """
            This function should really only be used in debugging cases. To
            obtain a power spectrum from the emulator, use the function
            predict_pspectrum
            """
            #! Maybe we should actually experiment with these uncertainties and
            # see if they ever come in handy. Or we could just keep throwing
            # them out.
            guess, uncertainties = self.gpr.predict(x)
            return guess
            
        def predict_pspectrum(self, x):
            # Instead of solving the x formatting complaints by blindly
            # re-nesting x, let's try to get to the bottom of *why* the
            # formatting is so messed up in the first place.
            return inverse_transform(_predict_normalized_spectrum(x))

    def __init__(self, *args):
        """
        There are two ways one can go about instantiating:
        1. provide an X and Y data set as well as a set of priors -> emulator i
            trained
        2. provide the file handle associated with a trained emulator object
        """
        if len(args) == 1:
            # We're expecting an already-trained emulator object
            file_handle = args[0]
            assert isinstance(file_handle, str), constructor_complaint

            self.emu = pickle.load(open(file_handle), "rb")

        elif len(args) == 3:
            self.X = args[0]

            self.Y = args[1]
            priors = args[2]

            assert isinstance(self.X, np.ndarray) and \
                isinstance(self.Y, np.ndarray) and \
                isinstance(self.priors, dict), constructor_complaint

            self.normalized_Y, ymu, ystdev = normalize_spectra(self.Y)

            xmin = np.array([])
            xrange = np.array([])

            ### This section is INCREDIBLY unsightly. Let's clean up and
            # generalize.

            for key in par_ranges.keys():
                xmin = np.append(xmin, par_ranges[key][0])
                xrange = np.append(xrange,
                                   par_ranges[key][1] - par_ranges[key][0])

                if sigma12_sampling != 'linear':
                    if key == "n_s":
                        xmin = np.append(xmin, 0)
                        xrange = np.append(xrange, 0)
                    if key == "omnuh2":
                        break
                
            if sigma12_sampling == 'root':
                xmin[3] = 0.2 ** 0.5
                xrange[3] = 1 - 0.2 ** 0.5
            elif sigma12_sampling == 'square':    
                xmin[3] = 0.04
                xrange[3] = 0.96
                
            ### End unsightly section

            self.emu = Emulator(priors, xmin, xrange, ymu, ystdev)
            self.emu.dim = len(self.X[0])
        else:
            raise Exception(constructor_complaint)
            
            
    def train(self):
        assert self.X is not None and self.normalized_Y is not None, \
            "No data found over which to train."

        # The dimension is automatically the length of an X element.
        remaining_variance = np.var(self.normalized_Y)

        kernel1 = GPy.kern.RBF(input_dim=self.emu.dim,
                               variance=remaining_variance,
                               lengthscale=np.ones(self.emu.dim), ARD=True)

        kernel2 = GPy.kern.Matern32(input_dim=self.emu.dim, ARD=True,
                                    variance=remaining_variance,
                                    lengthscale=np.ones(self.emu.dim))

        kernel3 = GPy.kern.White(input_dim=self.emu.dim,
                                 variance=remaining_variance)

        kernel = kernel1 + kernel2 + kernel3

        self.emu.gpr = \
            Gpy.models.GPRegression(self.X, self.normalized_Y, kernel)
        
        # '' is a regex matching all parameter names
        self.emu.gpr.constrain_positive('')
        self.emu.gpr.optimize()
        
    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.deltas = 
        print("Errors computed!")

    def save(self, file_handle):
        pickle.dump(self, open(name, "wb"), protocol=5)
