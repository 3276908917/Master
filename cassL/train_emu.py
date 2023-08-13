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

def normalize_X(priors, ):
    # If we're given a unit hc, this function should transform external test
    # points...

class emulator:

    # priors
    # y normalization
    # X set is optional
    # Y set is optional
    # emu object

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

            self = pickle.load(open(file_handle), "rb")

        elif len(args) == 3:
            self.X = args[0]
            self.Y = args[1]
            self.priors = args[2]
            assert isinstance(self.X, np.ndarray) and \
                isinstance(self.Y, np.ndarray) and \
                isinstance(self.priors, dict), constructor_complaint

            self.normalized_Y = normalize_spoctra(self.Y)

        else:
            raise Exception(constructor_complaint)

    def train(self):
        assert self.normalized_X is not None and \
            self.normalized_Y is not None, "No data found over which to train."

        # The dimension is automatically the length of an X element.
        self.dim = len(self.X[0])
        remaining_variance = np.var(Y)

        kernel1 = GPy.kern.RBF(input_dim=self.dim, variance=remaining_variance,
                               lengthscale=np.ones(self.dim), ARD=True)

        kernel2 = GPy.kern.Matern32(input_dim=self.dim,
                                    variance=remaining_variance,
                                    lengthscale=np.ones(self.dim), ARD=True)

        kernel3 = GPy.kern.White(input_dim=self.dim,
                                 variance=remaining_variance)

        kernel = kernel1 + kernel2 + kernel3

        self.gpr = Gpy.models.GPRegression(self.X, self.Y, kernel)

    def save(self, file_handle):
        pickle.dump(self, open(name, "wb"), protocol=5)
