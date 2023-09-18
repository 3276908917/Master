import numpy as np
import matplotlib.pyplot as plt

import GPy
import copy as cp
import pickle

from cassL import camb_interface as ci
from cassL import generate_emu_data as ged
from cassL import user_interface as ui
from cassL import utils

constructor_complaint = "emu objects require either two NumPy arrays " + \
    "(the data sets) or an emu file handle."

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
    if len(X_raw) != len(Y_raw):
        raise ValueError("X and Y must have the same length")
    
    def unusable(row, bad_vals):
        for bad_val in bad_vals:
            if bad_val in row:
                return True
            if isinstance(bad_val, float) and np.isnan(bad_val) and \
                True in np.isnan(row):
                return True
        return False
        
    bad_row_numbers = np.array([], dtype=np.int64)
    
    for i in range(len(X_raw)):
        row_x = X_raw[i]
        row_y = Y_raw[i]
        if unusable(row_x, bad_X_vals) or unusable(row_y, bad_Y_vals):
            bad_row_numbers = np.append(bad_row_numbers, i)
   
    if len(bad_row_numbers) > 0:
        cleaned_X = np.delete(X_raw, bad_row_numbers, 0)
        cleaned_Y = np.delete(Y_raw, bad_row_numbers, 0)
        return cleaned_X, cleaned_Y
    
    return X_raw, Y_raw

def _normalize_spectra(Y):
    Ylog = np.log(Y)
    ymu = np.mean(Ylog, axis=0)
    Y_shifted = np.subtract(Ylog, ymu)
    ystdev = np.std(Ylog, axis=0)
    Y_normalized = np.divide(Y_shifted, ystdev)
    return Y_normalized, ymu, ystdev

def is_normalized_X(X):
    # By default, np.min and np.max, without an explicit axis, will check every
    # value in the 2D array X
    return np.min(X) >= 0 and np.max(X) <= 1


class Emulator_Trainer:

    ### To-do:
    # this should automatically handle different kinds of sampling, i.e. in
    # sigma_{12}: square linear or root linear stuff. Right now the handling is
    # incomplete.

    class Emulator:

        def __init__(self, name, xmin, xrange, ymu, ystdev):
            # We're requiring all emulators to follow the same approach:
            # all mappings can be represented with just the two variables,
            # xmin and xrange
            self.name = name

            self.xmin = xmin
            self.xrange = xrange
            self.xdim = len(xmin)

            self.ymu = ymu
            self.ystdev = ystdev
            self.ydim = len(ymu)

        def convert_to_normalized_params(self, config):
            """
            Using the originally input prior, convert a particular cosmological
            configuration into a set of unit-LHC coordinates.
            """
            if len(config) != self.dim:
                raise ValueError("This is a " + str(self.dim) + \
                    "-dimensional emulator. Input vector has " + \
                    len(config) + " dimensions.")

            for i in range(len(config)):
                if config[i] < self.xmin[i] or \
                    config[i] > self.xmin[i] + self.xrange[i]:
                    raise ValueError("Parameter at index " + str(i) + \
                        "is outside of the range defined by the prior!")

            return (config - self.xmin) / self.xrange


        def inverse_ytransform(self, raw_prediction):
            """
            We'll have to write somewhere in the thesis that our emulators
            currently only support log normalization. i.e. this "np.exp" call
            is currently fixed.
            """
            return np.exp(raw_prediction * self.ystdev + self.ymu)


        def _predict_normalized_pspectrum(self, x):
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


        def predict_pspectrum(self, X):
            # Instead of solving the x formatting complaints by blindly
            # re-nesting x, let's try to get to the bottom of *why* the
            # formatting is so messed up in the first place.

            # I think the reason we need to nest is because the prediction
            # expects a collection of x, not just a single x point!
            if not isinstance(X[0], np.ndarray):
                X = np.array([X]) # nesting necessary to evaluate single config

            if not is_normalized_X(X):
                raise ValueError("The input parameters are not correctly " + \
                                 "normalized. Have you used " + \
                                 "convert_to_normalized_params?")

            normalized_pspec = self._predict_normalized_pspectrum(X)
            return self.inverse_ytransform(normalized_pspec)


    def __init__(self, *args):
        """
        There are two ways one can go about instantiating:
        1. provide an X and Y data set as well as a set of priors -> emulator i
            trained
            
            IMPORTANT: the X and Y sets should not contain any unusable entries
                (when in doubt, use eliminate_unusable_entries). Furthermore,
                the Y set should not already be normalized, because the
                normalization information (i.e. mean and standard deviation of
                the log'ed spectra) becomes part of the emulator's prediction
                pipeline!
            
        2. provide the file handle associated with a trained emulator object
        """
        if len(args) == 1:
            # We're expecting an already-trained emulator object
            file_handle = args[0]
            if not isinstance(file_handle, str):
                raise TypeError(constructor_complaint)

            self.emu = pickle.load(open(file_handle, "rb"))

        elif len(args) == 4:
            emu_name = args[0]
            self.X_train = args[1]
            self.Y_train = args[2]
            self.priors = args[3]
            
            if not isinstance(self.X_train, np.ndarray):
                raise TypeError(constructor_complaint)
            
            if not isinstance(self.Y_train, np.ndarray):
                raise TypeError(constructor_complaint)
            
            if not isinstance(self.priors, dict):
                raise TypeError(constructor_complaint)
            
            if len(self.X_train) != len(self.Y_train):
                raise ValueError("X and Y are unequal in length!")
                
            self.normalized_Y, ymu, ystdev = _normalize_spectra(self.Y_train)

            xmin = np.array([])
            xrange = np.array([])

            for key in self.priors.keys():
                xmin = np.append(xmin, self.priors[key][0])
                xrange = np.append(xrange,
                                   self.priors[key][1] - self.priors[key][0])

            self.emu = self.Emulator(emu_name, xmin, xrange, ymu, ystdev)
        else:
            raise TypeError(constructor_complaint)
            
            
    def train(self):
        if self.X_train is None or self.normalized_Y is None:
            raise AttributeError("No data found over which to train.")

        # The dimension is automatically the length of an X element.
        remaining_variance = np.var(self.normalized_Y)

        kernel1 = GPy.kern.RBF(input_dim=self.emu.xdim,
                               variance=remaining_variance,
                               lengthscale=np.ones(self.emu.xdim), ARD=True)

        kernel2 = GPy.kern.Matern32(input_dim=self.emu.xdim, ARD=True,
                                    variance=remaining_variance,
                                    lengthscale=np.ones(self.emu.xdim))

        kernel3 = GPy.kern.White(input_dim=self.emu.xdim,
                                 variance=remaining_variance)

        kernel = kernel1 + kernel2 + kernel3

        self.emu.gpr = \
            GPy.models.GPRegression(self.X_train, self.normalized_Y, kernel)
        
        # '' is a regex matching all parameter names
        self.emu.gpr.constrain_positive('')
        self.emu.gpr.optimize()
        
    def test(self, X_test, Y_test):
        """
        This function can also be used to generate training-error curves,
        simply pass in the training X and Y again, this time as the test X and
        Y for this function.
        """
        if len(X_test) != len(Y_test):
            raise ValueError("Size of X and Y do not match!")
        if len(X_test[0]) != self.emu.xdim:
            raise ValueError("Dimension of test X does not match the " + \
                "dimension of the training X!")
        if len(Y_test[0]) != self.emu.ydim:
            raise ValueError("Dimension of test Y does not match the " + \
                "dimension of the training Y!")
        
        self.X_test = X_test
        self.Y_test = Y_test

        self.test_predictions = np.zeros(Y_test.shape)

        for i in range(len(X_test)):
            #! This might complain about lack of nesting, but don't just nest,
            # try to find a better way to resolve the issue.
            self.test_predictions[i] = self.emu.predict_pspectrum(X_test[i])

        self.deltas = self.test_predictions - Y_test
        self.sq_errors = np.square(self.deltas)
        self.rel_errors = self.deltas / Y_test

        print("Errors computed!")
        print("Sum of squared errors across all models:",
              sum(sum(self.sq_errors)))

    def set_scales(self, scales):
        if len(scales) != self.emu.ydim:
            raise ValueError("The dimension of the given set of scales " + \
                "does not match the dimension of the spectra!")
        
        self._scales = scales

    def enforce_error_calculation(self):
        if not hasattr(self, "deltas"):
            raise AttributeError("Errors have not been computed yet! Use " + \
                " the function 'test'")

    def error_curves(self, deltas=False, plot_every=1, param_index=None,
        param_label=None, param_range=None, fixed_k=None, save_label=None):
        """
        If param_index is None, all error curves are plotted together and in
        the same color.

        If fixed_k is None, the whole set of scales is plotted together, and
        the param specified by param_index becomes the new x-axis. A non-None
        fixed_k value only makes sense with a non-None
        param_index, so the function will complain if this criterion is unmet.

        Thanks to Dante for the recommendation of the fixed_k functionality!
        """
        self.enforce_error_calculation()
        
        if not hasattr(self, "_scales"):
            raise AttributeError("This object has no k values to which " + \
            "the spectra correspond! Specify them with set_scales")

        if (param_index is None) ^ (param_label is None): 
            raise ValueError("If a param_index is given, a param_label " + \
                "must also be given, and vice versa.")
                
        if param_range is not None and param_index is None:
            raise ValueError("A parameter range was given, but no " + \
                "parameter was specified.")
        
        if fixed_k is not None and param_index is None:
            raise ValueError("A fixed-k plot is not possible without a " + \
                "replacement x-axis. Specify a cosmological parameter to " + \
                "represent the x coordinate.")

        k_index = None

        if fixed_k:
            # Issue a warning if we weren't able to find the exact k value.
            if fixed_k not in self._scales:
                k_index = utils.closest_index(self._scales, fixed_k)
                raise UserWarning("No exact match was found for the given " + \
                    "fixed k (" + str(fixed_k) + "). Approximating to " + \
                    str(self._scales[k_index]))
            else:
                k_index = self._scales.index(fixed_k)

        valid_indices = list(range(len(self.X_test[:, param_index])))
        if param_range is not None:
            valid_indices = np.where(np.logical_and(
                self.X_test[:, param_index] < param_range[1],
                self.X_test[:, param_index] > param_range[0]))[0]
        valid_vals = self.X_test[:, param_index][valid_indices]
        normalized_vals = utils.normalize(valid_vals)

        colors = plt.cm.plasma(normalized_vals)
 
        valid_errors = self.deltas[valid_indices] if deltas \
            else self.rel_errors[valid_indices]

        for i in range(len(valid_errors)):
            if i % plot_every == 0:
                if fixed_k:
                    # This approach might generate a bunch of meaningless
                    # colors. If it does, we should switch to building plot
                    # points and then plotting all of them together when the
                    # loop is complete.
                    plt.scatter(self.valid_vals[i], valid_errors[i][k_index])
                else:
                    if param_index:
                        plt.plot(self._scales, valid_errors[i],
                            color=colors[i], alpha=0.05)
                    else:
                        plt.plot(self._scales, valid_errors[i], alpha=0.05)
                    plt.xscale('log')

        title = "Emulator " + self.emu.name + ", " + \
            str(len(valid_errors)) + r" Random Massive-$\nu$ Models"

        if param_index:
            title += "\ncolored by " + param_label + " value"

        plt.title(title)
        plt.ylabel("% error between CAMB and CassL")
        plt.xlabel("scale $k$ [1 / Mpc]")

        if param_index:
            norm = mpl.colors.Normalize(vmin=min(valid_vals),
                                        vmax=max(valid_vals))
            plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm))

        if save_label is not None:
            plt.savefig("../plots/emulator/performance/" + save_label + ".png")

        plt.show()

    def error_statistics(self, deltas=False, error_aggregator=np.median):
        """
        Maybe this function, like error_curve, should include a parameter range
        constraint parameter.
        
        @error_aggregator: function
            This should be a numpy aggregator, because this function is always
            called with the argument "axis=1"
            Some recommendations:
                numpy.median
                numpy.max
                numpy.min
                numpy.mean
                numpy.ptp (i.e. the peak-to-peak, or range)
                numpy.std (i.e. standard deviation)
        """
        self.enforce_error_calculation()
        
        errors = self.deltas if deltas else self.rel_errors
        error_array = error_aggregator(errors, axis=1)
        
        print("The chosen error statistic...")
        print("ranges from", np.min(meds), "to", np.max(meds))
        print("median is", np.median(meds))
        print("mean is", np.mean(meds))
        print("st.dev. is", np.std(meds))

    def error_hist(self, deltas=False, error_aggregator=np.median,
                   title="Histogram of Median Relative Errors", bins=None):
        """
        Maybe this function, like error_curve, should include a parameter range
        constraint parameter.
        
        If bins is left as None, the histogram automically uses Sturges' rule.
        """
        self.enforce_error_calculation()
        
        errors = self.deltas if deltas else self.rel_errors
        error_array = error_aggregator(errors, axis=1)

        if bins == None:
            bins="sturges"

        plt.hist(error_array, bins=bins)
        plt.title(title)
        plt.ylabel("Frequency [counts]")

        if deltas:
            plt.xlabel("Delta between CAMB and Cassandra-L Power-spectrum")
        else:
            plt.xlabel("Relative error between CAMB and Cass-L P-spectrum")

        plt.savefig("plots/err_hist_" + self.emu.name + ".png")

        plt.show()

    def save(self, file_name=None):
        """
        If file_name is None, this function saves under the name of the
        emulator.
        """
        if file_name is None:
            file_name = self.emu.name
        if file_name[:-4] != ".cle":
            file_name += ".cle"
        pickle.dump(self.emu, open("emulators/" + file_name, "wb"), protocol=5)
