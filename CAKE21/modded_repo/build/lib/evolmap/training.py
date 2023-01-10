"""

Module for training the emulator

"""

import numpy as np
import GPy
import pickle


class EmulatorException(Exception):
    """
    Class for Exceptions in the
    training module

    """
    pass


class Emulator():
    """
    Class for training and using the emulator

    """
    def __init__(self):
        """
        Initializer of the class
    Parameters
    ----------
    ranges : collections:OrderedDict
        an ordered dictionary, the order
        of the outputs in latin hypercube
        is set by this order
    n_samples : int
        The number of samples to generate
    n_trials : int
        Regenerate cube this number of times
        to find the cube that maximises the
        minimum distance between samples
    validation : bool (Optional)
        Generate random verification data

    Returns
    -------
    samples : 2d array
        the latin hypercube
    """
        self.params = None
        self.nsample = None
        self.ndim = None
        self.kvec = None
        self.table = None
        self.table_transformed = None
        self.emu = None

    def load_training_set(self, params, table):
        """
        Load parameter sample and corresponding table

        Loads the parameter sample and the corresponding table,
        and stores them as class attributes. Transforms the table
        and stores it as a class attribute.

        Parameters
        ----------
        params: numpy.ndarray
            2-d array containing the shape parameters, with the first
            and second dimensions corresponding to the size of the sample
            and the size of the parameter space, respectively.
        table: numpy.ndarray
            2-d array containing the table to emulate, with the first and
            second dimension corresponding to the size of the sample and
            the numbers of bins at which the table is evaluated, respectively.
        """
        self.params = params
        self.nsample = params.shape[0]
        self.ndim = params.shape[1]
        self.table = table
        self._transform_table()

    def _transform_table(self):
        """
        Transform the table

        Transforms the table to reduce the dynamical range of the quantity
        to emulate. Stores the mean, standard deviation, and the actual
        transformed table as class attributes.
        """
        temp = np.log10(self.table)
        self.tmean = np.mean(temp, axis=0)
        self.tstd = np.std(temp, axis=0)
        self.table_transformed = (temp - self.tmean) / self.tstd

    def _GPy_model(self):
        """
        Call GPy Regression model

        Defines covariance kernel and calls internal regression method,
        specifying the parameter sample and the table to emulate.

        Returns
        -------
        reg_model: GPy.models.GPRegression
            An instance of a GPy.models.GPRegression object, trained
            using the class attributes self.params and self.table_trasnformed
        """
        kernel = GPy.kern.RBF(input_dim=self.ndim,
                              variance=np.var(self.table_transformed),
                              lengthscale=np.ones(self.ndim),
                              ARD=True)
        reg_model = GPy.models.GPRegression(self.params,
                                            self.table_transformed,
                                            kernel)
        return reg_model

    def train_emu(self, max_f_eval=1000, num_restarts=5):
        """
        Train the emulator

        Trains the emulator by first calling the _GPy_model method
        and subsequently optimizing it. Stores the final emulator as
        a class attribute.

        Parameters
        ----------
        max_f_eval: int
            Maximum number of function evaluations
        num_restarts: int
            Number of restarts for the training
        """
        emu = self._GPy_model()
        emu.optimize(max_f_eval=max_f_eval)
        emu.optimize_restarts(num_restarts=num_restarts)
        self.emu = emu

    def save_emu(self, filepath):
        """
        Save the emulator

        Save the trained emulator to file, using the pickle format.
        If the class attribute corresponding to the emulator is None,
        raises an exception.

        Parameters
        ----------
        filepath: str
            Name of output file

        Raises
        ------
        EmulatorException: if the class attribute for the emulator
        object is None
        """
        if self.emu is None:
            raise EmulatorException('Emulator has not been trained yet')
        pickle.dump(self.emu, filepath)

    def load_emu(self, filepath):
        """
        Load the emulator

        Load a previously trained emulator from file, and assigns it to the
        corresponding class attribute.

        Parameters
        ----------
        filepath: str
            Name of input file containing the emulator
        """
        self.emu = pickle.load(open(filepath))

    def predict(self, param):
        """
        Evaluate the emulator

        Evaluates the emulator on a new position of the parameter space, and
        properly rescales the output to transform back to the original table

        Parameters
        ----------
        param: np.ndarray
            1-d array containing a new position in the parameter space

        Returns
        -------
        pred: np.ndarray
            1-d array containing the predictions of the emulator on param
        """
        pred = self._transform_inv(self.emu.predict(param)[0][0])
        return pred

    def _transform_inv(self, table):
        """
        Transform back to original table

        Transforms the input table back to the original scale of self.table

        Parameters
        ----------
        table: np.ndarray
            1-d array containing a rescaled table

        Returns
        -------
        orig_table: np.ndarray
            1-d array containing the rescaled-back table
        """
        orig_table = 10**(table*self.tstd + self.tmean)
        return orig_table
