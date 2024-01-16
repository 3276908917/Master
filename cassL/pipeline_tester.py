from cassL import camb_interface as ci
from cassandralin import emulator_interface as ei
from cassandralin import simple_tests as st

"""
We want a hypercube varying the following:
* omega_b
* omega_c
* n_s
* A_s
* omega_nu
* wa
* w0
* omega_K
* h

9-dimensional LHC for the massive case. 5000 samples should be enough, I wager.
8-dimensional for the massless: we keep A_s since that is still an input, even
    though it only functions as an evolution parameter.

Two sets of experiments: full priors, 2% smaller priors
1. CAMB test set, 5000 samples, full COMET_PLUS
2. CAMB test set, 5k samples, 98% COMET_PLUS
3. ", full MASSLESS
4. ", 98% MASSLESS

Then these four data sets can be re-used for all of the emulators we want to
build:
1. 5k massive
2. 4k massive
3. 3k massive
4. 5k massless
5. 4k massless
6. 3k massless

At this point, we'll want to research even smaller emulators:
Decrease by increments of 500 samples, maybe even 250...

What are the constraints with wa and w0? w + wa <= 0
"""

ei.get_Pk_interpolator()

# Automatically offer the user both the full and 98% results at once
def test(nu_massive=False):
        """
        This function can also be used to generate training-error curves,
        simply pass in the training X and Y again, this time as the test X and
        Y for this function.
        """

        test_set_cosmos = 23 # np.load
        test_set_spectra = 23 # np.load
        
        

        if len(X_test) != len(Y_test):
            raise ValueError("Size of X and Y do not match!")
        if len(X_test[0]) != self.p_emu.xdim:
            raise ValueError("Dimension of test X does not match the " + \
                "dimension of the training X!")
        if len(Y_test[0]) != self.p_emu.ydim:
            raise ValueError("Dimension of test Y does not match the " + \
                "dimension of the training Y!")
        
        self.X_test = X_test
        self.Y_test = Y_test

        test_predictions = np.zeros(Y_test.shape)

        for i in range(len(X_test)):
            test_predictions[i] = self.p_emu.predict(X_test[i])

        self.deltas = test_predictions - Y_test
        self.sq_errors = np.square(self.deltas)
        self.rel_errors = self.deltas / Y_test
        
        try:
            delta_predictions = np.zeros(Y_test.shape)
            for i in range(len(X_test)):
                delta_predictions[i] = self.delta_emu.predict(X_test[i])
            
            self.unc_deltas = delta_predictions - self.deltas
            self.unc_sq_errors = np.square(self.unc_deltas)
            self.unc_rel_errors = self.unc_deltas / self.deltas
        except AttributeError:
            pass
 
        print("Errors computed!")
        print("Sum of squared errors across all models:",
              sum(sum(self.sq_errors))) 
