import numpy as np
import pylab as pb
import GPy
import copy as cp
import pickle

from cassL import camb_interface as ci
from cassL import generate_emu_data as ged
from cassL import user_interface as ui

def eliminate_unusable_entries(X_raw, Y_raw,
    bad_X_vals = [float('-inf'), float('inf'), None, True, np.nan],
    bad_Y_vals = [float('-inf'), float('inf'), None, True, np.nan, 0]:
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