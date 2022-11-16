import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

import copy

def loadData(ns = 50, N_train = 70, N_test = 29, cov_len = 126):
    """
    Load the covariance matrices
    """
    name_root = "samples/ns{}/xiell_cov_noisy_ns{}_".format(ns, ns)
    
    train_data = np.zeros((N_train, cov_len, cov_len))
    test_data = np.zeros((N_test, cov_len, cov_len))
    
    for i in range(N_train):
        train_data[i] = np.loadtxt(name_root+"{:04d}.dat".format(i+1))
    for i in range(N_test):
        test_data[i] = np.loadtxt(name_root+"{:04d}.dat".format(N_train+i+1))
    
    return train_data, test_data

def loadData_tri(ns = 50, N_train = 60, N_val = 20, N_test = 19, cov_len = 126):
    """
    Load the covariance matrices
    """
    name_root = "samples/ns{}/xiell_cov_noisy_ns{}_".format(ns, ns)
    
    train_data = np.zeros((N_train, cov_len, cov_len))
    val_data = np.zeros((N_val, cov_len, cov_len))
    test_data = np.zeros((N_test, cov_len, cov_len))
    
    for i in range(N_train):
        train_data[i] = np.loadtxt(name_root+"{:04d}.dat".format(i+1))
    for i in range(N_val):
        val_data[i] = np.loadtxt(name_root+"{:04d}.dat".format(N_train+i+1))
    for i in range(N_test):
        test_data[i] = np.loadtxt(name_root+"{:04d}.dat".format(N_train+N_val+i+1))
    
    return train_data, val_data, test_data

def preprocess_cov(array, theory_cov):
    array = copy.deepcopy(array)
    cov_len = len(theory_cov)
    for i in range(cov_len):
        for j in range(cov_len):
            array[:, i, j] /= np.sqrt(theory_cov[i, i] * theory_cov[j, j])
    array = np.reshape(array, (len(array), cov_len, cov_len, 1))
    return array

def preprocess_alpha(covs, theory_cov):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    covs_norm = []
    for cov in covs:
      theory_diag = np.diagonal(theory_cov)
      cov_ii = cov/np.sqrt(theory_diag)
      cov_jj = np.transpose(cov_ii)/np.sqrt(theory_diag)
      covs_norm.append(np.transpose(cov_jj))

    return np.array(covs_norm)

def preprocess_theory(theory_cov, N = 70):
    cov_len = len(theory_cov)
    array = np.zeros((N, cov_len, cov_len))
    for i in range(cov_len):
        for j in range(cov_len):
            array[:, i, j] = theory_cov[i, j] / np.sqrt(theory_cov[i, i] * theory_cov[j, j])
    array = np.reshape(array, (len(array), cov_len, cov_len, 1))
    return array
