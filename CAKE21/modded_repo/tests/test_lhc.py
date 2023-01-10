"""

Tests for the latin hypercube module

"""
import pytest
from numpy import isfinite
from collections import OrderedDict
from evolmap.lhc import generate_samples


@pytest.mark.parametrize("n_samples", [100, 200, 300])
@pytest.mark.parametrize("n_trials", [1, 10, 20])
@pytest.mark.parametrize("validation", [False, True])
def test_generate_samples(n_samples, n_trials, validation):
    """
    Check generate samples returns reasonable results

    """
    ranges = OrderedDict([("obh2", [0.005, 0.05]),
                          ("och2", [0.05, 0.4])])

    hpc = generate_samples(ranges, n_samples, n_trials=n_trials,
                           validation=validation)

    assert all(isfinite(hpc.flatten()))
