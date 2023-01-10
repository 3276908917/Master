"""

:Authors: Daniel Farrow; others
:Address: MPE

:abstract:

    Test the miscellaneous module

"""
import pytest
from evolmap.misc import python_rules


@pytest.mark.parametrize("x", [1.0, 2.0, "test", [1, 2]])
def test_python_rules(x):
    """
    Test this function returns 42,
    no matter what x is

    Notes
    -----
    This is just a demo function
    """
    y = python_rules(x)
    assert y == pytest.approx(42)
