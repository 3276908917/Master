"""

Modules to deal the the i/o of tabulated
T(k) values on the latin hypercube

"""

from six import iteritems
from numpy import array
from astropy.io import fits


class TkgridWrongShapeException(Exception):
    pass


class TransferGrid(object):
    """
    Class for storing the T(k)
    values for given cosmological
    parameters

    Parameters
    ----------
    kbins : array
        the k-bins corresponding to
        the T(k), in 1/Mpc
    params : dict of arrays
        a dictionary of cosmological
        parameters and array of values.
        The arrays of values should
        be in the order of the T(k)
        rows
    tk2d : array
        a 2D array of k-bin and the
        T(k) values for each of the
        sets of cosmological parameters.
        The 2nd dimenstion should be
        k-bins, the 1st should be
        the different cosmologies.
    """

    def __init__(self, kbins, params, tk2d):

        self.kbins = kbins
        self.params = params
        self.tk2d = array(tk2d)

        if self.tk2d.shape[1] != len(self.kbins):
            raise TkgridWrongShapeException(
                "N k-bins does not match array size"
                )

    @classmethod
    def load(cls, filename):
        """
        Load a precomputed set of T(k)
        versus cosmological parameter

        Parameters
        ----------
        filename : str
            filename to load
        """
        with fits.open(filename) as hdul:
            tk2d = hdul["PRIMARY"].data
            kbins = hdul["KBINS"].data
            params = {}
            tab = hdul["CPARAMS"]
            for c in tab.columns:
                params[c.name] = tab.data[c.name]

        return cls(kbins, params, tk2d)

    def save(self, filename):
        """
        Save this object to
        a file

        Parameters
        ----------
        filename : str
            filename to save

        """
        hdu1 = fits.PrimaryHDU(self.tk2d)
        hdu2 = fits.ImageHDU(self.kbins, name="KBINS")

        column_list = []
        for key, value in iteritems(self.params):
            column_list.append(fits.Column(name=key, array=value, format='D'))

        hdu3 = fits.BinTableHDU.from_columns(column_list, name="CPARAMS")

        fits.HDUList([hdu1, hdu2, hdu3]).writeto(filename)
