"""

Generate T(k) using Python's CAMB module

References
----------

`CAMB python documentation <https://camb.readthedocs.io/en/latest/>`_

"""

import camb
from camb import model
from scipy.interpolate import interp1d


def transfer(kh, ombh2, omch2):
    """
    Set a CAMBparams object by specifying ombh2 and omch2.
    The values for H0 and  ns are arbitary, z=0.
    Call CAMB to compute transfer function.
    Interpolate to input k-value

    Parameters
    ----------
    kh : float
        Input k-value in 1/Mpc
    ombh2:float
        Physical baryon density
    omch2: float
        Physical CDM density

    Returns
    -------
    delta : float
        The value for the transfer function
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params()
    pars.set_matter_power(redshifts=[0], kmax=10.)

    results = camb.get_results(pars)

    trans = results.get_matter_transfer_data()

    kh_camb = trans.transfer_data[0, :, 0]
    delta_camb = trans.transfer_data[model.Transfer_tot-1, :, 0]

    delta = interp1d(kh_camb, delta_camb, kind='cubic')

    return delta(kh)
