"""

Mock up a T(k) versus cparam
grid for test purposes

"""

from numpy import logspace
from evolmap.tkgrid import TransferGrid


def generate_mock_grid(fnout):
    """
    Generate a mock TransferGrid
    and save it to a file. The mock
    data is not any sort of realistic
    model, it's just to check the mechanics
    of the API.
    """

    # Some made up data
    omh2 = [0.1, 0.2, 0.3]
    obh2 = [0.05, 0.06, 0.07]
    ks = logspace(-2, 1, 100)

    transfers = []
    for tomh2, tobh2 in zip(omh2, obh2):
        # Just some mock data that has some dependence
        # on the parameters, this model makes no
        # physical sense.
        transfers.append((tomh2 + tobh2) + ks)

    tgrid = TransferGrid(ks, {"omh2": omh2, "obh2": obh2},
                         transfers)
    tgrid.save(fnout)

    tgrid2 = TransferGrid.load(fnout)
    print(tgrid2.params)


if __name__ == "__main__":
    generate_mock_grid("test_tk_grid.fits")
