from evolmap.lhc import generate_samples
from collections import OrderedDict
from evolmap.camb_interface import transfer
import numpy as np
from evolmap.tkgrid import TransferGrid

ranges = OrderedDict([('obh2', [0.015, 0.03]),
                      ('och2', [0.08, 0.17])])

params = generate_samples(ranges, 10, 3, False)

kh = np.logspace(-4.9, 1.0, num=200)

transfers = []

for i in range(0, len(params)):
    Tk = transfer(kh, params[i, 0], params[i, 1])
    transfers.append(Tk)

tgrid = TransferGrid(kh, {"obh2": params[:, 0], "och2": params[:, 1]},
                     transfers)

tgrid.save('../Tk_tables.fits')


# plt.xlabel(r'$k\, [\rm Mpc]^{-1}$', fontsize=16)
# plt.ylabel(r'$T(k)$', fontsize=16)
# plt.title('Matter transfer functions')
# plt.savefig('../plots/test_transfer.png')
