import numpy as np
import exciton
import optical_responses
from mpi import MPI, comm, size, rank
import time

if __name__ == '__main__':

    nvband = 6
    ncband = 6
    nxct = 200

    seed_name = 'BTO'
    eqp_fname = 'eqp.dat'
    xct_fname = 'eigenvectors.h5'

    Xct = exciton.exciton(nb=nxct, low_comm=False, loadevecs=True, fname=xct_fname)

    wmin = 3.4
    wmax = 4.6
    dw = 0.2
    omega = np.arange(wmin, wmax + dw, dw)

    eta = 0.2
    tetra = False
    brdfun = 'Lorentzian'

    op = optical_responses.optical_responses(omega, eta, tetra=tetra, \
                                             brdfun=brdfun, exciton=Xct)
    start = time.time()

    op.calc_shift_current_with_eh()


    end = time.time()

    if rank == 0:
        print('time elapsed', end - start)