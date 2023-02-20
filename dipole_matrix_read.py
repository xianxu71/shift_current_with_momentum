import numpy as np

def read_eh_dipole(nxct):
    eh_diple = np.empty([nxct,3], dtype=complex)

    f1 = np.loadtxt('eigenvalues_b1.dat', skiprows=4, max_rows = nxct)
    f2 = np.loadtxt('eigenvalues_b2.dat', skiprows=4, max_rows = nxct)
    f3 = np.loadtxt('eigenvalues_b3.dat', skiprows=4, max_rows = nxct)

    data1 = f1[:,2] + 1j * f1[:,3]
    data2 = f2[:, 2] + 1j * f2[:, 3]
    data3 = f3[:, 2] + 1j * f3[:, 3]

    eh_diple[:,0] = data1
    eh_diple[:,1] = data2
    eh_diple[:,2] = data3
    print('test')
    return eh_diple