import h5py as h5
import numpy as np
from mpi import size, rank, comm
from constants import Ry2eV, eRy, eps0, tol5, au2muAdivV2, au2pmdivV, a2bohr


class exciton:
    """
    Exciton properties:

    energy, envelop functions,
    exciton-phonon coupling matrix elements

    """
    def __init__(self, nb=-1, loadevecs=True,low_comm=False, fname='eigenvectors.h5'):
       """
       Initialize exciton properties

       """
       self.nb = nb
       self.loadevecs = loadevecs
       self.low_comm = low_comm
       self.init_from_file(fname)

       return

    def init_from_file(self, fname):

       if rank == 0:
         print('\n  Initializing exciton from file:', fname)

       with h5.File(fname, 'r') as f:
         # !! Be careful that Q-shift is NOT exciton COM
         self.Qpts = f.get('exciton_header/kpoints/exciton_Q_shifts')[()]
         #self.Qpts = f.get('exciton_header/kpoints/Qpts')[()]
         self.kpts = f.get('exciton_header/kpoints/kpts')[()]
         self.nfk = f.get('exciton_header/kpoints/nk')[()]
         self.nevecs = f.get('exciton_header/params/nevecs')[()]
         if self.nb < 0:
           self.nb = self.nevecs
         else:
           self.nevecs = self.nb
           
         self.nvb = f.get('exciton_header/params/nv')[()]
         self.ncb = f.get('exciton_header/params/nc')[()]
         self.xdim = f.get('exciton_header/params/bse_hamiltonian_size')[()]
            
         self.blat = f.get('mf_header/crystal/blat')[()]
         self.bvec = f.get('mf_header/crystal/bvec')[()]

         # eigenvalues in eV 
         self.evals = f.get('exciton_data/eigenvalues')[:self.nb]/Ry2eV
         self.celvol = f.get('mf_header/crystal/celvol')[()]
         self.nfk = f.get('mf_header/kpoints/nrk')[()]


       self.distribute_workload()

       if not self.loadevecs:
         f = h5.File(fname,'r')
         self.evecs = f['exciton_data/eigenvectors']
       else:

         with h5.File(fname, 'r') as f:
           if rank == 0:
             mem = self.xdim*self.nevecs*16*2/1e9
             print('\n  Estimate memory needed for loading excitons {0:6.3e} (GB)'.format(mem))
           # FIX!! ignore spin, assume nQ=1
           # leave eigenvectors in the matrix form
           # evecs[nQ, nevecs, nk, nc, nv, ns, :]
           if self.low_comm:
             tmp = f.get('exciton_data/eigenvectors')[0,:self.nb]
             self.evecs = tmp[...,0,0] + 1j*tmp[...,0,1]
           else:
            if self.my_nxct > 0:
             i0 = self.my_xcts[0]
             i1 = self.my_xcts[-1] + 1
             tmp = f.get('exciton_data/eigenvectors')[0,i0:i1]
             self.evecs = tmp[...,0,0] + 1j*tmp[...,0,1]
            else:
             self.evecs = np.empty([0])

       return


    def distribute_workload(self):

       max_nkpt_per_worker = int( np.ceil(self.nevecs *1.0 / size) )

       self.my_xcts = list()

       for i in range(max_nkpt_per_worker):

          ikpt = rank * max_nkpt_per_worker + i

          if ikpt >= self.nevecs:
              break

          self.my_xcts.append(ikpt)

       self.my_nxct = len(self.my_xcts)

       self.owners = np.zeros([self.nevecs], dtype=int)
       # record which process got which xct
       for i in range(max_nkpt_per_worker):

         for j in range(size):
          ikpt = j * max_nkpt_per_worker + i

          if ikpt >= self.nevecs:
              break

          self.owners[ikpt] = j

       #for i in range(size):
       #  if i == rank:
       #    print('rank', rank, self.my_xcts)

       #if rank == 0:
       #    print('owners table', self.owners)

       return
