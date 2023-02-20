import numpy as np
from mpi import MPI, comm, size, rank
import IO
from math_functions import delta_lorentzian, delta_gaussian
from constants import Ry2eV, eRy, eps0, tol5, au2muAdivV2, au2pmdivV, a2bohr, hbar
from dipole_matrix_read import *
import time


class optical_responses:

   """
   Compute optical resposnses from first-principles inputs

   Using the sum-over-state method from the perturbation theory

   """

   def __init__(self, QP, w, eta, brdfun='Lorentzian',\
                tetra=False, exciton=None):
      """
      Initialize the calculation

      """
      self.exciton = exciton

      self.w = np.array(w[:])
      self.nw = len(self.w)
      self.eta = eta
      
      self.tetra = tetra

      # Note: tetrahedron supress broadening options
      if brdfun == 'Lorentzian':
        self.brdfunc = delta_lorentzian
      else:
        self.brdfunc = delta_gaussian
      return

   
   def calc_shift_current_with_eh(self, use_dipole_W90, dipole_W90, fname='xct-shift_current'):

       celvol = self.exciton.celvol
       nfk = self.exciton.nfk
       ncb = self.exciton.ncb
       nvb = self.exciton.nvb
       nevecs = self.exciton.nevecs
       evecs = self.exciton.evecs
       evals = self.exciton.evals
       peh = read_eh_dipole(self.exciton.nb)  # dim: (ns,3)

       eta = self.eta / Ry2eV
       wrange = self.w / Ry2eV
       brdf = self.brdfunc



       pref = -eRy ** 3 / (2 * nfk * celvol)/hbar**2
       pref2 = eRy ** 3 / (2 * nfk * celvol)/hbar**2


       sigma0 = np.zeros([3, 3, 3, len(wrange)], dtype=complex)

       for i in range(self.exciton.nevecs):

           if rank == 0 and i % np.ceil(self.exciton.nevecs / 10) == 0:
               print('  Progress: {0:4.1f}%'.format(i / self.exciton.nevecs * 100))

           if self.exciton.low_comm:
               peh_i = peh[i]
               evecs_i = evecs[i]
           else:
               if i in self.exciton.my_xcts:
                   i_loc = self.exciton.my_xcts.index(i)
                   evecs_i = self.exciton.evecs[i_loc]
                   peh_i = peh[i_loc]
                   if rank != self.exciton.owners[i]:
                       print(i, i_loc, rank, self.exciton.owners[i])
                       raise Exception('Wrong exciton owner')
               else:
                   evecs_i = np.empty([nfk, ncb, nvb], dtype=complex)
                   peh_i = np.empty([3], dtype=complex)


               evecs_i = np.ascontiguousarray(evecs_i)

               comm.Bcast(evecs_i, root=self.exciton.owners[i])
               comm.Bcast(peh_i, root=self.exciton.owners[i])

           for j_loc, j in enumerate(self.exciton.my_xcts):

               if self.exciton.low_comm:
                   evecs_j = evecs[j]
                   peh_j = peh[j]
               else:
                   evecs_j = evecs[j_loc]
                   peh_j = peh[j_loc]

               Pij = 1 # dim 3
               # Intraband velocity

               num1 = np.einsum('a,bc->abc', peh_i, \
                                np.einsum('b,c->bc', Pij, peh_j.conjugate()))
               num2 = num1.conjugate()

               num3 = np.einsum('b,ac->abc', peh_i.conjugate(), \
                                np.einsum('a,c->ac', Pij, peh_j))
               num4 = num3.conjugate()



               # Implementation 2
               sigma0 = sigma0 + pref * 1j * np.pi / evals[i] * ( \
                           -np.einsum('abc,w->abcw', num1, brdf(wrange, evals[j], eta))
                           + np.einsum('abc,w->abcw', num2, brdf(wrange, -evals[j], eta))
                           - np.einsum('acb,w->abcw', num1, brdf(wrange, -evals[j], eta))
                           + np.einsum('acb,w->abcw', num2, brdf(wrange, evals[j], eta))) \
                        + pref2 * 1j * np.pi * ( \
                                    +np.einsum('abc,w->abcw', num3,
                                               brdf(wrange, evals[j], eta) * (1. / (wrange - evals[i] + 1j * eta)).real)
                                    - np.einsum('abc,w->abcw', num3, brdf(wrange, evals[i], eta) * (
                                        1. / (wrange - evals[j] - 1j * eta)).real)
                                    - np.einsum('acb,w->abcw', num4, brdf(wrange, -evals[j], eta) * (
                                        1. / (wrange + evals[i] - 1j * eta)).real)
                                    + np.einsum('acb,w->abcw', num4, brdf(wrange, -evals[i], eta) * (
                                        1. / (wrange + evals[j] + 1j * eta)).real))

       sigma0 = comm.allreduce(sigma0)

       # write shift current spectrum
       if rank == 0:
           for i, d1 in enumerate(['x', 'y', 'z']):
               for j, d2 in enumerate(['x', 'y', 'z']):
                   for l, d3 in enumerate(['x', 'y', 'z']):
                       tmp = fname + '-' + d1 + d2 + d3 + '.txt'
                       IO.write_shiftcurrent(tmp, wrange * Ry2eV, sigma0[i, j, l] * au2muAdivV2)

       return
