#!/usr/bin/env python3
import numpy as np
from pyscf.pbc import gto as pbcgto
import pyqmc
from pyqmc.coord import PeriodicConfigs 
from pyqmc.accumulators import PGradTransform, LinearTransform
from pyqmc.wftools import generate_jastrow
from pyqmc.linemin import line_minimization
from egas import EnergyAccumulator
import os

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--rs', type=float, default=4)
  parser.add_argument('--diffusion', type=bool,default=False)
  parser.add_argument('--dt', type=float)
  parser.add_argument('--nconf', type=int, default=512)
  parser.add_argument('--verbose', action='store_true')
  parser.add_argument('--outdir',type=str, default='.')
  
  args = parser.parse_args()
  outdir = args.outdir
  diffusion = args.diffusion
  rs = args.rs  # inter-electron spacing, controls density
  nelec = 2
  ndim = 3
  lbox = (4*np.pi/3*nelec)**(1/3) * rs  # sys. size/length measured in a0; multiply by 2 since 2 = # of electrons

  axes = lbox*np.eye(ndim)
  pos = lbox*np.random.rand(args.nconf, nelec, ndim)

  # simulation cell
  cell = pbcgto.M(
    atom = 'He 0 0 0',
    a = axes,
    unit='B',  # B = units of Bohr radii
  )
  # ee Jastrow (only bcoeff for ee, no acoeff for ei)
  wf, to_opt = generate_jastrow(cell, ion_cusp=[], na=0)
  # initialize electrons uniformly inside the box
  configs = PeriodicConfigs(pos, axes)
  # use hacked energy in gradient estimator
  wf.diffusion = diffusion
  acc = EnergyAccumulator(cell)
  pgacc = PGradTransform(
    acc,
    LinearTransform(wf.parameters, to_opt)
  )
  # do Jastrow optimization
  line_minimization(
    wf,
    configs,
    pgacc,
    verbose=True,
    hdf_file=os.path.join(outdir,'opt-rs%d.h5' % (rs,)),
    max_iterations=10,
  )

if __name__ == '__main__':
  main()
