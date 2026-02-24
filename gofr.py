#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
from qharv.field import sugar, kyrt
from qharv.cross.pwscf import cart2polar
from qharv.seed import hamwf_h5
from qharv.inspect import volumetric
import pandas as pd
import os
import scipy.io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from pathlib import Path
import glob
from numpy import matlib as mb
import numpy.ma as ma
import grscript

axfont=16
legfont=14
titlefont=16

''' 
pick out simulations with largest timestep, and calculate g(r) for each simulation (for phase diagram)
To do: 
1) calculate g(r) for all datapoints on phase diagram 
2) fit 1D cuts to Gaussians and extract means + variances
3) plot max(g(r)) or even just plots of g(r) arranged in order along two axes (eta and l)
a) plot electron separation distance, keeping track of ancestry
'''
def Calc_gofr_PD(folder,nx=16):
  filename = 'DMC*.csv' #only look for sims that successfully completed running, i.e. have an existing csv file in addition to an h5 file
  results = np.array(glob.glob(os.path.join(folder,filename)),dtype=object)
  simlens = np.array([int(fname.split('nstep')[1].split('_')[0]) for fname in results])
  results = results[simlens == max(simlens)]
  results = [os.path.splitext(csv)[0] + '.h5' for csv in results]
  matching = [s for fname in results for s in fname.split('_') if "seed" in s]
  seeds = [int(s.split('seed')[1]) for s in matching]
  nevery = h5py.File(results[0], 'r')['meta/arrstep'][0] #how frequently the position arrays are stored
  #l = h5py.File(results[0], 'r')['meta/l'][0]
  print(nevery)
  for csv,seed in zip(results,seeds):
      print(csv)
      fh5 = os.path.splitext(csv)[0] + '.h5'
      grscript.Calc_3D_gofr(fh5,nx,nevery,seed,generate=False)

def Plot_1Dgofr_PD(folder,num=3,xs=[],ys=[],cols=['eta','l']):
  '''
  plot 1D g(r) plots along a slice of the phase diagram
  '''
  def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # now find the indices of all values 
    print(array[idx])
    return (array == array[idx])

  filename = 'DMC*.csv' #only look for sims that successfully completed running, i.e. have an existing csv file in addition to an h5 file
  results = np.array(glob.glob(os.path.join(folder,filename)),dtype=object)
  simlens = np.array([int(fname.split('_nstep')[1].split('_')[0]) for fname in results])
  results = results[simlens == max(simlens)]
  results = [os.path.splitext(csv)[0] + '.h5' for csv in results]
  etas = np.array([float(fname.split('_%s' %cols[0])[1].split('_')[0]) for fname in results])
  ls = np.array([float(fname.split('_%s' %cols[1])[1].split('_')[0]) for fname in results])
  
  print(etas.min(),etas.max())
  print(ls.min(),ls.max())

  if len(xs) == 0:
    xs = np.linspace(etas.min(),etas.max(),num)
  if len(ys) == 0:
    ys = np.linspace(ls.min(),ls.max(),num)
  
  for x in xs:
    # find closest eta (x) value that I have to the written value
    idx = find_nearest(etas,x)   
    for y in ys:
      print('requested: (',x,y,')')
      # find closest l (y) value that I have to the writte
      idy = find_nearest(ls,y)   
      foundid = np.where(idx & idy)[0][0]
      print('found: (',etas[foundid],ls[foundid],')')
      print(results[foundid])
      grscript.Plot_2D_gofr([results[foundid]])
       
  return

if __name__ == '__main__':
  files = sys.argv[1:]
  Plot_1Dgofr_PD(sys.argv[1],xs=[0],ys=[5],num=1)
  #Calc_gofr_PD(sys.argv[1])
