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

axfont=16
legfont=14
titlefont=16
def extract_walkers(fp, nevery=1000, nequil=2000,opt=False,minstep=0,maxstep=np.inf):
  posl = []
  weights = []
  if opt == True:
    dists = [] #distances between electrons
    ancestry = [] #keep track of each walker's history
  ct = 0
  # make sure that the keys themselves are in order, first
  steps = []

  arrstep = fp.get('meta/arrstep')[0,0]
  print('arrstep',arrstep)
  if nevery < arrstep: nevery = arrstep

  for key in fp.keys():
    if key.startswith('step'):
      istep = int(key[4:])

      if istep < nequil: continue #ignore t=0 because system hasn't equilibrated at that time
      if (istep < minstep) | (istep > maxstep): continue
      
      if (istep % nevery) == 0:
        steps.append(istep)
        pos1 = fp[key]['pos'][()]
        posl.append(pos1)
        wt = fp[key]['weight'][()]
        weights.append(wt)
        if opt == True:
          distarr = np.sqrt(np.sum((pos1[:,0,:]-pos1[:,1,:])**2,axis=1)) #Nw x 1 array
          #print(distarr.shape,list(fp[key].keys()))
          dists.append(distarr)
          if 'ancestor_indices' in list(fp[key].keys()):
            ancestors = fp[key]['ancestor_indices'][()]
          else:
            ancestors = ancestry[ct-1]
          ancestry.append(ancestors)
      ct = ct + 1

  sortid = np.argsort(steps)
  steps = np.array(steps)[sortid]
  posl = np.array(posl)[sortid,:,:,:] #Nt x Nw x Nelec x Ndim
  weights = np.array(weights)[sortid,:] #Nt x Nw
  if opt == True:
    distarr = np.array(distarr)[sortid]
    ancestry = np.array(ancestry)[sortid]
    return steps, posl, weights, distarr, ancestry
  else:
    return steps, posl, weights

def extract_phonons(fp, nevery=1000, nequil=2000,plotting=False):
  ''' 
    extract phonon amplitudes (complex) from simulation.
  ''' 
  ph_amps = [] 
  ancestors = []
  #Nt=int(Nsteps/arrstep)+1 #will erase 0 cols at end (keys are unsorted so it's easiest to first just shove everything into an array and then cut it down), so no -int(nequil/arrstep)

  #Nks = np.array(f.get('ks')).shape[0]   
  #farray = np.zeros((Nt,Nks),dtype=complex)
 
  if plotting == True:
    fig,ax = plt.subplots(1,1,figsize=(6,4.5))
    ax.set_xlabel('$|\\vec k|$')
    ax.set_ylabel('$|f_k|$')
    ct = 0
    

  ct = 0
  ks = np.array(fp.get('kmags'))
  for key in fp.keys():
    if key.startswith('step'):
      istep = int(key[4:])
      #if istep <= nequil: continue #ignore t=0 because system hasn't equilibrated at that time
      if (istep % nevery) == 0:
        f_ks = fp[key]['f_ks'][()] #Nk x Nw
        ph_amps.append(f_ks)
        if 'ancestor_indices' in fp[key].keys():
          ancs = fp[key]['ancestor_indices'][()]
        else:
          ancs = ancestors[ct-1]
        ancestors.append(ancs)
        ct = ct + 1
        if plotting == True:
          ax.plot(ks,np.mean(np.abs(f_ks),axis=1),'.')
          #fig.canvas.draw()
          #fig.canvas.flush_events()
          ct = ct + 1

  if plotting == True:
    plt.tight_layout()
    plt.show()

  return ks,np.array(ph_amps),np.array(ancestors)

def PlotPhononAmps(fh5,tequil=1500,walk=None):
  '''
    Plot phonon amplitudes averaged over all walkers OR for walker number 'walk'
  '''
  fp = h5py.File(fh5, 'r')
  arrstep = fp['meta/arrstep'][0][0]
  popstep = fp['meta/popstep'][0][0]
  tau = fp['meta/tau'][0][0]
  Nsteps = fp.get('meta/Nsteps')[0,0]
  save_phonons = fp['meta/save_phonons'][0][0]
  ts = np.arange(0,Nsteps+1,save_phonons)*arrstep*tau
  #save_phonons = 10*np.maximum(arrstep,popstep)
  ks, ph_amps,ancestors = extract_phonons(fp, nevery=save_phonons, nequil=int(tequil/tau), plotting=False)
  print(ks.shape,ph_amps.shape, ancestors.shape) #Nt x Nk x Nw?
  #ktile = np.tile(ks,(ph_amps.shape[-1],1))
  #ks = np.array(f.get('ks'))
  ktile = np.tile(ks,(ph_amps.shape[0],1))
  print(ktile.shape)
  y = np.mean(np.abs(ph_amps),axis=2)
  fig,ax = plt.subplots(1,3,figsize=(10,4.5))
  ax[0].plot(ktile.ravel(),y.ravel(),'.')#,label='$|f|, \\tau = %.2f$' %(tau,) )
  ax[0].set_xlabel('$|\\vec k|$')
  ax[0].set_ylabel('$<|f_k|>$ (avg over %d walkers)' %ph_amps.shape[-1])
  y = np.mean(ph_amps,axis=2) # Nt x Nk x Nw
  ax[1].plot(ktile.ravel(),y.real.ravel(),'.')#,label='$|f|, \\tau = %.2f$' %(tau,) )
  ax[1].set_xlabel('$|\\vec k|$')
  ax[1].set_ylabel('$\Re(<f_k>)$ (avg over %d walkers)' %ph_amps.shape[-1])
  ax[2].plot(ktile.ravel(),y.imag.ravel(),'.')#,label='$|f|, \\tau = %.2f$' %(tau,) )
  ax[2].set_xlabel('$|\\vec k|$')
  ax[2].set_ylabel('$\Im(<f_k>)$ (avg over %d walkers)' %ph_amps.shape[-1])

  if walk is not None:
    ax[1].plot(ktile.ravel(),ph_amps[:,:,walk].real.ravel(),'r.')
    ax[2].plot(ktile.ravel(),ph_amps[:,:,walk].imag.ravel(),'r.')
    print(ancestors[:,walk])
  fig.tight_layout()

  fig2,ax2 = plt.subplots(1,2,figsize=(10,4.5))
  if walk is not None:
    ax2[0].plot(ts,ph_amps[:,:,walk].real[:,10],label='$k_{10}$, w%d' %walk)
    ax2[1].plot(ts,ph_amps[:,:,walk].imag[:,10],label='$k_{10}$, w%d' %walk)
  else:
    ax2[0].plot(ts,y.real[:,10],label='$k_10$')
    ax2[1].plot(ts,y.imag[:,10],label='$k_10$')

  ax2[0].set_xlabel('sim time')
  ax2[1].set_xlabel('sim time')
  ax2[0].legend()
  ax2[1].legend()
  ax2[0].set_ylabel('$\Re(<f_k>)$')
  ax2[1].set_ylabel('$\Im(<f_k>)$')
  
  plt.show()

  '''
  flat_ph = ph_amps.flatten()
  flat_ks = ktile.flatten()
  fig,ax = plt.subplots(1,1,figsize=(6,4.5))
  ax.plot(flat_ks,np.abs(flat_ph),'.',label='$|f|, \\tau = %.2f$' %(tau,) )
  ax.set_xlabel('$|\\vec k|$')
  ax.set_ylabel('$|f_k|$')
  plt.tight_layout()
  plt.show()
  '''
  '''
    ktile = np.tile(ks,int(len(f_ks)/len(ks)))
    #Nks = np.array(ks).shape[0]   
    f_ks = f_ks.flatten()
    fig,ax = plt.subplots(1,1,figsize=(6,4.5))
    ax.plot(ktile,np.abs(f_ks),'.',label='$|f|, \\tau = %.2f$' %(tau,) )
    ax[0].plot(ks,np.abs(h_ks),'.',label='$|h|, \\tau = %.2f$' %(tau,) )
    ax[1].plot(ktile,np.abs(n_ks),'.',label='$\\tau=%.2f$' %(tau,))
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('$|n_k|$')
    ax[0].set_ylabel('count')
    ax[1].set_xlabel('$|\\vec k|$')
    ax[1].set_ylabel('Mom. density $|n_k|=|h_k^*f_k|$')
    fig.suptitle('$\eta=%.2f,\,U=%.2f,\,r_s = %d,\, N_k = %d$' %(eta,2*l,r_s,len(ks)))
    plt.tight_layout()
    plt.show()
  '''

def calculate_displacements(posa, lbox):
  '''Calculate separation distance r12 = r1-r2, and subtract off multiples of box length to find resulting displacements within a single cell
  posa: Nw x Nelec x Nt array
  
  returns: 
  Nw x 3 x Nt array of electron separation distances calculated for periodic box size lbox (for each direction)
  '''
  frames = []
  for iconf, walkers in enumerate(posa):
    xyz = (walkers[:, 0] - walkers[:, 1])/lbox
    xyz = xyz - np.rint(xyz)
    frames.append(lbox*xyz)
  return np.array(frames)

def calc_dists_in_box(h5,nevery=1,nequil=0):
  L = h5.get('meta/L')[0,0]
  dt, data = sugar.time(extract_walkers)(h5, nevery=1,nequil=0)
  ts,posa, wts = data
  nconf, nwalker, nelec, ndim = posa.shape
  msg = 'extracted %d frames in %.4f s' % (nconf, dt)
  print(msg)

  # step 2: calculate displacement vectors
  # Nt x Nw x Ndims (= 3)
  dt, disps = sugar.time(calculate_displacements)(posa, L)
  msg = 'calculate displacements in %.4f s' % dt
  print(msg)
  dists = np.sqrt(np.sum(disps**2,axis=-1)) # Nt x Nw, confined to the box dimensions
  return ts,dists

def box_gofr(disps, lbox, nelec, nbin=64):
  # calculate pair distribution function g(r), i.e. 1/rho <\sum \delta(r-r_i)>
  nconf, nwalker, ndim = disps.shape
  bin_edges = np.linspace(0, lbox/2, nbin+1)
  # bin distances
  grl = np.zeros([nconf, nbin])
  for iframe, xyz in enumerate(disps):
    dists = np.linalg.norm(xyz, axis=-1)
    rmin = bin_edges[0]; rmax = bin_edges[-1]; dr = bin_edges[1]-bin_edges[0]
    ir = (dists[(rmin<dists) & (dists<rmax)]-rmin)//dr # 2 fwd slashes (//): floor division operator
    ilist, counts = np.unique(ir.astype(int), return_counts=True)
    grl[iframe, ilist] = counts
  grm, gre = yl_ysql(grl)
  # normalize
  vnorm = 4*np.pi/3*np.diff(bin_edges**3)  # bin volume
  rho = nelec*(nelec-1)/2/lbox**3  # pair density
  nvec = 1./(rho*vnorm)  # norm vector
  grm *= nvec/nwalker; gre *= nvec/nwalker
  myx = (bin_edges[1:]+bin_edges[:-1])/2
  return myx, grm, gre

def box_gr3d(disps, wts, lbox, nbin=16):
  mesh = (nbin,)*3
  counts = np.zeros(mesh, dtype=float) #literally the number of counts in each bin
  weightmat = np.zeros(mesh,dtype=float) #walker weights with which to normalize g(r)
  xmin = -lbox/2; xmax = lbox/2; dx = (xmax-xmin)/nbin
  #bin_edges = np.linspace(xmin, xmax, nbin+1)
  ix = ((disps[:, 0]-xmin)//dx).astype(int) #double slash (//) = floor division operator, takes only the integer part of the division operation (basically which bin each electron position falls into)
  iy = ((disps[:, 1]-xmin)//dx).astype(int)
  iz = ((disps[:, 2]-xmin)//dx).astype(int)
  #print(ix.shape)
  ct = 0
  # v1
  #for i, j, k in zip(ix, iy, iz):
    #counts[i, j, k] += 1 # v1
    #weightmat[i,j,k] = wts[ct] # original code
    #ct += 1   
  #gofr = counts*weightmat * np.prod(mesh)/counts.sum() 
  for i, j, k, wt in zip(ix, iy, iz, wts):
    counts[i, j, k] += wt # v1
  gofr = counts * np.prod(mesh)/counts.sum() 
  # normalize by (# bins)/(total # events) = V/N!
  return gofr, counts

def box_sofk(posa, lbox, kcut):
  from qharv.seed import hamwf_h5

  kvecs = hamwf_h5.get_ksphere(np.eye(3)*2*np.pi/lbox, kcut)[1:]
  nframe = len(posa)
  nk = len(kvecs)
  skl = np.zeros([nframe, nk])
  for iframe, walker in enumerate(posa):
    eikr = np.exp(1j*np.inner(kvecs, walker))
    rhok = eikr.sum(axis=-1)
    sk = (rhok*rhok.conj()).real
    skl[iframe, :] = np.mean(sk, axis=-1)
  skm, ske = yl_ysql(skl)
  return kvecs, skm, ske

def yl_ysql(yl, ysql=None):
  """ calculate mean and error given a list of val and sq

  Args:
    yl (list): list of values, (nentry,)
    ysql (list, optional): list of squares, (nentry,)
  Return:
    (np.array, np.array): (ym, ye),
      (mean, error)
  """
  ym = np.mean(yl, axis=0)
  # calculate stddev
  if ysql is None:
    ysql = [y**2 for y in yl]
  y2m = np.mean(ysql, axis=0)
  ye = np.sqrt((y2m-ym**2)/(len(yl)-1))
  return ym, ye

def Calc_3D_gofr(fh5,nx=16,nevery=1000,seed=0,tequil=1500,save=True,generate=False,tmin=0,tmax=np.inf):
  '''
  Calculate 3D pair correlation function g(r)
  'generate' allows an override of previously saved files
  '''
  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  Nw = fp.get('meta/nconfig')[0,0]
  lbox = fp.get('meta/L')[0,0]
  ph = fp.get('meta/ph_bool')[0,0]
  el = fp.get('meta/elec_bool')[0,0]
  coul = int(not fp.get('meta/diffusion')[0,0])
  nstep = fp.get('meta/Nsteps')[0,0]
  if coul == -2: coul = 1
  elif coul == -1: coul = 0 
  eta = fp.get('meta/eta')[0,0]
  l = fp.get('meta/l')[0,0]
  arrstep = fp.get('meta/arrstep')[0,0]
  popstep = fp.get('meta/popstep')[0,0]
  print(el,ph,coul,seed)
  tau = fp.get('meta/tau')[0,0]

  minstep = int(tmin/tau)
  if ~np.isinf(tmax):
    maxstep = int(tmax/tau)
    maxsteplab = maxstep
  else: 
    maxsteplab = -1
    maxstep = np.inf

  savename = 'rs%d_eta%.2f_l%.2f_seed%d_Nw%d_el%d_ph%d_coul%d_nstep%d_popstep%d_arrstep%d_tau%.4f_tequil%d_f%d-%d.mat' % (rs,eta,l,seed, Nw,el, ph,coul,nstep,popstep,arrstep,tau,tequil,minstep,maxsteplab)
  savename = os.path.join(os.path.dirname(fh5),savename)
  my_file = Path(savename)
  if my_file.is_file() & (generate == False):
    print('file already exists!')
    mat = sio.loadmat(savename)
    return mat['gofr'], mat['counts']

  nequil = int(tequil/tau)
  print('num equilibration steps: ',nequil)
  dt, data = sugar.time(extract_walkers)(fp, nevery,nequil,minstep=minstep,maxstep=maxstep)
  _,posa, wts = data
  fp.close()
  print(posa.shape)
  nconf, nwalker, nelec, ndim = posa.shape
  msg = 'extracted %d frames in %.4f s' % (nconf, dt)
  print(msg)

  # step 2: calculate displacement vectors
  dt, disps = sugar.time(calculate_displacements)(posa, lbox)
  msg = 'calculate displacements in %.4f s' % dt
  print(msg)

  # symmetrize displacement
  xyz = np.r_[disps, -disps].reshape(-1, 3)
  symmwts = np.r_[wts,wts].reshape(-1)
  # bin in 3D
  gofr, counts = box_gr3d(xyz, symmwts, lbox, nbin=nx)

  if save:
    # data: save 3D g(r)
    data = {'gofr': gofr, 'counts': counts, 'axes': lbox*np.eye(3), 'origin': -lbox/2*np.ones(3), 'nbins': nx, 'L': lbox, 'seed': seed,'tmin':tmin, 'tmax':tmax,'tau':tau,'tequil':tequil}
    sio.savemat(savename,data)
  return gofr, counts

def CalcSecondMoment(files,nevery=1000,tequil=1500,tmin=0,tmax=np.inf):
  '''
  Calculate 3D pair correlation function g(r)
  'generate' allows an override of previously saved files
  '''

  
  mean_r = np.zeros(len(files))
  variances = np.zeros(len(files))
  mean_r2 = np.zeros(len(files))

  for i,fh5 in enumerate(files):

    fp = h5py.File(fh5, 'r')
    rs = fp.get('meta/rs')[0,0]
    Nw = fp.get('meta/nconfig')[0,0]
    lbox = fp.get('meta/L')[0,0]
    print('system size L: ',lbox)
    ph = fp.get('meta/ph_bool')[0,0]
    el = fp.get('meta/elec_bool')[0,0]
    coul = int(not fp.get('meta/diffusion')[0,0])
    nstep = fp.get('meta/Nsteps')[0,0]
    if coul == -2: coul = 1
    elif coul == -1: coul = 0 
    eta = fp.get('meta/eta')[0,0]
    l = fp.get('meta/l')[0,0]
    arrstep = fp.get('meta/arrstep')[0,0]
    popstep = fp.get('meta/popstep')[0,0]
    print(el,ph,coul)
    tau = fp.get('meta/tau')[0,0]

    minstep = int(tmin/tau)
    if ~np.isinf(tmax):
      maxstep = int(tmax/tau)
      #maxsteplab = maxstep
    else:
      #maxsteplab = -1
      maxstep = np.inf

    nequil = int(tequil/tau)
    print('num equilibration steps: ',nequil)
    dt, data = sugar.time(extract_walkers)(fp, nevery,nequil,minstep=minstep,maxstep=maxstep)
    _,posa, wts = data
    # shape of pos mat: Nt x Nwalkers x Nelec x Ndims
    # shape of wt (+ dist) mat: Nt x Nwalkers 

    fp.close()
  
    print(posa.shape)
    nconf, nwalker, nelec, ndim = posa.shape
    msg = 'extracted %d frames in %.4f s' % (nconf, dt)
    print(msg)

    # step 2: calculate displacement vectors
    dt, disps = sugar.time(calculate_displacements)(posa, lbox)
    msg = 'calculate displacements in %.4f s' % dt
    # displacements: Nt x Nw x 3 in 3D
    print(msg)

    # variance: <r^2> = \sum_i r_i^2 w_i / \sum_i w_i, summing over walkers + time
    print(disps.shape,wts.shape)
    dists = np.sqrt(np.sum(disps**2,axis=2)) #Nt x Nw
    #print('dist',dists.shape)
    mom2 = np.sum(dists**2 * wts, axis=None) / np.sum(wts,axis=None) #* 1/3 # divide by ndims (since the sum includes dx^2 + dy^2 + dz^2, 3 terms)
    mom1 = np.sum(dists*wts,axis=None) / np.sum(wts,axis=None) #*1/3
    print(mom2,mom1)
    #print(mom2.real,mom1.real,mom2.real-mom1.real)
    var = np.sqrt(mom2.real-(mom1.real)**2)
    print('<r> +- SD: ',mom1.real,np.sqrt(var))

    mean_r[i] = mom1.real
    variances[i] = var
    mean_r2[i] = mom2.real

  print('avg')
  r_avg_all = np.mean(mean_r)
  #r_err_all = np.std(mean_r) / np.sqrt(len(files))
  r_err_all = np.sqrt(np.sum(variances)) / len(files)
  print(f'{r_avg_all} +/- {r_err_all}')
  print('mean sd: ',np.mean(np.sqrt(variances)))
  print('mean r2: ',np.mean(mean_r2))

def Plot_3D_gofr(fh5,nx=16,nevery=1000,zlim=[0,5.5],cutoff=2):
  '''
  Calculate and visualize 3D g(r)
  fh5: h5 file of sim data
  nx: # bins per direction

  nevery: how often to extract trajectory
  '''  
  # to load data
  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  Nw = fp.get('meta/nconfig')[0,0]
  ph = fp.get('meta/ph_bool')[0,0]
  eta = fp.get('meta/eta')[0,0]
  l = fp.get('meta/l')[0,0]
  el = fp.get('meta/elec_bool')[0,0]
  seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
  seed = int(fh5.split('_')[seedid+1])
  '''
  coul = fp.get('meta/diffusion')[0,0]
  #savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
  savename = 'rs%d_eta%.2f_l%.2f_seed%d_Nw%d_el%d_ph%d_coul%d.mat' % (rs,eta,l,seed, nwalker,el, ph,coul)
  savename = os.path.join(os.path.dirname(fh5),savename)
  nevery = np.lcm(fp.get('meta/arrstep')[0,0],fp.get('meta/popstep')[0,0])
  my_file = Path(savename)
  if my_file.is_file():
    mat = sio.loadmat(savename)
    counts = mat['gofr']
  else:
    print('file does not exist. Generating one...')
    counts,_ = Calc_3D_gofr(fh5,nx,nevery,seed)
  '''
  counts,nums = Calc_3D_gofr(fh5,nx,nevery,seed)

  # visualize: 3D g(r)
  mesh = (nx,)*3
  rvecs = hamwf_h5.get_rvecs(np.eye(3), mesh)-0.5 #in units of box length
  # 3D
  vals = counts.ravel()
  sel = vals > cutoff #only plot values above this to clear up the screen a bit
   
  fig, ax = volumetric.figax3d()
  ax.set_xlim(-0.5, 0.5)
  ax.set_ylim(-0.5, 0.5)
  ax.set_zlim(-0.5, 0.5)
  #volumetric.isosurf(ax, counts, level_frac=0.7)
  cs = kyrt.color_scatter(ax, rvecs[sel], vals[sel], alpha=0.5,zlim=zlim)
  kyrt.scalar_colorbar(*zlim)
  #fig.savefig('rs%d-ne%d-g3d.png' % (rs, nevery), dpi=320)
  ax.set_title('3D g(r)')
  plt.show()

def calc_closest_factors(c: int):
    """Calculate the closest two factors of c.
    
    Returns:
      [int, int]: The two factors of c that are closest; in other words, the
        closest two integers for which a*b=c. If c is a perfect square, the
        result will be [sqrt(c), sqrt(c)]; if c is a prime number, the result
        will be [1, c]. The first number will always be the smallest, if they
        are not equal.
    """    
    if c//1 != c:
        raise TypeError("c must be an integer.")

    a, b, i = 1, c, 0
    while a < b:
        i += 1
        if c % i == 0:
            a = i
            b = c//a
    
    return [b, a]

def Plot_2D_gofr(files,nx=16,cut=['xy','yz'],err=1E-3,tequil=300):
  ''' Plot 2D slices of g(r) from .h5py files. If multiple files given, plot the average (need to interpolate, which I don't feel like doing). z0 can only have two values.'''

  if len(files) > 1: #show results from different seeds
    sz = np.array(calc_closest_factors(np.ceil(len(files)/2)*2)).astype(int)
    # create two figures, one for each value of z0
    fig, ax = plt.subplots(sz[0],sz[1],figsize=(3*sz[1],2.5*sz[0]))
    fig2, ax2 = plt.subplots(sz[0],sz[1],figsize=(3*sz[1],2.5*sz[0]))
  else:
    sz = [1,1]

  # then create a third figure plotting the average over all different seeds
  fig3, ax3 = plt.subplots(1,2,figsize=(10,4))
  avg_z1 = []; avg_z2 = []; avgr_z1 = []; avgr_z2 = []
  # extract 2D slice
  mesh = (nx,)*3
  rvecs = hamwf_h5.get_rvecs(np.eye(3), mesh)-0.5 #in units of box length
  kyrt.set_style()
  ct = 0
  for i in range(sz[0]):
    for j in range(sz[1]):
      if ct < len(files):
        fh5 = files[ct]
      else: break

      # to load data
      fp = h5py.File(fh5, 'r')
      rs = fp.get('meta/rs')[0,0]
      Nw = fp.get('meta/nconfig')[0,0]
      ph = fp.get('meta/ph_bool')[0,0]
      el = fp.get('meta/elec_bool')[0,0]
      try:
        # batch sims run with js2.sh (jobarray.py)
        seed = int(fh5.split('_seed')[1].split('_')[0])
      except:
        # ind sims run with testphonons.py
        seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
        seed = int(fh5.split('_')[seedid+1])
      eta = fp.get('meta/eta')[0,0]
      l = fp.get('meta/l')[0,0]
      #coul = fp.get('meta/diffusion')[0,0]
      #savename = 'rs%d_eta%.2f_l%.2f_seed%d_Nw%d_el%d_ph%d_coul%d.mat' % (rs,eta,l,seed, Nw,el, ph,coul)
      #savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
      #savename = os.path.join(os.path.dirname(fh5),savename)
      nevery = fp.get('meta/arrstep')[0,0]
      #my_file = Path(savename)
      #if my_file.is_file():
      #  mat = sio.loadmat(savename)
      #  counts = mat['gofr']
      #else:
      #  print('file does not exist. Generating one...')
      counts, _ = Calc_3D_gofr(fh5,nx,nevery,seed,tequil=tequil)
  
      vals = counts.ravel()
      
      # get z=0 cut (xy plane) & x=0 cut (yz plane)
      sel = abs(rvecs[:, 2]-0) < err
      sel2 = abs(rvecs[:, 0]-0) < err
      labs = ['z=0 (xy)','x=0 (yz)']
      # get theta = pi/4, theta = 3pi/4 slices, i.e. z = sin(pi/4)*0.5 (since plotting goes from -L/2 to L/2)
      #sel = abs(rvecs[:,2]-0.3) < np.diff(rvecs[:,2])[0]
      #sel2 = abs(rvecs[:,2] + 0.3) < np.diff(rvecs[:,2])[0]
      #labs = ['$z=0.3L$','$z=-0.3L$']
      if ct == 0:
        avg_z1 = vals[sel]
        avg_z2 = vals[sel2]
        avgr_z1 = rvecs[sel,:2]
        avgr_z2 = rvecs[sel2,1:]
      else:
        avg_z1 = avg_z1 + vals[sel]
        avg_z2 = avg_z2 + vals[sel2]
        avgr_z1 = avgr_z1 + rvecs[sel,:2]
        #avgr_z2 = avgr_z2 + rvecs[sel2,:2]
        avgr_z2 = avgr_z2 + rvecs[sel2,1:]

      if len(files) > 1:
        if sz[0] > 1:
          ax[i,j].set_title(r'seed %d' % seed)
          ax[i,j].set_xlabel('x/L')
          ax[i,j].set_ylabel('y/L')
          ax[i,j].set_aspect(1)
          cs = kyrt.contour_scatter(ax[i,j], rvecs[sel, :2], vals[sel])#, zlim=zlim)
          plt.colorbar(cs,ax=ax[i,j])
          ax2[i,j].set_title(r'seed %d' % seed)
          ax2[i,j].set_xlabel('y/L')
          ax2[i,j].set_ylabel('z/L')
          ax2[i,j].set_aspect(1)
          cs = kyrt.contour_scatter(ax2[i,j], rvecs[sel2, 1:], vals[sel2])#, zlim=zlim)
          #cs = kyrt.contour_scatter(ax2[i,j], rvecs[sel2, :2], vals[sel2])#, zlim=zlim)
          plt.colorbar(cs,ax=ax2[i,j])
        else:
          ax[j].set_title(r'seed %d' % seed)
          ax[j].set_xlabel('x/L')
          ax[j].set_ylabel('y/L')
          ax[j].set_aspect(1)
          cs = kyrt.contour_scatter(ax[j], rvecs[sel, :2], vals[sel])#, zlim=zlim)
          plt.colorbar(cs,ax=ax[j])

          ax2[j].set_title(r'seed %d' % seed)
          ax2[j].set_xlabel('x/L')
          ax2[j].set_ylabel('y/L')
          ax2[j].set_aspect(1)
          cs = kyrt.contour_scatter(ax2[j], rvecs[sel2,1:], vals[sel2])#, zlim=zlim)
          #cs = kyrt.contour_scatter(ax2[j], rvecs[sel2,:2], vals[sel2])#, zlim=zlim)
          plt.colorbar(cs,ax=ax2[j])

      ct = ct + 1
  if len(files) > 1:
    fig.suptitle(r'$r_s$=%d, %s' % (rs,labs[0]))
    fig2.suptitle(r'$r_s$=%d, %s' % (rs,labs[1]))
    fig.tight_layout()
    fig2.tight_layout()

  avg_z1 = avg_z1 / len(files)
  avgr_z1 = avgr_z1 / len(files)
  avg_z2 = avg_z2 / len(files)
  avgr_z2 = avgr_z2 / len(files)
  cs = kyrt.contour_scatter(ax3[0], avgr_z1, avg_z1,cmap='plasma')#, zlim=zlim)
  cb0 = plt.colorbar(cs,ax=ax3[0])
  cb0.set_label('g(r)',fontsize=axfont)
  cs = kyrt.contour_scatter(ax3[1], avgr_z2, avg_z2, cmap='plasma')#, zlim=zlim)
  cb1 = plt.colorbar(cs,ax=ax3[1])
  cb1.set_label('g(r)',fontsize=axfont)
  ax3[0].set_title(labs[0])
  ax3[1].set_title(labs[1])
  ax3[0].set_aspect(1)
  ax3[1].set_aspect(1)
  ax3[0].set_xlabel('x/L',fontsize=axfont)
  ax3[0].set_ylabel('y/L',fontsize=axfont)
  ax3[1].set_xlabel('x/L',fontsize=axfont)
  ax3[1].set_ylabel('y/L',fontsize=axfont)
  fig3.suptitle('avg over %d seeds' %len(files))
  fig3.tight_layout()
  plt.show() 

def Plot_1D_gofr(fh5,nx=16,nevery=1000,vec1=[1,0,0], vec2=[1,1,1],err=0.01,plotting=True):
  ''' Plot 1D cuts along vec1 and vec2. Average along symmetry-related directions'''
  from numpy import matlib as mb
  import numpy.ma as ma

  # to load data
  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  Nw = fp.get('meta/nconfig')[0,0]
  ph = fp.get('meta/ph_bool')[0,0]
  el = fp.get('meta/elec_bool')[0,0]
  lbox = fp.get('meta/L')[0,0]
  seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
  seed = int(fh5.split('_')[seedid+1])
  eta = fp.get('meta/eta')[0,0]
  l = fp.get('meta/l')[0,0]
  coul = fp.get('meta/diffusion')[0,0]
  savename = 'rs%d_eta%.2f_l%.2f_seed%d_Nw%d_el%d_ph%d_coul%d.mat' % (rs,eta,l,seed, nwalker,el, ph,coul)
  #savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
  savename = os.path.join(os.path.dirname(fh5),savename)
  nevery = np.lcm(fp.get('meta/arrstep')[0,0],fp.get('meta/popstep')[0,0])
  my_file = Path(savename)
  if my_file.is_file():
    mat = sio.loadmat(savename)
    gofr = mat['gofr']
    counts = mat['counts']
  else:
    print('file does not exist. Generating one...')
    gofr, counts = Calc_3D_gofr(fh5,nx,nevery,seed)

  vals = gofr.ravel()
  mesh = (nx,)*3
  rvecs = hamwf_h5.get_rvecs(np.eye(3), mesh)-0.5 #in units of box length
  # 1D cut of g(r) along v1 and v2
  # pick out points along these two lines by comparing alignment between angles and vectors
  norms = mb.repmat(np.linalg.norm(rvecs,axis=1),3,1).T
  r = rvecs/norms
  r[np.isnan(r)] = 0
 
  v1 = vec1/np.linalg.norm(vec1)
  v2 = vec2/np.linalg.norm(vec2)
  s1 = np.sum(v1*r,axis=1)
  s2 = np.sum(v2*r,axis=1)
  s1[s1>1] = 1; s1[s1<-1] = -1; s2[s2>1] = 1; s2[s2<-1] = -1; #fix rounding errors
  thetas1 = np.arccos(s1)
  thetas2 = np.arccos(s2)
 
  sel1 = np.array(abs(thetas1) < err)
  sel2 = np.array(abs(thetas2) < err)
  print(np.sum(sel1.astype(int)),np.sum(sel2.astype(int)))
  # define x axis (distance along cut)
  mask1 = np.tile(sel1,(3,1)).T
  mask2 = np.tile(sel2,(3,1)).T
  r1 = ma.array(rvecs,mask=~mask1)
  r2 = ma.array(rvecs,mask=~mask2)
  d1 = ma.array(np.linalg.norm(r1,axis=1)*lbox,mask=~sel1)
  d2 = ma.array(np.linalg.norm(r2,axis=1)*lbox,mask=~sel2)

  dg = vals* np.sqrt(1/counts.ravel() + 1/np.sum(counts)) 
  sortid1 = np.argsort(d1.compressed())
  sortid2 = np.argsort(d2.compressed())
  
  # need to find error bars for x (dist) + y (g(r)) 
  # g(r) = count(r) * V/sum(counts)
  # r.v = rv cos theta --> dr = r tan theta * dtheta
  dr1 = d1*np.tan(ma.array(thetas1,mask=~sel1))*err
  dr2 = d2*np.tan(ma.array(thetas2,mask=~sel2))*err
  y1 = ma.array(vals,mask=~sel1)
  dy1 = ma.array(dg,mask=~sel1)
  y2 = ma.array(vals,mask=~sel2)
  dy2 = ma.array(dg,mask=~sel2)

  if plotting: 
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    ax.errorbar(d1.compressed()[sortid1],y1.compressed()[sortid1],xerr=dr1.compressed()[sortid1],yerr=dy1.compressed()[sortid1], fmt='o-',label='$r_s=%d$, along [%d,%d,%d]' %(rs,vec1[0],vec1[1],vec1[2]))
    #ax.errorbar(d1[sortid1],(vals[sel1])[sortid1],xerr=dr1[sortid1],yerr=(dg[sel1])[sortid1], fmt='o-',label='$r_s=%d$, along [%d,%d,%d]' %(rs,vec1[0],vec1[1],vec1[2]))
    ax.errorbar(d2.compressed()[sortid2],y2.compressed()[sortid2],xerr=dr2.compressed()[sortid2],yerr=dy2.compressed()[sortid2], fmt='o-',label='$r_s=%d$, along [%d,%d,%d]' %(rs,vec2[0],vec2[1],vec2[2]))
    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title('L = %.2f' %lbox)
  
    #valdict = {'gridvecs':rvecs, 'counts':vals,'L':lbox,'rs':rs,'dist1':d1,'gcut1':vals[sel1],'vec1':vec1,'vec2':vec2,'dist2':d2,'gcut2':vals[sel2]}
    #sio.savemat(os.path.splitext(fh5)[0] + '.mat',valdict)
    ax.legend()
    fig.tight_layout()
    plt.show()
  return d1, y1, dr1, dy1, d2, y2, dr2, dy2

def Plot_1D_gofr_avg(files,nx=16,err=0.01,save=False,tequil=1500,tmin=0,tmax=np.inf):
  ''' Plot average 1D slices over many different seeds. Also overlay the corresponding jellium plots.
      manyplot: whether to plot the giant plots containing results from all the seeds 
  '''

  from numpy import matlib as mb
  import numpy.ma as ma

  if len(files) == 1:
    sz = [1,1]
  else:
    sz = np.array(calc_closest_factors(np.ceil(len(files)/2)*2)).astype(int)

  # create two figures, one for an averaged 1D g(r) plot and scattered raw values for each seed
  if len(files) > 1: 
    fig, ax = plt.subplots(sz[0],sz[1],figsize=(3*sz[1],2.5*sz[0]))
    fig2, ax2 = plt.subplots(sz[0],sz[1],figsize=(3*sz[1],2.5*sz[0]))
  #if len(files) > 1:
  # then create a third figure plotting the average over all different seeds
  fig3, ax3 = plt.subplots(1,1,figsize=(5,4))
  avg_g1 = []; avg_g2 = []; avg_g3 = []
  avgr_g1 = []; avgr_g2 = []; avgr_g3 = []
  avg_g1_err = []; avg_g2_err = []; avg_g3_err = []

  # focus on high symmetry directions: (100), (110), (111)
  vecs1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
  vecs1 = np.concatenate((vecs1,-vecs1),axis=0)
  vecs2 = np.array([[1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1]])
  vecs2 = np.concatenate((vecs2,-vecs2),axis=0)
  vecs3 = np.array([[1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0]])
  vecs3_1 = np.roll(vecs3,1,axis=1)
  vecs3_2 = np.roll(vecs3,2,axis=1)
  vecs3 = np.vstack((vecs3,vecs3_1,vecs3_2))
  mesh = (nx,)*3
  rvecs = hamwf_h5.get_rvecs(np.eye(3), mesh)-0.5 #in units of box length
  # 1D cut of g(r) along v1 and v2
  # pick out points along these two lines by comparing alignment between angles and vectors
  norms = mb.repmat(np.linalg.norm(rvecs,axis=1),3,1).T
  r = rvecs/norms
  r[np.isnan(r)] = 0

  ct = 0
  for i in range(sz[0]):
    for j in range(sz[1]):
      if ct < len(files):
        fh5 = files[ct]
      else: break

      # to load data
      fp = h5py.File(fh5, 'r')
      rs = fp.get('meta/rs')[0,0]
      Nw = fp.get('meta/nconfig')[0,0]
      ph = fp.get('meta/ph_bool')[0,0]
      el = fp.get('meta/elec_bool')[0,0]
      lbox = fp.get('meta/L')[0,0]
      try:
        seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
        seed = int(fh5.split('_')[seedid+1])
      except:
        seed = int(fh5.split('_seed')[1].split('_')[0])
      eta = fp.get('meta/eta')[0,0]
      l = fp.get('meta/l')[0,0]
      coul = fp.get('meta/diffusion')[0,0]
      savename = 'rs%d_eta%.2f_l%.2f_seed%d_Nw%d_el%d_ph%d_coul%d.mat' % (rs,eta,l,seed, Nw,el, ph,coul)
      #savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
      savename = os.path.join(os.path.dirname(fh5),savename)
      nevery = fp.get('meta/arrstep')[0,0]
      popstep = fp.get('meta/popstep')[0,0]
      #print(nevery,fp.get('meta/arrstep')[0,0],fp.get('meta/popstep')[0,0])
      my_file = Path(savename)
      print(seed)
      if my_file.is_file():
        mat = sio.loadmat(savename)
        gofr = mat['gofr']
        counts = mat['counts']
      else:
        print('file does not exist. Generating one...')
        gofr, counts = Calc_3D_gofr(fh5,nx,nevery,seed,tequil=tequil,tmin=tmin,tmax=tmax)

      vals = gofr.ravel()
      dg = vals* np.sqrt(1/counts.ravel() + 1/np.sum(counts)) 
      d1 = np.linalg.norm(rvecs,axis=1)*lbox
      sortid1 = np.argsort(d1)
      d1 = d1[sortid1]
      rcollect = []
      y1collect = []
      y2collect = []
      y3collect = []
 
      for vec1 in vecs1:
        v1 = vec1/np.linalg.norm(vec1)
        s1 = np.sum(v1*r,axis=1)
        s1[s1>1] = 1; s1[s1<-1] = -1;
        thetas1 = np.arccos(s1)
 
        sel1 = np.array(abs(thetas1) < err)
        # define x axis (distance along cut)
        mask1 = np.tile(sel1,(3,1)).T
        r1 = ma.array(rvecs,mask=~mask1)

        y1 = ma.array(vals,mask=~sel1)[sortid1]
        # need to find error bars for x (dist) + y (g(r)) 
        # g(r) = count(r) * V/sum(counts)
        # r.v = rv cos theta --> dr = r tan theta * dtheta
        dr1 = (d1*np.tan(ma.array(thetas1,mask=~sel1))*err)[sortid1]
        dy1 = ma.array(dg,mask=~sel1)[sortid1]
        rcollect = np.concatenate((rcollect,ma.array(d1,mask=~sel1[sortid1]).compressed()))
        y1collect = np.concatenate((y1collect,y1.compressed()))
        if len(files) > 1:
          if sz[0] > 1:
            ax2[i,j].errorbar(d1/lbox,y1,xerr=dr1,yerr=dy1, fmt='ko-')#,label='along [%d,%d,%d]' %(vec1[0],vec1[1],vec1[2]))
          else:
            if sz[1] == 1:
              ax2.errorbar(d1/lbox,y1,xerr=dr1,yerr=dy1, fmt='ko-')#,label='along [%d,%d,%d]' %(vec1[0],vec1[1],vec1[2]))
            else:
              ax2[j].errorbar(d1/lbox,y1,xerr=dr1,yerr=dy1, fmt='ko-')#,label='along [%d,%d,%d]' %(vec1[0],vec1[1],vec1[2]))
       
      x1 = np.unique(rcollect) 
      plot1 = np.zeros(x1.shape)
      err1 = np.zeros(x1.shape)
        
      for n,d in enumerate(x1):
        idx = (rcollect == d)
        # average over preexisting bins. Calculate errors of these averages as weighted by 1/sqrt(N)
        # err = SD(data)/sqrt(len(data))
        data1 = y1collect[idx]
        plot1[n] = np.mean(data1)
        err1[n] = np.std(data1)/np.sqrt(len(data1)) #standard error of mean

      if ct == 0:
        avg_g1 = np.zeros((len(files),len(x1)))
        avgr_g1 = np.zeros((len(files),len(x1)))
        avg_g1_err = np.zeros((len(files),len(x1)))
      avg_g1[ct,:] = plot1
      avg_g1_err[ct,:] = err1
      avgr_g1[ct,:] = x1
        
      if len(files) > 1:
        if sz[0] > 1:
          ax[i,j].errorbar(x1/lbox,plot1,yerr=err1, fmt='ko-',label='along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))
        else:
          if sz[1] == 1:
            ax.errorbar(x1/box,plot1,yerr=err1, fmt='ko-',label='along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))
          else:
            ax[j].errorbar(x1/lbox,plot1,yerr=err1, fmt='ko-',label='along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))

      rcollect=[]
      for vec2 in vecs2:
        v2 = vec2/np.linalg.norm(vec2)
        s2 = np.sum(v2*r,axis=1)
        s2[s2>1] = 1; s2[s2<-1] = -1;
        thetas2 = np.arccos(s2)
  
        sel2 = np.array(abs(thetas2) < err)
        # define x axis (distance along cut)
        mask2 = np.tile(sel2,(3,1)).T
        r2 = ma.array(rvecs,mask=~mask2)

        y2 = ma.array(vals,mask=~sel2)[sortid1]
        dr2 = (d1*np.tan(ma.array(thetas2,mask=~sel2))*err)[sortid1]
        dy2 = ma.array(dg,mask=~sel2)[sortid1]
        rcollect = np.concatenate((rcollect,ma.array(d1,mask=~sel2[sortid1]).compressed()))
        y2collect = np.concatenate((y2collect,y2.compressed()))
        if len(files) > 1:
          if sz[0] > 1:
            ax2[i,j].errorbar(d1/lbox,y2,xerr=dr2,yerr=dy2, fmt='ro-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
            ax2[i,j].set_xlabel('r/L')
            ax2[i,j].set_ylabel('g(r)')
            ax2[i,j].set_title('seed %d' %seed)
            #ax2[i,j].legend()
          else:
            if sz[1] == 1:
              ax2.errorbar(d1/lbox,y2,xerr=dr2,yerr=dy2, fmt='ro-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
              ax2.set_xlabel('r/L')
              ax2.set_ylabel('g(r)')
              ax2.set_title('seed %d' %seed)
              #ax2.legend()
            else:
              ax2[j].errorbar(d1,y2,xerr=dr2,yerr=dy2, fmt='ro-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
              ax2[j].set_xlabel('r/L')
              ax2[j].set_ylabel('g(r)')
              ax2[j].set_title('seed %d' %seed)
              #ax2[j].legend()
    
      x2 = np.unique(rcollect) 
      plot2 = np.zeros(x2.shape)
      err2 = np.zeros(x2.shape)
      for n,d in enumerate(x2):
        idx = (rcollect == d)
        # average over preexisting bins. Calculate errors of these averages as weighted by 1/sqrt(N)
        # err = SD(data)/sqrt(len(data))
        data2 = y2collect[idx]
        plot2[n] = np.mean(data2)
        err2[n] = np.std(data2)/np.sqrt(len(data2)) #standard error of mean

      if len(files) > 1:
        if sz[0] > 1:
          ax[i,j].errorbar(x2/lbox,plot2,yerr=err2, fmt='ro-',label='along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
          ax[i,j].set_xlabel('r/L')
          ax[i,j].set_ylabel('g(r)')
          ax[i,j].set_title('seed %d' %seed)
          ax[i,j].legend()
        else:
          if sz[1] == 1:
            ax.errorbar(x2/lbox,plot2,yerr=err2, fmt='ro-',label='along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
            ax.set_xlabel('r/L')
            ax.set_ylabel('g(r)')
            ax.set_title('seed %d' %seed)
            ax.legend()
          else:
            ax[j].errorbar(x2/lbox,plot2,yerr=err2, fmt='ro-',label='along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
            ax[j].set_xlabel('r/L')
            ax[j].set_ylabel('g(r)')
            ax[j].set_title('seed %d' %seed)
            ax[j].legend()

      if ct == 0:
        avg_g2 = np.zeros((len(files),len(x2)))
        avgr_g2 = np.zeros((len(files),len(x2)))
        avg_g2_err = np.zeros((len(files),len(x2)))
      avg_g2[ct,:] = plot2
      avg_g2_err[ct,:] = err2
      avgr_g2[ct,:] = x2
        
      rcollect=[]
      for vec3 in vecs3:
        v3 = vec3/np.linalg.norm(vec3)
        s3 = np.sum(v3*r,axis=1)
        s3[s3>1] = 1; s3[s3<-1] = -1;
        thetas3 = np.arccos(s3)
  
        sel3 = np.array(abs(thetas3) < err)
        # define x axis (distance along cut)
        mask3 = np.tile(sel3,(3,1)).T
        r3 = ma.array(rvecs,mask=~mask3)

        y3 = ma.array(vals,mask=~sel3)[sortid1]
        dr3 = (d1*np.tan(ma.array(thetas3,mask=~sel3))*err)[sortid1]
        dy3 = ma.array(dg,mask=~sel3)[sortid1]
        rcollect = np.concatenate((rcollect,ma.array(d1,mask=~sel3[sortid1]).compressed()))
        y3collect = np.concatenate((y3collect,y3.compressed()))
        if len(files) > 1:
          if sz[0] > 1:
            ax2[i,j].errorbar(d1/lbox,y3,xerr=dr3,yerr=dy3, fmt='bo-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
            ax2[i,j].set_xlabel('r/L')
            ax2[i,j].set_ylabel('g(r)')
            ax2[i,j].set_title('seed %d' %seed)
            #ax2[i,j].legend()
          else:
            if sz[1] == 1:
              ax2.errorbar(d1/lbox,y3,xerr=dr3,yerr=dy3, fmt='bo-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
              ax2.set_xlabel('r/L')
              ax2.set_ylabel('g(r)')
              ax2.set_title('seed %d' %seed)
              #ax2.legend()
            else:
              ax2[j].errorbar(d1/lbox,y3,xerr=dr3,yerr=dy3, fmt='bo-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
              ax2[j].set_xlabel('r/L')
              ax2[j].set_ylabel('g(r)')
              ax2[j].set_title('seed %d' %seed)
              #ax2[j].legend()
    
      x3 = np.unique(rcollect) 
      plot3 = np.zeros(x3.shape)
      err3 = np.zeros(x3.shape)
      for n,d in enumerate(x3):
        idx = (rcollect == d)
        # average over preexisting bins. Calculate errors of these averages as weighted by 1/sqrt(N)
        # err = SD(data)/sqrt(len(data))
        data3 = y3collect[idx]
        plot3[n] = np.mean(data3)
        err3[n] = np.std(data3)/np.sqrt(len(data3)) #standard error of mean

      if len(files) > 1:
        if sz[0] > 1:
          ax[i,j].errorbar(x3/lbox,plot3,yerr=err3, fmt='bo-',label='along [%d,%d,%d]' %(vecs3[0,0],vecs3[0,1],vecs3[0,2]))
          ax[i,j].set_xlabel('r/L')
          ax[i,j].set_ylabel('g(r)')
          ax[i,j].set_title('seed %d' %seed)
          ax[i,j].legend()
        else:
          if sz[1] == 1:
            ax.errorbar(x3/lbox,plot3,yerr=err3, fmt='bo-',label='along [%d,%d,%d]' %(vecs3[0,0],vecs3[0,1],vecs3[0,2]))
            ax.set_xlabel('r/L')
            ax.set_ylabel('g(r)')
            ax.set_title('seed %d' %seed)
            ax.legend()
          else:
            ax[j].errorbar(x3,plot3,yerr=err3, fmt='bo-',label='along [%d,%d,%d]' %(vecs3[0,0],vecs3[0,1],vecs3[0,2]))
            ax[j].set_xlabel('r')
            ax[j].set_ylabel('g(r)')
            ax[j].set_title('seed %d' %seed)
            ax[j].legend()
      
      if ct == 0:
        avg_g3 = np.zeros((len(files),len(x3)))
        avgr_g3 = np.zeros((len(files),len(x3)))
        avg_g3_err = np.zeros((len(files),len(x3)))
      avg_g3[ct,:] = plot3
      avg_g3_err[ct,:] = err3
      avgr_g3[ct,:] = x3
      ct = ct + 1
  
  #now overlay jellium results for comparison
  #jellfile=files[0].split('ph')[0] + 'ph0' + files[0].split('ph')[1][1:]
  # check whether jellium file exists - if not, be sad
  if save:
      infodict = {'vecs1': vecs1, 'vecs2': vecs2, 'vecs3': vecs3, 'g1_avg': avg_g1, 'g1_avg_err': avg_g1_err, 'g1_avg_r': avgr_g1, 'g2_avg': avg_g2, 'g2_avg_err': avg_g2_err, 'g2_avg_r': avgr_g2, 'g3_avg': avg_g3, 'g3_avg_err': avg_g3_err, 'g3_avg_r': avgr_g3}
      sname = savename.split('seed')[0]+'_'.join(savename.split('seed')[1].split('_')[1:])
      sname = os.path.splitext(sname)[0] + '_popstep%d_arrstep%d_1d_avg'%(popstep,nevery) + os.path.splitext(sname)[1] 
      print(sname)
      sio.savemat(sname,infodict)
  if len(files) > 1:
    fig.suptitle('avg over symm. dir, $r_s=%d$, L = %.2f' %(rs,lbox))
    fig2.suptitle('scatter over symm. dir, $r_s=%d$, L = %.2f' %(rs,lbox))
    fig.tight_layout()
    fig2.tight_layout()

    plot3a = np.mean(avg_g1,axis=0)
    err3a = np.std(avg_g1,axis=0)/np.sqrt(len(files)) #plot3a/len(files)*np.sqrt(np.sum(avg_g1_err**2,axis=0))

    plot3b = np.mean(avg_g2,axis=0)
    err3b = np.std(avg_g2,axis=0)/np.sqrt(len(files)) #plot3b/len(files)*np.sqrt(np.sum(avg_g2_err**2,axis=0))

    plot3c = np.mean(avg_g3,axis=0)
    err3c = np.std(avg_g3,axis=0)/np.sqrt(len(files)) 
  else: 
    plot3a = avg_g1[0,:]
    err3a = avg_g1_err[0,:]
    plot3b = avg_g2[0,:]
    err3b = avg_g2_err[0,:]
    plot3c = avg_g3[0,:]
    err3c = avg_g3_err[0,:]

  x3a = np.mean(avgr_g1,axis=0)
  x3b = np.mean(avgr_g2,axis=0)
  x3c = np.mean(avgr_g3,axis=0)
    
  ax3.errorbar(x3a,plot3a,yerr=err3a,fmt='ko-',label='avg along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))
  ax3.errorbar(x3b,plot3b,yerr=err3b,fmt='ro-',label='avg along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
  ax3.errorbar(x3c,plot3c,yerr=err3c,fmt='bo-',label='avg along [%d,%d,%d]' %(vecs3[0,0],vecs3[0,1],vecs3[0,2]))

  # find peak of each curve and avg over all
  print('avg max g(r) value: ',np.mean([max(plot3a), max(plot3b), max(plot3c)]))
  xmean = [x3a[plot3a == max(plot3a)], x3b[plot3b == max(plot3b)], x3c[plot3c == max(plot3c)]]
  print('avg max g(r) associated x-value: ',xmean,np.mean(xmean))
  xwtd = [np.sum(plot3a*x3a)/np.sum(plot3a),np.sum(plot3b*x3b)/np.sum(plot3b),np.sum(plot3c*x3c)/np.sum(plot3c),]
  print('wtd x value: ',xwtd,np.mean(xwtd))

  ax3.set_xlabel('r')
  ax3.set_ylabel('g(r)')
  ax3.legend()
  fig3.suptitle('$r_s = %d$, avg over %d seeds' %(rs, len(files)))
  fig3.tight_layout()
  plt.show()

def Comp2_1Dgofr(folder,diffkey='popstep',diffvals=[50,200],opt='elph'):
  ''' 
  compare 1D slices of g(r) between simulations. Need to first run Plot_1D_gofr_avg 
  diffkey: the key by which the files will be split up into two categories based on the values of diffvals
  '''
  #savename = 'rs%d_Nw%d_el%d_ph%d_1d_avg.mat' % (rs,seed, Nw,el, ph)
  #savename = os.path.join(os.path.dirname(fh5),savename)
  filename = '*_1d_avg.mat' 
  phvals = [int(name.split('ph')[1].split('_')[0]) for name in results]
  elvals = [int(name.split('el')[1].split('_')[0]) for name in results]
  jellvals = [1 if ((ph == 0) & (el > 0)) else 0 for ph,el in zip(phvals,elvals)]
  elphvals = [1 if ((ph > 0) & (el > 0)) else 0 for ph,el in zip(phvals,elvals)]
  polvals = [1 if ((ph > 0) & (el == 0)) else 0 for ph,el in zip(phvals,elvals)]
  
  results = glob.glob(os.path.join(folder,filename))
  print(results)
  if opt == 'elph':
    results = results[elphvals]
  elif opt == 'jell':
    results = results[jellvals]
  elif opt == 'pol':
    results = results[polvals]
  #vals = []
  #labs = []
  #colors = []
  '''
  for i,name in enumerate(results):
    try: 
      v = int(name.split(diffkey)[1].split('_')[0])
      vals.append(v)
    except: 
      v = 0
      vals.append(0)
    if v == diffvals[0]: 
      labs.append(str(diffvals[0]))
      colors.append('k')
    elif v == diffvals[1]: 
      labs.append(str(diffvals[1]))
      colors.append('r')
    else: 
      labs.append('0')
      colors.append(None)
  '''
  vals = [int(name.split(diffkey)[1].split('_')[0]) for name in results]
  print(vals)
  labs = [diffvals[0] if v == diffvals[0] else diffvals[1] for v in vals]
  #print(labs) 
  colors = ['k' if v == diffvals[0] else 'r' for v in vals]
  rs = int(folder.split('rs')[1].split('_')[0])
  L = (4*np.pi/3*2)**(1/3) * rs #sys size
  eta = float(folder.split('eta')[1].split('_')[0])
  l = int(folder.split('l')[1].split('/')[0])
  fig,ax = plt.subplots(1,1,figsize=(6,5))
  mat1 = sio.loadmat(results[0])
  mat2 = sio.loadmat(results[1])
  print(mat1.keys())
  print(mat2.keys())
  vecs1 = mat1['vecs1']
  vecs2 = mat1['vecs2']
  vecs3 = mat1['vecs3']
  g1a = mat1['g1_avg']
  g1a_err = mat1['g1_avg_err']
  g1b = mat1['g2_avg']
  g1b_err = mat1['g2_avg_err']
  g1c = mat1['g3_avg']
  g1c_err = mat1['g3_avg_err']
  r1a = mat1['g1_avg_r']
  r1b = mat1['g2_avg_r']
  r1c = mat1['g3_avg_r']
  if g1a.shape[0] == 1:
    g1a = g1a[0]
    g1a_err = g1a_err[0]
    r1a = r1a[0]
    r1b = r1b[0]
    r1c = r1c[0]
    g1b = g1b[0]
    g1b_err = g1b_err[0]
    g1c = g1c[0]
    g1c_err = g1c_err[0]
  elif g1a.shape[0] > 1:
    g1a_err = np.std(g1a,axis=0)/g1a.shape[0]
    g1a = np.mean(g1a,axis=0)
    g1b_err = np.std(g1b,axis=0)/g1b.shape[0]
    g1b = np.mean(g1b,axis=0)
    g1c_err = np.std(g1c,axis=0)/g1c.shape[0]
    g1c = np.mean(g1c,axis=0)
    r1a = np.mean(r1a,axis=0)
    r1b = np.mean(r1b,axis=0)
    r1c = np.mean(r1c,axis=0)
  g2a = mat2['g1_avg']
  g2a_err = mat2['g1_avg_err']
  g2b = mat2['g2_avg']
  g2b_err = mat2['g2_avg_err']
  g2c = mat2['g3_avg']
  g2c_err = mat2['g3_avg_err']
  r2a = mat2['g1_avg_r'][0]
  r2b = mat2['g2_avg_r'][0]
  r2c = mat2['g3_avg_r'][0]
  if g2a.shape[0] == 1:
    g2a = g2a[0]
    g2a_err = g2a_err[0]
    g2b = g2b[0]
    g2b_err = g2b_err[0]
    g2c = g2c[0]
    g2c_err = g2c_err[0]
  elif g2a.shape[0] > 1:
    g2a_err = np.std(g2a,axis=0)/g2a.shape[0]
    g2a = np.mean(g2a,axis=0)
    g2b_err = np.std(g2b,axis=0)/g2b.shape[0]
    g2b = np.mean(g2b,axis=0)
    g2c_err = np.std(g2c,axis=0)/g2c.shape[0]
    g2c = np.mean(g2c,axis=0)
  ax.errorbar(r1a,g1a,yerr=g1a_err,color=colors[0], fmt='o-',label='[%d,%d,%d] %s' %(vecs1[0,0],vecs1[0,1],vecs1[0,2],labs[0]))
  ax.errorbar(r1b,g1b,yerr=g1b_err,color=colors[0],fmt='o--',label='[%d,%d,%d] %s' %(vecs2[0,0],vecs2[0,1],vecs2[0,2],labs[0]))
  ax.errorbar(r1c,g1c,yerr=g1c_err,color=colors[0],fmt='o:',label='[%d,%d,%d] %s' %(vecs3[0,0],vecs3[0,1],vecs3[0,2],labs[0]))
  ax.errorbar(r2a,g2a,yerr=g2a_err,color=colors[1],fmt='o-',label='[%d,%d,%d] %s' %(vecs1[0,0],vecs1[0,1],vecs1[0,2],labs[1]))
  ax.errorbar(r2b,g2b,yerr=g2b_err,color=colors[1],fmt='o--',label='[%d,%d,%d] %s' %(vecs2[0,0],vecs2[0,1],vecs2[0,2],labs[1]))
  ax.errorbar(r2c,g2c,yerr=g2c_err,color=colors[1],fmt='o:',label='[%d,%d,%d] %s' %(vecs3[0,0],vecs3[0,1],vecs3[0,2],labs[1]))
  ax.set_xlabel('r',fontsize=axfont)
  ax.set_ylabel('g(r)',fontsize=axfont)
  ax.legend(fontsize=legfont)
 
  fig.suptitle('$r_s = %d, (\eta, l) = (%.2f, %.2f)$' %(rs, eta,l),fontsize=titlefont)
  fig.tight_layout()
  plt.show()

def Comp_elph_jell(folder):
  ''' 
  compare 1D slices of g(r) between phonon simulations and jellium simulations. Need to first run Plot_1D_gofr_avg 
  '''
  #savename = 'rs%d_Nw%d_el%d_ph%d_1d_avg.mat' % (rs,seed, Nw,el, ph)
  #savename = os.path.join(os.path.dirname(fh5),savename)
  filename = '*_1d_avg.mat'
  results = glob.glob(os.path.join(folder,filename))
  phvals = [int(name.split('ph')[1].split('_')[0]) for name in results]
  print(phvals)
  labs = ['el_ph' if ph else 'jell' for ph in phvals]
  print(labs) 
  colors = ['k' if ph else 'r' for ph in phvals]
  rs = int(folder.split('rs')[1].split('_')[0])
  L = (4*np.pi/3*2)**(1/3) * rs #sys size
  eta = float(folder.split('eta')[1].split('_')[0])
  l = int(folder.split('l')[1].split('/')[0])
  fig,ax = plt.subplots(1,1,figsize=(6,5))
  mat1 = sio.loadmat(results[0])
  mat2 = sio.loadmat(results[1])
  vecs1 = mat1['vecs1']
  vecs2 = mat1['vecs2']
  vecs3 = mat1['vecs3']
  g1a = mat1['g1_avg']
  g1a_err = mat1['g1_avg_err']
  g1b = mat1['g2_avg']
  g1b_err = mat1['g2_avg_err']
  g1c = mat1['g3_avg']
  g1c_err = mat1['g3_avg_err']
  r1a = mat1['g1_avg_r']
  r1b = mat1['g2_avg_r']
  r1c = mat1['g3_avg_r']
  if g1a.shape[0] == 1:
    g1a = g1a[0]
    g1a_err = g1a_err[0]
    r1a = r1a[0]
    r1b = r1b[0]
    r1c = r1c[0]
    g1b = g1b[0]
    g1b_err = g1b_err[0]
    g1c = g1c[0]
    g1c_err = g1c_err[0]
  elif g1a.shape[0] > 1:
    g1a = np.mean(g1a,axis=0)
    g1a_err = np.std(g1a,axis=0)/g1a.shape[0]
    g1b = np.mean(g1b,axis=0)
    g1b_err = np.std(g1b,axis=0)/g1b.shape[0]
    g1c = np.mean(g1c,axis=0)
    g1c_err = np.std(g1c,axis=0)/g1c.shape[0]
    r1a = np.mean(r1a,axis=0)
    r1b = np.mean(r1b,axis=0)
    r1c = np.mean(r1c,axis=0)
  g2a = mat2['g1_avg']
  g2a_err = mat2['g1_avg_err']
  g2b = mat2['g2_avg']
  g2b_err = mat2['g2_avg_err']
  g2c = mat2['g3_avg']
  g2c_err = mat2['g3_avg_err']
  r2a = mat2['g1_avg_r'][0]
  r2b = mat2['g2_avg_r'][0]
  r2c = mat2['g3_avg_r'][0]
  if g2a.shape[0] == 1:
    g2a = g2a[0]
    g2a_err = g2a_err[0]
    g2b = g2b[0]
    g2b_err = g2b_err[0]
    g2c = g2c[0]
    g2c_err = g2c_err[0]
  elif g2a.shape[0] > 1:
    g2a_err = np.std(g2a,axis=0)/g2a.shape[0]
    g2a = np.mean(g2a,axis=0)
    g2b_err = np.std(g2b,axis=0)/g2b.shape[0]
    g2b = np.mean(g2b,axis=0)
    g2c_err = np.std(g2c,axis=0)/g2c.shape[0]
    g2c = np.mean(g2c,axis=0)
  scale=eta
  ax.errorbar(r1a,g1a,yerr=g1a_err,color=colors[0], fmt='o-',label='[%d,%d,%d] %s' %(vecs1[0,0],vecs1[0,1],vecs1[0,2],labs[0]))
  ax.errorbar(r1b,g1b,yerr=g1b_err,color=colors[0],fmt='o--',label='[%d,%d,%d] %s' %(vecs2[0,0],vecs2[0,1],vecs2[0,2],labs[0]))
  ax.errorbar(r1c,g1c,yerr=g1c_err,color=colors[0],fmt='o:',label='[%d,%d,%d] %s' %(vecs3[0,0],vecs3[0,1],vecs3[0,2],labs[0]))
  ax.errorbar(r2a,g2a*scale,yerr=g2a_err,color=colors[1],fmt='o-',label='[%d,%d,%d] %s' %(vecs1[0,0],vecs1[0,1],vecs1[0,2],labs[1]))
  ax.errorbar(r2b,g2b*scale,yerr=g2b_err,color=colors[1],fmt='o--',label='[%d,%d,%d] %s' %(vecs2[0,0],vecs2[0,1],vecs2[0,2],labs[1]))
  ax.errorbar(r2c,g2c*scale,yerr=g2c_err,color=colors[1],fmt='o:',label='[%d,%d,%d] %s' %(vecs3[0,0],vecs3[0,1],vecs3[0,2],labs[1]))
  ax.set_xlabel('r',fontsize=axfont)
  ax.set_ylabel('g(r)',fontsize=axfont)
  ax.legend(fontsize=legfont)
 
  fig.suptitle('$r_s = %d, (\eta, l) = (%.2f, %.2f)$' %(rs, eta,l),fontsize=titlefont)
  fig.tight_layout()
  plt.show()

def Calc_g_rthetaphi(files,nx=16,err=0.01,save=False,nbins_r=12,nbins_theta=10,nbins_phi=10,cutoff=0.5):
  '''
    avg over all theta to get g(r), and avg over all r,phi to get g(theta)
    g(theta) ideally encodes information about the bipolaron bound state anisotropy
    Have to select region of constant radius r + dr for angular averages, or else box corners (theta = pi/4) will get overrepresented
  '''

  thetabins = np.linspace(0,np.pi,nbins_theta+1)
  phibins = np.linspace(0,2*np.pi,nbins_phi+1)
  fig, ax = plt.subplots(1,3,figsize=(12,4))

  mesh = (nx,)*3
  rvecs = hamwf_h5.get_rvecs(np.eye(3), mesh)-0.5 #in units of box length
  # 1D cut of g(r) along v1 and v2
  # pick out points along these two lines by comparing alignment between angles and vectors
  norms = mb.repmat(np.linalg.norm(rvecs,axis=1),3,1).T
  r = rvecs/norms
  r[np.isnan(r)] = 0
  
  # get angles
  thetas = np.arctan2(rvecs[:,2],np.sqrt(rvecs[:,0]**2+rvecs[:,1]**2))
  thetas = np.abs(thetas - np.pi/2) #map to [0,pi]
  distnorms = np.linalg.norm(rvecs,axis=1)
  #R = 0.12
  #dR = 0.1*R
  tmask = (distnorms < cutoff) # & (distnorms >= R-dR)
  print(len(tmask),np.sum(tmask))
  thetas = ma.array(thetas,mask=~tmask)
  '''
  # visualize what I'm cutting
  fig2,ax2 = plt.subplots(1,1)
  for x in np.unique(rvecs[:,0]):
    ax2.axvline(x)
  for y in np.unique(rvecs[:,1]):
    ax2.axhline(y)
  fis = np.linspace(0,2*np.pi,101)
  ax2.plot(0.5*np.cos(fis),0.5*np.sin(fis),'r')
  ax2.set_aspect(1)
  plt.show()
  '''
  
  phis = np.arctan2(rvecs[:,1],rvecs[:,0])
  phis = np.mod(phis,2*np.pi)
  phis = ma.array(phis,mask=~tmask) 

  #check for jellium file for comparison
  splstr = files[0].split('ph')
  spl2 = splstr[1].split('_')
  if spl2[0] != '0':
    filename = 'DMC_*_el1_ph0_*_nstep_*_*.h5'
    results = glob.glob(os.path.join(os.path.dirname(files[0]),filename))
    jellfile = results[0]
    #spl2[0] = '0'
    #jellfile = splstr[0] + 'ph' + '_'.join(spl2)

    fp = h5py.File(jellfile, 'r')
    rs = fp.get('meta/rs')[0,0]
    Nw = fp.get('meta/nconfig')[0,0]
    eta = fp.get('meta/eta')[0,0]
    l = fp.get('meta/l')[0,0]
    jellname = 'rs%d_eta%.2f_l%.2f_seed0_Nw%d_el1_ph0.mat' % (rs,eta,l, nwalker)
    #jellname = 'rs%d_seed0_Nw%d_el1_ph0.mat' % (rs,Nw)
    jellname = os.path.join(os.path.dirname(jellfile),jellname)
    print(jellfile)
    if Path(jellname).is_file():
      files.append(jellfile) 
      jell = True
    else:
      print('no jellium file found...')
      #jell = False
      _,_ = Calc_3D_gofr(jellfile,nx,nevery,seed)
      jell = True
  else:
    # if the jellium file has already been specified
    jell = False

  grcollect = np.zeros((len(files),nbins_r)) #g(r)
  gtcollect = np.zeros((len(files),nbins_theta)) #g(theta)
  gpcollect = np.zeros((len(files),nbins_phi)) #g(phi)
  grerr = np.zeros((len(files),nbins_r))
  gterr = np.zeros((len(files),nbins_theta)) #g(theta)
  gperr = np.zeros((len(files),nbins_phi)) #g(phi)

  for i in range(len(files)):
      print(i)
      # to load data
      fh5 = files[i]
      fp = h5py.File(fh5, 'r')
      rs = fp.get('meta/rs')[0,0]
      Nw = fp.get('meta/nconfig')[0,0]
      ph = fp.get('meta/ph_bool')[0,0]
      el = fp.get('meta/elec_bool')[0,0]
      lbox = fp.get('meta/L')[0,0]
      eta = fp.get('meta/eta')[0,0]
      l = fp.get('meta/l')[0,0]
      seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
      seed = int(fh5.split('_')[seedid+1])
      eta = fp.get('meta/eta')[0,0]
      l = fp.get('meta/l')[0,0]
      coul = fp.get('meta/diffusion')[0,0] #if diffusion = True, no Coulomb interaction
      savename = 'rs%d_eta%.2f_l%.2f_seed%d_Nw%d_el%d_ph%d_coul%d.mat' % (rs,eta,l,seed, nwalker,el, ph,coul)
      #savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
      savename = os.path.join(os.path.dirname(fh5),savename)

      nevery = fp.get('meta/arrstep')[0,0]
      my_file = Path(savename)
      if my_file.is_file():
        mat = sio.loadmat(savename)
        gofr = mat['gofr']
        counts = mat['counts']
      else:
        print('file does not exist. Generating one...')
        gofr, counts = Calc_3D_gofr(fh5,nx,nevery,seed)

      vals = gofr.ravel()
      dg = vals* np.sqrt(1/counts.ravel() + 1/np.sum(counts)) 
      # get distances
      d1 = np.linalg.norm(rvecs,axis=1)*lbox
      sortid1 = np.argsort(d1)
      d1 = d1[sortid1] #sort box distances from smallest to largest
      rbins = np.linspace(0,lbox,nbins_r+1) 

      sortids_r = np.digitize(d1,rbins) #sort distances from origin into binsto get g(r)
      for j in range(len(rbins)-1):
        binids = sortids_r == j+1 #sorting indices start at 1
        rb = d1[binids]
        valb = vals[sortid1][binids]
        grcollect[i,j] = np.mean(valb)            
        grerr[i,j] = np.std(valb)/np.sqrt(len(valb))

      sortid2 = np.argsort(thetas)
      thetasort = thetas[sortid2] #sort polar angles from smallest to largest
      sortids_theta = np.digitize(thetasort,thetabins)
      sortids_theta = ma.array(sortids_theta,mask=ma.getmask(thetasort))
      for j in range(len(thetabins)-1):
        binids = sortids_theta == j+1 #sorting indices start at 1
        thb = thetasort[binids]
        valb = vals[sortid2][binids]
        gtcollect[i,j] = np.mean(valb)            
        gterr[i,j] = np.std(valb)/np.sqrt(len(valb))

      sortid3 = np.argsort(phis)
      phisort = phis[sortid3] #sort azimuthal angles from smallest to largest
      sortids_phi = ma.array(np.digitize(phisort,phibins),mask=ma.getmask(phisort))
      for j in range(len(phibins)-1):
        binids = sortids_phi == j+1 #sorting indices start at 1
        phb = phisort[binids]
        valb = vals[sortid3][binids]
        gpcollect[i,j] = np.mean(valb)            
        gperr[i,j] = np.std(valb)/np.sqrt(len(valb))
  if jell:
    jell_gr = grcollect[-1,:]
    jell_gt = gtcollect[-1,:]
    jell_gp = gpcollect[-1,:]
    grcollect = grcollect[:-1,:]
    gtcollect = gtcollect[:-1,:]
    gpcollect = gpcollect[:-1,:]
    jell_gr_err = grerr[-1,:]
    jell_gt_err = gterr[-1,:]
    jell_gp_err = gperr[-1,:]
    grerr = grerr[:-1,:]
    gterr = gterr[:-1,:]
    gperr = gperr[:-1,:]
    files.pop(-1)

  gravg = np.mean(grcollect,axis=0)
  gtavg = np.mean(gtcollect,axis=0)
  gpavg = np.mean(gpcollect,axis=0)
  if len(files) > 1:
    gravg_err = np.std(grcollect,axis=0)/np.sqrt(len(files))
    gtavg_err = np.std(gtcollect,axis=0)/np.sqrt(len(files))
    gpavg_err = np.std(gpcollect,axis=0)/np.sqrt(len(files))
  else:
    gravg_err = grerr[0,:]
    gtavg_err = gterr[0,:]
    gpavg_err = gperr[0,:]
  rbincents = (rbins[:-1] + rbins[1:])/2
  thbincents = (thetabins[:-1] + thetabins[1:])/2
  phibincents = (phibins[:-1] + phibins[1:])/2

  ax[0].errorbar(rbincents,gravg,yerr=gravg_err,fmt='ko-',label='el+ph')
  ax[1].errorbar(thbincents,gtavg,yerr=gtavg_err,fmt='ko-',label='el+ph')
  ax[2].errorbar(phibincents,gpavg,yerr=gpavg_err,fmt='ko-',label='el+ph')
  if jell:
    ax[0].errorbar(rbincents,jell_gr,yerr=jell_gr_err,fmt='ro-',label='jell')
    ax[1].errorbar(thbincents,jell_gt,yerr=jell_gt_err,fmt='ro-',label='jell')
    ax[2].errorbar(phibincents,jell_gp,yerr=jell_gp_err,fmt='ro-',label='jell')

  # FIX THIS
  if save:
    infodict = {'rbins': rbins, 'gr_avg': gravg, 'gr_avg_err': gravg_err, 'g1_avg_r': avgr_g1, 'g2_avg': avg_g2, 'g2_avg_err': avg_g2_err, 'g2_avg_r': avgr_g2}
    sname = savename.split('seed')[0]+'_'.join(savename.split('seed')[1].split('_')[1:])
    sname = os.path.splitext(sname)[0] + '_1d_avg_spherical' + os.path.splitext(sname)[1] 
    print(sname)
    sio.savemat(sname,infodict)

  ax[0].set_xlabel('r')
  ax[0].set_ylabel('g(r)')
  ax[0].legend()
  ax[1].set_xlabel('$\\theta$')
  ax[1].set_ylabel('g($\\theta$)')
  ax[1].legend()
  ax[2].set_xlabel('$\phi$')
  ax[2].set_ylabel('$g(\phi)$')
  ax[2].legend()
  #ax[0].legend()
  fig.suptitle('$r_s = %d$ (L = %.2f), ($\eta$,l) = (%.1f, %.1f), avg over %d seeds' %(rs,lbox,eta,l, len(files)))
  fig.tight_layout()
  plt.show()
  
def Plot_sofk(fh5,nevery=1000):
  ''' Plot structure factor'''

  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  lbox = fp.get('meta/L')[0,0]
  dt, _,posa,_ = sugar.time(extract_walkers)(fp, nevery)
  fp.close()
  nconf, nwalker, nelec, ndim = posa.shape
  #msg = 'extracted %d frames in %.4f s' % (nconf, dt)
  #print(msg)

  # step 5: calculate S(k)
  kf = (9*np.pi/4)**(1./3)/rs
  kcut = 5*kf
  kvecs, skm, ske = box_sofk(posa, lbox, kcut)
  skm /= nelec; ske /= nelec
  kmags = np.linalg.norm(kvecs, axis=-1)

  # data: save S(k)
  #data = np.c_[kmags, skm, ske, kvecs]
  #np.savetxt('rs%d-ne%d-sofk.dat' % (rs, nevery), data)

  # visualize: S(k)
  fig, ax = plt.subplots()
  ax.set_xlim(0, 5)
  ax.set_ylim(0, 1.05*skm.max())
  ax.set_xlabel('k/kf')
  ax.set_ylabel('S(k)')

  ax.errorbar(kmags/kf, skm, ske, marker='.', ls='')

  fig.tight_layout()
  plt.show()

def PlotElecDensities(fh5):
  from h5_plotting import GetPosArr
  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  lbox = fp.get('meta/L')[0,0]
  h,ts,posarr,distarr = GetPosArr(fh5)
  fig2,ax2 = plt.subplots(1,3,figsize=(9,4.5)) #histogram of positions 0 < x < L
  for elec in [0,1]:
    modpos = posarr[:,elec,:,:]; #raw elec pos
    modpos = modpos - np.rint(modpos)
    # elec separation distance coordinates - minimum image convention (map back to [-L/2, L/2]
    modx = np.ravel(modpos[:,0,:])
    mody = np.ravel(modpos[:,1,:])
    modz = np.ravel(modpos[:,2,:])
    ax2[0].hist(modx,bins=20,density=True,histtype='step',label='elec %d' %elec)
    ax2[1].hist(mody,bins=20,density=True,histtype='step',label='elec %d' %elec)
    ax2[2].hist(modz,bins=20,density=True,histtype='step',label='elec %d' %elec)
    ax2[0].set_xlabel('$x_i$')
    ax2[1].set_xlabel('$y_i$')
    ax2[2].set_xlabel('$z_i$')
    #ax2[0].set_xlim([0,L])
    #ax2[1].set_xlim([0,L])
    #ax2[2].set_xlim([-0.01,L])
  ax2[0].legend(loc=3)
  ax2[1].legend(loc=3)
  ax2[2].legend(loc=3)
  plt.suptitle('$r_s=%d, L = %.2f$' %(rs,lbox))
  plt.tight_layout()
  plt.show()

def PlotWalkerWeights(fh5,nevery=200,nequil=0):
  '''Plot walker weights to make sure they're not fluctuating like crazy'''
  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  Nw = fp.get('meta/nconfig')[0,0]
  lbox = fp.get('meta/L')[0,0]
  ph = fp.get('meta/ph_bool')[0,0]
  el = fp.get('meta/elec_bool')[0,0]
  
  dt, data = sugar.time(extract_walkers)(fp, nevery,nequil)
  _,posa, wts = data
  fp.close()
  print(wts.shape)
  print(posa.shape)
  nconf, nwalker, nelec, ndim = posa.shape
  msg = 'extracted %d frames in %.4f s' % (nconf, dt)
  print(msg)
  
  fig, ax = plt.subplots(1,1,figsize=(5,4))
  ns=[50,100,150,200,300]
  #ns = np.arange(0,wts.shape[1]-1)
  steps = np.arange(nequil,(wts.shape[0])*nevery,nevery)
  for n in ns:
      #print(wts[:,n])
      ax.plot(steps,wts[:,n].real,label='walker %d' %n)
  ax.set_xlabel('step')
  ax.set_ylabel('walker weight')
  ax.set_xlim([0,10000])
  ax.set_ylim([0,4])
  #ax.legend()
  plt.show()

if __name__ == '__main__':
  #main()  # set no global variable
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs30_nconfig512_data/DMC_bind_diffusion_0_el1_ph1_rs_30_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_140000_popstep150_tau_0.75.h5' #el+ph, rs=30
  #fh5='/mnt/home/llin1/scratch/E_Nw_tests/rs30_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_30_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_140000_popstep150_tau_0.75.h5' #rs=30, jellium
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs110_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_110_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_405000_popstep10_tau_2.75.h5' #rs=110, jellium
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs110_nconfig512_data/DMC_bind_diffusion_0_el1_ph1_rs_110_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_250000_popstep150_tau_2.75.h5' #rs=110, el+ph
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs30_nconfig512_data_eta0.35_l15_Econstraint_tau075/DMC_10_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.35_l15.00_nstep100000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0_l20_Econstraint_tau075_pathinit/DMC_pathinit_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.00_l20.00_nstep80000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.1_l8_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.10_l8.00_nstep80000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.2_l8_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.20_l8.00_nstep80000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.25_l10_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.25_l10.00_nstep80000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.35_l10_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.35_l10.00_nstep80000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.35_l20_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.35_l20.00_nstep80000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.25_l20_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.25_l20.00_nstep80000_popstep50_arrstep200_tau0.75.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.1_l5_Econstraint_tau03/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.10_l5.00_nstep200000_popstep50_arrstep200_tau0.3.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.2_l5_Econstraint_tau03/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.20_l5.00_nstep200000_popstep50_arrstep200_tau0.3.h5'
  #fh5 = 'rs30_nconfig512_data_eta0.25_l5_Econstraint_tau03/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.25_l5.00_nstep200000_popstep50_arrstep200_tau0.3.h5'

  files = sys.argv[1:]
  #files = [fh5]
  tequil = 10000
  #Plot_3D_gofr(files[0],zlim=[0, 5],cutoff=1.5)
  #Plot_1D_gofr(fh5,err=0.05)
  #Plot_sofk(fh5)

  #PlotPhononAmps(files[0],walk=511)

  tmin = 150**2 #40000;
  tmax = 200**2 #55225; 
  #tmin = 0;
  #tmax = np.inf

  # useful functions
  #Plot_2D_gofr(files,tequil=tequil)
  Plot_1D_gofr_avg(files,err=0.05,save=True,tequil=tequil,tmin=tmin,tmax=tmax)
  #Comp2_1Dgofr(sys.argv[1])
  #Comp_elph_jell(sys.argv[1])
  #Calc_g_rthetaphi(files,nbins_theta=8,nbins_phi=8,cutoff=0.2)
  #PlotElecDensities(fh5)
  #CalcSecondMoment(files,nevery=1000,tequil=tequil,tmin=tmin,tmax=tmax)

  #PlotWalkerWeights(files[0])
  #Calc_3D_gofr(files[0], save=False,generate=True)


