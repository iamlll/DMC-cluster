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

def extract_walkers(fp, nevery=1000):
  posl = []
  for key in fp.keys():
    if key.startswith('step'):
      istep = int(key[4:])
      if (istep % nevery) == 0:
        pos1 = fp[key]['pos'][()]
        posl.append(pos1)
  return np.array(posl)

def calculate_displacements(posa, lbox):
  '''Calculate separation distance r12 = r1-r2, and subtract off multiples of box length to find resulting displacements within a single cell
  posa: Nw x Nelec x Nt array
  '''
  frames = []
  for iconf, walkers in enumerate(posa):
    xyz = (walkers[:, 0] - walkers[:, 1])/lbox
    xyz = xyz - np.rint(xyz)
    frames.append(lbox*xyz)
  return np.array(frames)

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

def box_gr3d(disps, lbox, nbin=16):
  mesh = (nbin,)*3
  counts = np.zeros(mesh, dtype=float)
  xmin = -lbox/2; xmax = lbox/2; dx = (xmax-xmin)/nbin
  #bin_edges = np.linspace(xmin, xmax, nbin+1)
  ix = ((disps[:, 0]-xmin)//dx).astype(int) #double slash (//) = floor division operator, takes only the integer part of the division operation (basically which bin each electron position falls into)
  iy = ((disps[:, 1]-xmin)//dx).astype(int)
  iz = ((disps[:, 2]-xmin)//dx).astype(int)
  for i, j, k in zip(ix, iy, iz):
    counts[i, j, k] += 1
  # normalize by (# bins)/(total # events) = V/N!
  gofr = counts* np.prod(mesh)/counts.sum()
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

def Calc_3D_gofr(fh5,nx=16,nevery=1000,seed=0):
  '''Calculate 3D pair correlation function g(r)'''

  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  Nw = fp.get('meta/nconfig')[0,0]
  lbox = fp.get('meta/L')[0,0]
  ph = fp.get('meta/ph_bool')[0,0]
  el = fp.get('meta/elec_bool')[0,0]
  print(type(el),ph,seed)

  dt, posa = sugar.time(extract_walkers)(fp, nevery)
  fp.close()
  nconf, nwalker, nelec, ndim = posa.shape
  msg = 'extracted %d frames in %.4f s' % (nconf, dt)
  print(msg)

  # step 2: calculate displacement vectors
  dt, disps = sugar.time(calculate_displacements)(posa, lbox)
  msg = 'calculate displacements in %.4f s' % dt
  print(msg)

  # symmetrize displacement
  xyz = np.r_[disps, -disps].reshape(-1, 3)
  # bin in 3D
  gofr, counts = box_gr3d(xyz, lbox, nbin=nx)

  # data: save 3D g(r)
  data = {'gofr': gofr, 'counts': counts, 'axes': lbox*np.eye(3), 'origin': -lbox/2*np.ones(3), 'nbins': nx, 'L': lbox, 'seed': seed}
  savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, nwalker,el, ph)
  savename = os.path.join(os.path.dirname(fh5),savename)
  sio.savemat(savename,data)
  return gofr, counts
  
def Plot_3D_gofr(fh5,nx=16,nevery=1000,zlim=[0,5.5],cutoff=2):
  '''
  Calculate and visualize 3D g(r)
  fh5: h5 file of sim data
  nx: # bins per direction

  nevery: how often to extract trajectory
  '''  
  try:
    # to load data
    fp = h5py.File(fh5, 'r')
    rs = fp.get('meta/rs')[0,0]
    Nw = fp.get('meta/nconfig')[0,0]
    ph = fp.get('meta/ph_bool')[0,0]
    el = fp.get('meta/elec_bool')[0,0]
    seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
    seed = int(fh5.split('_')[seedid+1])
    savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
    savename = os.path.join(os.path.dirname(fh5),savename)
    mat = sio.loadmat(savename)
    counts = mat['gofr']
  except FileNotFoundError:
    print('file does not exist. Generating one...')
    counts,_ = Calc_3D_gofr(fh5,nx,nevery)

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

def Plot_2D_gofr(files,nx=16,nevery=1000,z0=[-0.5, 0],err=1E-3):
  ''' Plot 2D slices of g(r) from .h5py files. If multiple files given, plot the average (need to interpolate, which I don't feel like doing). z0 can only have two values.'''
  if len(files) == 1:
    sz = [1,1]
  else: 
    sz = np.array(calc_closest_factors(np.ceil(len(files)/2)*2)).astype(int)
  # create two figures, one for each value of z0
  fig, ax = plt.subplots(sz[0],sz[1],figsize=(4.5*sz[1],4*sz[0]))
  fig2, ax2 = plt.subplots(sz[0],sz[1],figsize=(4.5*sz[1],4*sz[0]))
  if len(files) > 1:
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
      seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
      seed = int(fh5.split('_')[seedid+1])
      savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
      #savename = 'rs%d_Nw%d_el%d_ph%d.mat' % (rs,Nw,el, ph)
      savename = os.path.join(os.path.dirname(fh5),savename)
      my_file = Path(savename)
      if my_file.is_file():
        mat = sio.loadmat(savename)
        counts = mat['gofr']
      else:
        print('file does not exist. Generating one...')
        _, counts = Calc_3D_gofr(fh5,nx,nevery,seed)
  
      vals = counts.ravel()
      sel = abs(rvecs[:, 2]-z0[0]) < err
      sel2 = abs(rvecs[:, 2]-z0[1]) < err
      if ct == 0:
        avg_z1 = vals[sel]
        avg_z2 = vals[sel2]
        avgr_z1 = rvecs[sel,:2]
        avgr_z2 = rvecs[sel2,:2]
      else:
        avg_z1 = avg_z1 + vals[sel]
        avg_z2 = avg_z2 + vals[sel2]
        avgr_z1 = avgr_z1 + rvecs[sel,:2]
        avgr_z2 = avgr_z2 + rvecs[sel2,:2]

      if sz[0] > 1:
        ax[i,j].set_title(r'seed %d' % seed)
        ax[i,j].set_xlabel('x/L')
        ax[i,j].set_ylabel('y/L')
        cs = kyrt.contour_scatter(ax[i,j], rvecs[sel, :2], vals[sel])#, zlim=zlim)
        plt.colorbar(cs,ax=ax[i,j])
        ax2[i,j].set_title(r'seed %d' % seed)
        ax2[i,j].set_xlabel('x/L')
        ax2[i,j].set_ylabel('y/L')
        cs = kyrt.contour_scatter(ax2[i,j], rvecs[sel2, :2], vals[sel2])#, zlim=zlim)
        plt.colorbar(cs,ax=ax2[i,j])
      else:
        if sz[1] == 1:
          ax.set_title(r'seed %d' % seed)
          ax.set_xlabel('x/L')
          ax.set_ylabel('y/L')
          cs = kyrt.contour_scatter(ax, rvecs[sel, :2], vals[sel])#, zlim=zlim)
          plt.colorbar(cs,ax=ax)

          ax2.set_title(r'seed %d' % seed)
          ax2.set_xlabel('x/L')
          ax2.set_ylabel('y/L')
          cs = kyrt.contour_scatter(ax2, rvecs[sel2, :2], vals[sel2])#, zlim=zlim)
          plt.colorbar(cs,ax=ax2)
        else:
          ax[j].set_title(r'seed %d' % seed)
          ax[j].set_xlabel('x/L')
          ax[j].set_ylabel('y/L')
          cs = kyrt.contour_scatter(ax[j], rvecs[sel, :2], vals[sel])#, zlim=zlim)
          plt.colorbar(cs,ax=ax[j])

          ax2[j].set_title(r'seed %d' % seed)
          ax2[j].set_xlabel('x/L')
          ax2[j].set_ylabel('y/L')
          cs = kyrt.contour_scatter(ax2[j], rvecs[sel2, :2], vals[sel2])#, zlim=zlim)
          plt.colorbar(cs,ax=ax2[j])

      ct = ct + 1

  if len(files) > 1:
    avg_z1 = avg_z1 / len(files)
    avgr_z1 = avgr_z1 / len(files)
    avg_z2 = avg_z2 / len(files)
    avgr_z2 = avgr_z2 / len(files)
    cs = kyrt.contour_scatter(ax3[0], avgr_z1, avg_z1)#, zlim=zlim)
    plt.colorbar(cs,ax=ax3[0])
    cs = kyrt.contour_scatter(ax3[1], avgr_z2, avg_z2)#, zlim=zlim)
    plt.colorbar(cs,ax=ax3[1])
    ax3[0].set_title('z=%.2f' %(z0[0],))
    ax3[1].set_title('z=%.2f' %(z0[1],))
    fig3.suptitle('avg over %d seeds' %len(files)
    fig3.tight_layout()
  fig.suptitle(r'$r_s$=%d, z = %.2f' % (rs,z0[0]))
  fig2.suptitle(r'$r_s$=%d, z = %.2f' % (rs,z0[1]))
  fig.tight_layout()
  fig2.tight_layout()

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
  savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
  savename = os.path.join(os.path.dirname(fh5),savename)
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

def Plot_1D_gofr_avg(files,nx=16,nevery=1000,err=0.01):
  from numpy import matlib as mb
  import numpy.ma as ma

  if len(files) == 1:
    sz = [1,1]
  else:
    sz = np.array(calc_closest_factors(np.ceil(len(files)/2)*2)).astype(int)

  # create two figures, one for an averaged 1D g(r) plot and scattered raw values for each seed
  fig, ax = plt.subplots(sz[0],sz[1],figsize=(4*sz[1],3.5*sz[0]))
  fig2, ax2 = plt.subplots(sz[0],sz[1],figsize=(4*sz[1],3.5*sz[0]))
  if len(files) > 1:
    # then create a third figure plotting the average over all different seeds
    fig3, ax3 = plt.subplots(1,1,figsize=(5,4))
    avg_g1 = []; avg_g2 = []; avgr_g1 = []; avgr_g2 = []
    avg_g1_err = []; avg_g2_err = [];

  vecs1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
  vecs1 = np.concatenate((vecs1,-vecs1),axis=0)
  vecs2 = np.array([[1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1]])
  vecs2 = np.concatenate((vecs2,-vecs2),axis=0)
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
      seedid = np.where(np.array(fh5.split('_')) == 'seed')[0][0]
      seed = int(fh5.split('_')[seedid+1])
      savename = 'rs%d_seed%d_Nw%d_el%d_ph%d.mat' % (rs,seed, Nw,el, ph)
      savename = os.path.join(os.path.dirname(fh5),savename)
      my_file = Path(savename)
      print(seed)
      if my_file.is_file():
        mat = sio.loadmat(savename)
        gofr = mat['gofr']
        counts = mat['counts']
      else:
        print('file does not exist. Generating one...')
        gofr, counts = Calc_3D_gofr(fh5,nx,nevery,seed)


      vals = gofr.ravel()
      dg = vals* np.sqrt(1/counts.ravel() + 1/np.sum(counts)) 
      d1 = np.linalg.norm(rvecs,axis=1)*lbox
      sortid1 = np.argsort(d1)
      d1 = d1[sortid1]
      rcollect = []
      y1collect = []
      y2collect = []
 
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
        if sz[0] > 1:
          ax2[i,j].errorbar(d1,y1,xerr=dr1,yerr=dy1, fmt='ko-')#,label='along [%d,%d,%d]' %(vec1[0],vec1[1],vec1[2]))
        else:
          if sz[1] == 1:
            ax2.errorbar(d1,y1,xerr=dr1,yerr=dy1, fmt='ko-')#,label='along [%d,%d,%d]' %(vec1[0],vec1[1],vec1[2]))
          else:
            ax2[j].errorbar(d1,y1,xerr=dr1,yerr=dy1, fmt='ko-')#,label='along [%d,%d,%d]' %(vec1[0],vec1[1],vec1[2]))
       
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

      if len(files) > 1:
        if ct == 0:
          avg_g1 = np.zeros((len(files),len(x1)))
          avgr_g1 = np.zeros((len(files),len(x1)))
          avg_g1_err = np.zeros((len(files),len(x1)))
        avg_g1[ct,:] = plot1
        avg_g1_err[ct,:] = err1
        avgr_g1[ct,:] = x1

      if sz[0] > 1:
        ax[i,j].errorbar(x1,plot1,yerr=err1, fmt='ko-',label='along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))
      else:
        if sz[1] == 1:
          ax.errorbar(x1,plot1,yerr=err1, fmt='ko-',label='along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))
        else:
          ax[j].errorbar(x1,plot1,yerr=err1, fmt='ko-',label='along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))

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
        if sz[0] > 1:
          ax2[i,j].errorbar(d1,y2,xerr=dr2,yerr=dy2, fmt='ro-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
          ax2[i,j].set_xlabel('r')
          ax2[i,j].set_ylabel('g(r)')
          ax2[i,j].set_title('seed %d' %seed)
          #ax2[i,j].legend()
        else:
          if sz[1] == 1:
            ax2.errorbar(d1,y2,xerr=dr2,yerr=dy2, fmt='ro-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
            ax2.set_xlabel('r')
            ax2.set_ylabel('g(r)')
            ax2.set_title('seed %d' %seed)
            #ax2.legend()
          else:
            ax2[j].errorbar(d1,y2,xerr=dr2,yerr=dy2, fmt='ro-')#,label='along [%d,%d,%d]' %(vec2[0],vec2[1],vec2[2]))
            ax2[j].set_xlabel('r')
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

      if sz[0] > 1:
        ax[i,j].errorbar(x2,plot2,yerr=err2, fmt='ro-',label='along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
        ax[i,j].set_xlabel('r')
        ax[i,j].set_ylabel('g(r)')
        ax[i,j].set_title('seed %d' %seed)
        ax[i,j].legend()
      else:
        if sz[1] == 1:
          ax.errorbar(x2,plot2,yerr=err2, fmt='ro-',label='along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
          ax.set_xlabel('r')
          ax.set_ylabel('g(r)')
          ax.set_title('seed %d' %seed)
          ax.legend()
        else:
          ax[j].errorbar(x2,plot2,yerr=err2, fmt='ro-',label='along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
          ax[j].set_xlabel('r')
          ax[j].set_ylabel('g(r)')
          ax[j].set_title('seed %d' %seed)
          ax[j].legend()


      if len(files) > 1:
        if ct == 0:
          avg_g2 = np.zeros((len(files),len(x2)))
          avgr_g2 = np.zeros((len(files),len(x2)))
          avg_g2_err = np.zeros((len(files),len(x2)))
        avg_g2[ct,:] = plot2
        avg_g2_err[ct,:] = err2
        avgr_g2[ct,:] = x2
      ct = ct + 1
  
  fig.suptitle('avg over symm. dir, $r_s=%d$, L = %.2f' %(rs,lbox))
  fig2.suptitle('scatter over symm. dir, $r_s=%d$, L = %.2f' %(rs,lbox))
  fig.tight_layout()
  fig2.tight_layout()

  if len(files) > 1:
    plot3a = np.mean(avg_g1,axis=0)
    err3a = plot3a/len(files)*np.sqrt(np.sum(avg_g1_err**2,axis=0))
    x3a = np.mean(avgr_g1,axis=0)
    ax3.errorbar(x3a,plot3a,yerr=err3a,fmt='ko-',label='avg along [%d,%d,%d]' %(vecs1[0,0],vecs1[0,1],vecs1[0,2]))

    plot3b = np.mean(avg_g2,axis=0)
    err3b = plot3b/len(files)*np.sqrt(np.sum(avg_g2_err**2,axis=0))
    x3b = np.mean(avgr_g2,axis=0)
    ax3.errorbar(x3b,plot3b,yerr=err3b,fmt='ro-',label='avg along [%d,%d,%d]' %(vecs2[0,0],vecs2[0,1],vecs2[0,2]))
    ax3.legend()
    fig3.suptitle('$r_s = %d$, avg over %d seeds' %(rs, len(files)))
    fig3.tight_layout()
  plt.show()

def Plot_sofk(fh5,nevery=1000):
  ''' Plot structure factor'''

  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  lbox = fp.get('meta/L')[0,0]
  dt, posa = sugar.time(extract_walkers)(fp, nevery)
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

# SPLIT THIS FUNCTION UP INTO MULTIPLE PARTS
# OVERLAY jellium result with el-ph
def main():
  import pandas as pd
  import os
  import scipy.io as sio
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  
  nbin = 64  # nbin in 1D g(r)
  nx = 16  # nbin per direction in 3D g(x, y, z)
  nevery = 1000  # how often to extract trajectory

  # trajectory file
  #rs = 110
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs%d_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_110_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_405000_popstep10_tau_2.75.h5' % rs

  # jellium
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/jell_rs%d_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_30_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_210000_popstep10_tau_0.75.h5' % rs
  # "bound" state
  fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs30_nconfig512_data/DMC_bind_diffusion_0_el1_ph1_rs_30_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_140000_popstep150_tau_0.75.h5'

  #rs = 5
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs%d_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_5_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_100000_popstep10_tau_0.12.h5' % rs


  # step 1: extract electron positions
  fp = h5py.File(fh5, 'r')
  rs = fp.get('meta/rs')[0,0]
  lbox = fp.get('meta/L')[0,0]
  dt, posa = sugar.time(extract_walkers)(fp, nevery)
  fp.close()
  nconf, nwalker, nelec, ndim = posa.shape
  msg = 'extracted %d frames in %.4f s' % (nconf, dt)
  print(msg)

  # step 2: calculate displacement vectors
  dt, disps = sugar.time(calculate_displacements)(posa, lbox)
  msg = 'calculate displacements in %.4f s' % dt
  print(msg)

  # step 3: calculate isotropic g(r)
  dt, (myx, grm, gre) = sugar.time(box_gofr)(disps, lbox, nelec, nbin=nbin)
  msg = 'histogrammed g(r) in %.4f s' % dt
  print(msg)

  # data: save g(r)
  #data = np.c_[myx, grm, gre]
  #np.savetxt('rs%d-ne%d-gofr.dat' % (rs, nevery), data)
  
  # visualize: plot g(r)
  fig1, ax1 = plt.subplots()
  ax1.set_xlabel('r')
  ax1.set_ylabel('g(r)')
  #ax1.set_ylim(0, 2)

  ax1.errorbar(myx, grm, gre)
  
  # step 4: 3D g(x, y, z)

  # symmetrize displacement
  xyz = np.r_[disps, -disps].reshape(-1, 3)
  # bin in 3D
  counts = box_gr3d(xyz, lbox, nbin=nx)

  # data: save 3D g(r)
  data = dict(
    data = counts,
    axes = lbox*np.eye(3),
    origin = -lbox/2*np.ones(3),
  )
  '''
  try:
    volumetric.write_gaussian_cube('rs%d.cube' % rs, data)
  except:
    # to load data
    cubedat = volumetric.read_gaussian_cube('rs%d.cube' % rs)
    counts = np.array(cubedat['data'])
    
  '''
  # visualize: 3D g(r)
  mesh = (nx,)*3
  rvecs = hamwf_h5.get_rvecs(np.eye(3), mesh)-0.5 #in units of box length
  zlim = (0, 5.5)
  # 3D
  vals = counts.ravel()
  sel = vals > 2
   
  fig, ax = volumetric.figax3d()
  ax.set_xlim(-0.5, 0.5)
  ax.set_ylim(-0.5, 0.5)
  ax.set_zlim(-0.5, 0.5)
  #volumetric.isosurf(ax, counts, level_frac=0.7)
  cs = kyrt.color_scatter(ax, rvecs[sel], vals[sel], alpha=0.5, zlim=zlim)
  kyrt.scalar_colorbar(*zlim)
  #fig.savefig('rs%d-ne%d-g3d.png' % (rs, nevery), dpi=320)
  ax.set_title('3D g(r)')

  # 2D slice
  z0 = 0
  err = 1E-3
  sel = abs(rvecs[:, 2]-z0) < err
  kyrt.set_style()
  
  fig, ax = plt.subplots(1,2,figsize=(9,4))
  ax[0].set_title(r'$r_s$=%d, z = %d' % (rs,z0))
  ax[0].set_xlabel('x/L')
  ax[0].set_ylabel('y/L')

  vals = counts.ravel()
  i = np.argmax(vals)
  print(vals[sel].shape)
  cs = kyrt.contour_scatter(ax[0], rvecs[sel, :2], vals[sel], zlim=zlim)
  #divider = make_axes_locatable(ax[0])
  #cax = divider.append_axes('right',size='5%',pad=0.05)
  plt.colorbar(cs,ax=ax[0])


  #fig.savefig('tb2_n2-rs%d-g2d.png' % rs, dpi=320)
   
  from numpy import matlib as mb
  # 1D cut of g(r) along v1 and v2
  vec1 = [1,0,0]
  vec2 = [1,1,1]
  v1 = vec1/np.linalg.norm(vec1)
  v2 = vec2/np.linalg.norm(vec2)
  # pick out points along these two lines by comparing alignment between angles and vectors
  norms = mb.repmat(np.linalg.norm(rvecs,axis=1),3,1).T
  r = rvecs/norms
  r[np.isnan(r)] = 0
  s1 = np.sum(v1*r,axis=1)
  s2 = np.sum(v2*r,axis=1)
  s1[s1>1] = 1; s1[s1<-1] = -1; s2[s2>1] = 1; s2[s2<-1] = -1;
  thetas1 = np.arccos(s1)
  thetas2 = np.arccos(s2)
 
  sel1 = np.array(abs(thetas1) < 0.005)
  sel2 = np.array(abs(thetas2) < 0.005)
  # define x axis (distance along cut)
  r1 = rvecs[sel1,:]
  r2 = rvecs[sel2,:]
  d1 = np.linalg.norm(r1,axis=1)
  d2 = np.linalg.norm(r2,axis=1)
  ax[1].plot(d1*lbox,vals[sel1],'o-',label='$r_s=%d$, along [%d,%d,%d]' %(rs,vec1[0],vec1[1],vec1[2]))
  ax[1].plot(d2*lbox,vals[sel2],'o-',label='$r_s=%d$, along [%d,%d,%d]' %(rs,vec2[0],vec2[1],vec2[2]))
  ax[1].set_xlabel('r')
  ax[1].set_ylabel('g(r)')
  ax[1].set_title('L = %.2f' %lbox)
  
  valdict = {'gridvecs':rvecs, 'counts':vals,'L':lbox,'rs':rs,'dist1':d1,'gcut1':vals[sel1],'vec1':vec1,'vec2':vec2,'dist2':d2,'gcut2':vals[sel2]}
  sio.savemat(os.path.splitext(fh5)[0] + '.mat',valdict)
  
  # compare rs=30 with rs = 110 
  mat = sio.loadmat('/mnt/home/llin1/scratch/E_Nw_tests/rs110_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_110_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_405000_popstep10_tau_2.75.mat')
  print(sorted(mat.keys()))
  print(mat['dist1'][0][0])
  ax[1].plot(mat['L'][0]*mat['dist1'][0],mat['gcut1'][0],'o-',label='$r_s = %d$, along [%d,%d,%d]' %(mat['rs'][0],mat['vec1'][0][0],mat['vec1'][0][1],mat['vec1'][0][2]))
  ax[1].plot(mat['L'][0]*mat['dist2'][0],mat['gcut2'][0],'o-',label='$r_s = %d$, along [%d,%d,%d]' %(mat['rs'][0],mat['vec2'][0][0],mat['vec2'][0][1],mat['vec2'][0][2]))
  ax[1].legend()

  fig.tight_layout()
  #fig.savefig('tb2_n2-rs%d-g2d.png' % rs, dpi=320)
  plt.show()
  ''' 
  # step 5: calculate S(k)
  kf = (9*np.pi/4)**(1./3)/rs
  kcut = 5*kf
  kvecs, skm, ske = box_sofk(posa, lbox, kcut)
  skm /= nelec; ske /= nelec
  kmags = np.linalg.norm(kvecs, axis=-1)

  # data: save S(k)
  data = np.c_[kmags, skm, ske, kvecs]
  np.savetxt('rs%d-ne%d-sofk.dat' % (rs, nevery), data)

  # visualize: S(k)
  fig, ax = plt.subplots()
  ax.set_xlim(0, 5)
  ax.set_ylim(0, 1.05*skm.max())
  ax.set_xlabel('k/kf')
  ax.set_ylabel('S(k)')

  ax.errorbar(kmags/kf, skm, ske, marker='.', ls='')

  fig.tight_layout()
  plt.show()
  '''
if __name__ == '__main__':
  #main()  # set no global variable
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs30_nconfig512_data/DMC_bind_diffusion_0_el1_ph1_rs_30_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_140000_popstep150_tau_0.75.h5' #el+ph, rs=30
  #fh5='/mnt/home/llin1/scratch/E_Nw_tests/rs30_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_30_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_140000_popstep150_tau_0.75.h5' #rs=30, jellium
  #fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs110_nconfig512_data/DMC_bind_diffusion_0_el1_ph0_rs_110_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_405000_popstep10_tau_2.75.h5' #rs=110, jellium
  fh5 = '/mnt/home/llin1/scratch/E_Nw_tests/rs110_nconfig512_data/DMC_bind_diffusion_0_el1_ph1_rs_110_popsize_512_seed_0_N_15_eta_0.00_l_5.00_nstep_250000_popstep150_tau_2.75.h5' #rs=110, el+ph
  files = sys.argv[1:]
  #Plot_3D_gofr(fh5,zlim=[0, 5],cutoff=1)
  #Plot_1D_gofr(fh5,err=0.05)
  #Plot_sofk(fh5)

  # useful functions
  Plot_2D_gofr(files)
  #Plot_1D_gofr_avg(files,err=0.05)
  #PlotElecDensities(fh5)
