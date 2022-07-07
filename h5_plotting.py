import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from scipy.optimize import curve_fit
import re
import pandas as pd
import os

def FitData(xvals, yvals, yerr=[], fit='lin', extrap=[]):
    def fitlinear(x,a,b):
        f = a*x + b 
        return f

    bnds = ([-10,-10],[5,5]) #bounds for weak coupling fit
    guess =[-1,-3]
    if len(yerr) > 0:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, sigma=yerr, p0=guess,bounds=bnds)
    else:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, p0=guess,bounds=bnds)
    #print(param)
    a,b = param
    aerr, berr = np.sqrt(np.diag(p_cov)) #standard deviation of the parameters in the fit
    
    if len(extrap) > 0:
        ans = np.array([fitlinear(x,a,b) for x in extrap])
    else:    
        ans = np.array([fitlinear(x,a,b) for x in xvals])
    
    textstr = '\n'.join((
        r'$E(\tau) = a\tau + b$',
        r'$a=%.4f \pm %.4f$' % (a, aerr),
        r'$b=%.6f \pm %.6f$' % (b, berr)
        ))

    print(textstr)
    return ans, textstr

def Get_h5_steps(filename,tequil=20):
    ''' Extract phonon distributions (Nwalkers x Nkpoints) + electron separation distance array (Nwalkers-length vector) from h5py file'''

    f = h5py.File(filename,'r')
   
    tau = f.get('meta/tau')[0,0]
    arrstep = f.get('meta/arrstep')[0,0]
    Nsteps = f.get('meta/Nsteps')[0,0]
    Nw = f.get('meta/nconfig')[0,0]
    nequil = int(tequil/tau)
    
    Nt=int(Nsteps/arrstep) #will erase 0 cols at end (keys are unsorted so it's easiest to first just shove everything into an array and then cut it down), so no -int(nequil/arrstep)
    print(Nt)
    keys = list(f.keys())
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    
    Nks = np.array(f.get('ks')).shape[0]   
    farray = np.zeros((Nt,Nks),dtype=complex)
    narray = np.zeros((Nt,Nks),dtype=complex)
    darray = np.zeros((Nt,Nw)) #inter-electron distance array
    ts = np.zeros(Nt)
    minid = 0
    for i,test_str in enumerate(keys):
        try:
            res = temp.match(test_str).groups()    
        except AttributeError:
            # doesn't correspond to a data header entry of "step#"
            continue
        if int(res[1]) < nequil:
            if int(res[1]) == 0: minid = i
            continue
        # now want to sort through these in order, loop through them and extract each set of f_k and n_k (also distances r), then concatenate them to make a giant array
        f_ks = np.array(f.get(test_str + '/f_ks'))
        # if f_ks is 2D (Nkpts x Nwalkers), average over all walkers at each k-point and find the SD
        if len(f_ks.shape) > 1:
            f_ks = f_ks.mean(axis=1)
         
        n_ks = np.array(f.get(test_str+'/n_ks'))
        edists = np.array(f.get(test_str+'/dists')) #dist bw elecs
        farray[i-minid,:] = f_ks 
        narray[i-minid,:] = n_ks 
        darray[i-minid,:] = edists 
        ts[i-minid] = int(res[1])   
         
    zidx = ~np.all(darray==0,axis=1)
    darray = darray[zidx,:]
    farray = farray[zidx,:]
    narray = narray[zidx,:]
    ts = ts[zidx]
    tidx = np.argsort(ts)
    darray = darray[tidx,:]
    farray = farray[tidx,:]
    narray = narray[tidx,:]
    ts = ts[tidx]

    return f,ts,farray, narray, darray

def Phonon_Mom_Density_h5(filename):
    '''
    n_k = a*a = h* f as a function of wave vector k magnitude |k|
    '''    
    f,_,n_ks, f_ks, _ = Get_h5_steps(filename)
    n_ks = n_ks.flatten()
    f_ks = f_ks.flatten() 
    ks = np.array(f.get('ks'))
    ktile = np.tile(ks,int(len(f_ks)/len(ks))) 
    h_ks = np.array(f.get('h_ks'))
    l = f.get('meta/l')[0,0]
    eta = f.get('meta/eta')[0,0]
    alpha = f.get('meta/alpha')[0,0]
    tau = f.get('meta/tau')[0,0]
    r_s = f.get('meta/rs')[0,0]
    
    fig,ax = plt.subplots(2,1,figsize=(6,4.5))
    # currently can only process one tau value at a time
    ax[0].plot(ktile,np.abs(f_ks),'.',label='$|f|, \\tau = %.2f$' %(tau,) )
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

def Elec_sep_dist(filename):
    '''
    Electron separation distance histogram
    '''    
    f,steps,_,_,dists = Get_h5_steps(filename)
    rs = dists.flatten()
    l = f.get('meta/l')[0,0]
    eta = f.get('meta/eta')[0,0]
    alpha = f.get('meta/alpha')[0,0]
    tau = f.get('meta/tau')[0,0]
    r_s = f.get('meta/rs')[0,0]
    fig,ax = plt.subplots(1,1,figsize=(5,7))
    # currently can only process one tau value at a time
    ax.hist(rs)
    ax.set_xlabel('$|\\vec r_{12}|$')
    ax.set_ylabel('count')
    fig.suptitle('$\eta=%.2f,\,U=%.2f,\,r_s = %d$' %(eta,2*l,r_s))

    fig2,ax2 = plt.subplots(1,1)
    avgdists = np.mean(dists,axis=1)
    d_err = np.std(dists,axis=1)
    
    #ax2.plot(ts,avgdists)
    ts = steps*tau
    ax2.errorbar(np.sqrt(ts),avgdists,yerr=d_err)
    ax2.set_xlabel('$\sqrt{t}$ (sim time$^{1/2}$)')
    ax2.set_ylabel('$|\\vec r_{12}|$')
    ax2.set_title('Averaged over $N_w=$%d walkers' %dists.shape[1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
  #main(['energytotal',])
  filename = sys.argv[1]
  #f = h5py.File(filename,'r')
  #print(list(f.keys()))
  #Get_h5_steps(f)
  #Phonon_Mom_Density_h5(f)
  Elec_sep_dist(filename)
