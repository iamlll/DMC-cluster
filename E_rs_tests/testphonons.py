'''
Testing electron + phonon DMC driver

Install harvest_qmcpack (qharv) here: https://github.com/Paul-St-Young/harvest_qmcpack
'''

#!/usr/bin/env python
import numpy as np
import os
import sys
#sys.path.append("../")
from wffiles.metropolis import metropolis_sample
import pandas as pd
import matplotlib.pyplot as plt
from qharv.reel import config_h5
from updatedjastrow import UpdatedJastrow, GetEnergy
import h5py

#define various constants
elec = 1.602E-19*2997924580 #convert C to statC
hbar = 1.054E-34 #J*s
m = 9.11E-31 #kg
w = 0.1*1.602E-19/hbar
epssr = 23000
epsinf = 2.394**2
conv = 1E-9/1.602E-19 #convert statC^2 (expressions with elec^2) to eV
convJ = 1/1.602E-19 #convert J to eV
eta_STO = epsinf/epssr
alpha = (elec**2*1E-9)/hbar*np.sqrt(m/(2*hbar*w))*1/epsinf*(1 - epsinf/epssr) #convert statC to J*m
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless
Ry = m*elec**4*(1E-9)**2/(2*epsinf**2 *hbar**2)*1/1.602E-19 #Rydberg energy unit in media, eV
a0 = hbar**2*epsinf/(m*elec**2 *1E-9); #Bohr radius in media
l = np.sqrt(hbar/(2*m*w))/ a0 #phonon length in units of the Bohr radius  
#####################################
'''phonon energy calculations'''

def gth_estimator(ke_coul, pos, wf,configs,g, tau, h_ks,f_ks, ks, kcopy,phonon=True):
    """ calculate kinetic + Coulomb + electron-phonon and phonon energies in growth estimator formulation
    Input:
      ke_coul: kinetic+Coulomb energy for its shape
      pos: electron positions (nconf,nelec,ndim) 
      wf: wavefunction
      ham: hamiltonian
      tau: timestep
      ks: allowed momentum values
      kcopy: array of k-vector magnitudes, (nconfig) x (# ks) matrix
    Return:
      ke: kinetic energy
      pot: Coulomb energy - a constant for fixed electrons
      ph: Phonon + electron-phonon (local) energies
    """
    if phonon == True:
        #swap 1st and 3rd axes in pos matrix so ks dot r1 = (Nx3) dot (3 x nconf) = N x nconf matrix 
        swappos = np.swapaxes(pos,0,2)

        #find elec density matrix
        dprod1 = np.matmul(ks,swappos[:,0,:]) #np array for each k value; k dot r1
        dprod2 = np.matmul(ks,swappos[:,1,:]) #k dot r2 
        rho = np.exp(1j*dprod1) + np.exp(1j*dprod2) #electron density eikr1 + eikr2
        #Update f_k from H_ph and H_eph; [tau] = 1/ha
        fp = f_ks* np.exp(-tau/(2*l**2))
        f2p = fp - 1j*tau* g/kcopy * np.conj(rho) #f'' = f' - it*g/k* (rho*)
    
        #Update weights from H_ph and H_eph, and calculate local energy
        ph = -1./tau* (np.sum( tau*1j* g * fp/kcopy*rho,axis=0) + np.sum( np.conj(h_ks)*(f2p-f_ks),axis=0) ) #sum over all k-values; coherent state weight contributions are normalized
    else:
        f2p = np.zeros(f_ks.shape)
        ph = np.zeros(ke_coul.shape)
    return ke_coul+ph, f2p

def update_f_ks(pos, wf,g, tau, h_ks,f_ks, ks, kcopy,phonon=True):
    """ calculate electron density and update phonon coherence amplitudes.
    Input:
      pos: electron positions (nconf,nelec,ndim) 
      wf: wavefunction
      g: density of states of electron-phonon interaction
      tau: timestep
      ks: allowed momentum values
      kcopy: array of k-vector magnitudes, (nconfig) x (# ks) matrix
    Return:
      rho: electron density
      newf_ks: updated coherence state amplitudes
    """
    if phonon == True:
        #swap 1st and 3rd axes in pos matrix so ks dot r1 = (Nx3) dot (3 x nconf) = N x nconf matrix 
        swappos = np.swapaxes(pos,0,2)

        #find elec density matrix
        dprod1 = np.matmul(ks,swappos[:,0,:]) #np array for each k value; k dot r1
        dprod2 = np.matmul(ks,swappos[:,1,:]) #k dot r2 
        rho = np.exp(1j*dprod1) + np.exp(1j*dprod2) #electron density eikr1 + eikr2
        #Update f_k from H_ph and H_eph; [tau] = 1/ha
        newf_ks = f_ks* np.exp(-tau/(2*l**2)) - 1j*tau* g/kcopy * np.conj(rho) #f'' = f' - it*g/k* (rho*)
    else:
        rho = np.zeros((len(ks),pos.shape[0])) 
        newf_ks = np.zeros(f_ks.shape)
    return rho, newf_ks

def mixed_estimator(ke_coul, pos, wf, configs, rho, g, h_ks, f_ks, kmag,phonon=True):
    '''
    Calculate energy (in ha) using the mixed estimator form E_0 = <psi_T| H |phi>, psi_T & phi are coherent states
    Also syncs DMC driver configs with internal wf electron configurations (GetEnergy)
    Input:
        ke_coul: kinetic+Coulomb energy for its shape
        pos: electron positions (nelec, ndim, nconfigs)
        rho: electron density (eikr1 + eikr2)
        kmag: k-vector magnitudes, matrix size (len(ks), nconfigs)
        h_ks: coherent state amplitudes of trial wave function psi_T (len(ks), nconfigs)
        f_ks: coherent state amplitudes of our time-evolved numerical coherent state |{f_k}>
    Output:
        total energy
    '''
    #Find electron phonon energy
    if phonon == True:
        H_eph = 1j* g*np.sum( (-f_ks * rho + np.conj(h_ks) *np.conj(rho))/kmag , axis=0) #sum over all k values; f/kmag = (# ks) x nconfigs matrix. See eqn 
        #find H_ph
        H_ph = 1/(2*l**2) * np.sum(f_ks* np.conj(h_ks),axis=0)
    else:
        H_eph = np.zeros(ke_coul.shape)
        H_ph = np.zeros(ke_coul.shape)
    return ke_coul + H_eph + H_ph

def init_f_k(ks, kmag, g, nconfig):
    '''
    Initialize the phonon displacement functions f_k from the optimized Gaussian result
    input:
        ks: allowed k-vectors in the supercell
    '''
    #find f_ks
    yopt = 1.39
    sopt = 1.05E-9/a0 #in units of the Bohr radius
    d = yopt*sopt #assume pointing in z direction
    f_ks = -4j*g*l**2/kmag* np.exp(-kmag**2 * sopt**2/4) * (np.cos(ks[:,2] * d/2) - np.exp(-yopt**2/2) )/(1- np.exp(-yopt**2/2))
    f_kcopy = np.array([[ f_ks[i] for j in range(nconfig)] for i in range(len(ks))]) #make f_ks array size (# ks) x (# configurations)
    return f_kcopy

#####################################

def acceptance(posold, posnew, driftold, driftnew, tau, wf):
    """
    Acceptance for importance sampling
    Input:
      poscur: electron positions before move (nelec,ndim,nconf) 
      posnew: electron positions after move (nelec,ndim,nconf)
      driftnew: drift vector at posnew 
      tau: time step
      wf: wave function object
      configs: DMC driver configuration
    Return:
      ratio: [backward move prob.]/[forward move prob.]
      """
    #check axes of summation: originally (nelec, ndim, nconfigs)
    #now (nconfigs, nelec, ndim)
    gfratio = np.exp(
        -np.sum((posold - posnew - driftnew) ** 2 / (2 * tau), axis=(1, 2))
        + np.sum((posnew - posold - driftold) ** 2 / (2 * tau), axis=(1, 2))
    )
    
    ratio = wf.val(posnew) ** 2 / wf.val(posold) ** 2
    return np.minimum(1,ratio * gfratio)

def popcontrol(pos, weight, f_ks, wavg, wtot):
    # keep track of ancestry (which walker numbers of the previous timestep led to the walkers in the current time step)
    probability = np.cumsum(weight / wtot)
    randnums = np.random.random(nconfig)
    # determine which walkers get duplicated / killed
    new_indices = np.searchsorted(probability, randnums) #indices at which new walkers should get inserted, i.e. indices of old walkers with probability closest to new ones?
    
    posnew = pos[new_indices, :, :]
    newf_ks = f_ks[:,new_indices]
    weight.fill(wavg)
    return posnew, newf_ks, weight, new_indices

def plotamps(kcopy, n_ks, N):
    # f_ks: (# ks) x (nconfig) array of coherent state amplitudes. Want to make histogram of f_ks vs |k| for final config of f_ks.
    fig,ax = plt.subplots(2,1)
    ax[0].plot(kcopy.flatten(), n_ks.real.flatten(),'.')
    ax[0].set_xlabel('$|\\vec{k}|$')
    ax[0].set_ylabel('Re($n_k$)')
    ax[1].plot(kcopy.flatten(), n_ks.imag.flatten(),'.')
    ax[1].set_xlabel('$|\\vec{k}|$')
    ax[1].set_ylabel('Im($n_k$)')
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    fig.suptitle('$N = $' + str(N))

    plt.tight_layout()
    plt.show()
    #want to find the relationship between k_cut and L 

def InitPos(wf,opt='rand',d=None):
    if opt == 'bcc':
        #initialize one electron at center of box and the other at the corner
        # i.e. put the elecs as far away from each other as possible
        pos= np.zeros((wf.nconfig, wf.nelec, wf.ndim))
        pos0 = np.full((wf.nconfig, wf.ndim),wf.L/2)
        pos1 = np.full((wf.nconfig, wf.ndim),wf.L)
        pos[:,0,:] = pos0
        pos[:,1,:] = pos1
    elif opt == 'bind':
        # configuration more conducive to binding - put elecs close together, with sep dist ~ phonon length scale
        if d is not None: 
            #randomly place the first electron, then place the 2nd electron a distance l away
            pos= np.zeros((wf.nconfig, wf.nelec, wf.ndim))
            pos[:,0,:] = wf.L* np.random.rand(wf.nconfig, wf.ndim)
            pos[:,1,:] = pos[:,0,:] + np.tile([d,0,0],(wf.nconfig,1))
        else:
            print('Invalid choice of electron separation distance')
    else:    
        pos= wf.L* np.random.rand(wf.nconfig, wf.nelec, wf.ndim)
    return pos

from itertools import product
def simple_dmc(wf, tau, pos, popstep=10,savestep=5, arrstep=10,tproj=128, nstep=None,N=5, L=10,elec=True,phonon=True,l=l,eta=eta_STO,gth=True,h5name="dmc.h5",resumeh5=''):
    """
  Inputs:
  L: box length (units of a0)
  pos: initial position
  nstep: total number of steps in the sim
  N: number of allowed k-vals in each direction
  popstep: number of steps between running population control / printing to screen
  arrstep: number of steps between saving phonon array f_k + full position array
  savestep: number of steps between saving energy / elec separation dist info
  elec: on/off switch for electron diffusion
  phonon: on/off switch for phonon coherent state updates
  gth: on/off switch for growth estimator

  Outputs:
  A Pandas dataframe with each 
  """
    from time import time
    # use HDF file for large data output
    if len(resumeh5) > 0: resume = True
    else: resume = False
    print(resume)
    if resume:
        origdf = pd.read_csv(os.path.splitext(resumeh5)[0] + '.csv')
       
        tick = time()
        df = origdf.to_dict('list')
        tock = time()
        #print(df)
        print('df copying time: %.2f s' %(tock-tick))
        origfile = config_h5.open_read(resumeh5,'r')
        tick = time()
        origfile.copy_file(h5name,overwrite=True)
        origfile.close()
        tock = time()
        print('h5 copying time: %.2f s' %(tock-tick))
         
        h5file = config_h5.open_read(h5name,'a')
        f = h5py.File(resumeh5,'r')
    else:
        df = {
            "step": [],
            "ke_coul": [],
            "elocal": [],
            "egth": [],
            "eref": [],
            "dists": [],
            "d_err": [], #walker error
            "acc_ratio": [], #acceptance ratio
        }
        h5file = config_h5.open_write(h5name)

    alpha = (1-eta)*l
    print('alpha',alpha)
    L = wf.L
    g = 1./l**2*np.sqrt(np.pi*alpha*l/L**3) #DOS, all lengths in units of Bohr radii a0

    nconfig = pos.shape[0]

    if resume:
        nstep_old = int(f.get('meta/Nsteps')[0,0])
        print('nstep_old',nstep_old,type(nstep_old))
        print('arrstep',arrstep,type(arrstep))
        arrstep = int(f.get('meta/arrstep')[0,0])
        popstep =int(f.get('meta/popstep')[0,0])
        savestep = int(f.get('meta/savestep')[0,0])
        if 'pos' in list(f.keys()):
            pos = np.array(f.get('pos'))
        else:
            pos = np.array(f.get('step%d/pos' %(arrstep*np.floor(nstep_old/arrstep),)))
        weight = np.array(f.get('step%d/weight' %(arrstep*np.floor(nstep_old/arrstep),)))
    else:
        nstep_old = 0
        weight = np.ones(nconfig)

    print(nstep)
    if resume:
        h5file.root.meta.Nsteps[0] = nstep
    else:
        meta = {'tau': tau, 'l': l, 'eta': eta, 'rs': wf.rs, 'L': L, 'N_cut': N, 'g': g, 'alpha': alpha, 'nconfig': nconfig, 'savestep':savestep, 'popstep':popstep, 'arrstep': arrstep,'elec_bool': elec,'ph_bool': phonon, 'gth_bool': gth, 'tproj': tproj, 'Nsteps': nstep, 'diffusion':int(wf.diffusion) }  # add more as needed
        # turn each value into an array
        for key, val in meta.items():
            meta[key] = [val]
        metagrp = h5file.create_group(h5file.root,'meta')
        config_h5.save_dict(meta, h5file, metagrp)

    #setup wave function
    configs = wf.setup(pos)
    if nconfig != wf.nconfig:
        print("Incompatible number of walkers: sim nconfig = " + str(nconfig) + ", but wf nconfig = " + str(wf.nconfig) + ". Please re-run step1_opt.py for " + str(nconfig) + " walkers, then try again. Exiting program...")
        return

    if resume:
        ks = np.array(f.get('ks'))
        kmag = np.array(f.get('kmags'))
        f_ks = np.array(f.get('step%d/f_ks' %(arrstep*np.floor(nstep_old/arrstep),)))
        h_ks = np.array(f.get('h_ks'))
    else:
        #Make a supercell/box
        #k = (nx, ny, nz)*2*pi/L for nx^2+ny^2+nz^2 <= n_c^2 for cutoff value n_c = N, where n_c -> inf is the continuum limit. 
        #A k-sphere cutoff is conventional as it specifies a unique KE cutoff
        ks = 2*np.pi/L* np.array([[nx,ny,nz] for nx,ny,nz in product(range(-N,N+1), range(-N,N+1), range(-N,N+1)) if nx**2+ny**2+nz**2 <= N**2 ])

        kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes
        #delete \vec k = 0
        idx = np.where(kmag !=0)[0]
        ks = ks[idx]
        kmag = kmag[idx]
        
        #initialize f_ks
        f_ks = init_f_k(ks, kmag, g, nconfig)
        if phonon == False: f_ks.fill(0.)
        h_ks = f_ks #this describes our trial wave fxn coherent state amplitudes
        #egth,_ = gth_estimator(pos, wf, configs, g, tau,h_ks, f_ks, ks, kcopy,phonon)
    
    kcopy = np.array([[ kmag[i] for j in range(nconfig)] for i in range(len(kmag))]) # (# ks) x nconfig matrix

    rho, _ = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy,phonon)
    ke_coul = GetEnergy(wf,configs,pos,'total')
    elocold = mixed_estimator(ke_coul, pos, wf, configs, rho, g, h_ks, f_ks, kcopy,phonon)

    eref = np.mean(elocold)
    print(eref)

    if resume:
        timers = dict(
          drift_diffusion = float(f.get('meta/drift_diffusion')[0]),
          mixed_estimator = float(f.get('meta/mixed_estimator')[0]),
          gth_estimator = float(f.get('meta/gth_estimator')[0]),
          update_coherent = float(f.get('meta/update_coherent')[0]),
          branch = float(f.get('meta/branch')[0]),
        )
    else:
        timers = dict(
          drift_diffusion = 0.0,
          mixed_estimator = 0.0,
          gth_estimator = 0.0,
          update_coherent = 0.0,
          branch = 0.0,
        )

    print('nstep_old',nstep_old)
    print('nstep',nstep)
    if nstep_old == 0:
        ts = range(nstep_old,nstep+1)
    else:
        ts = range(nstep_old+1,nstep+1)

    maxsave = np.floor(max(ts)/arrstep)*arrstep # last timestep at which position arrays are being saved
    save_phonons = 10*np.maximum(arrstep,popstep) #how frequently to dump phonon amplitude f_k information (do so only after population control) 
    print(save_phonons)

    for istep in ts:
        tick = time()
        if elec == True:
            driftold = tau * wf.grad(pos)

            # Drift+diffusion 
            #with importance sampling
            posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
            driftnew = tau * wf.grad(posnew)
            acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
            imove = acc > np.random.random(nconfig)
            pos[imove,:, :] = posnew[imove,:, :]
            acc_ratio = np.sum(imove) / nconfig
        tock = time()
        timers['drift_diffusion'] += tock - tick

        #update coherent state amplitudes
        tick = time()
        rho, f2p = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy,phonon)
        tock = time()
        timers['update_coherent'] += tock - tick

        #compute observables
        tick = time()
        ke_coul = GetEnergy(wf,configs,pos,'total') #syncs internal wf configs object + driver configs object
        eloc = mixed_estimator(ke_coul, pos, wf, configs, rho, g, h_ks, f_ks, kcopy,phonon)
        tock = time()
        timers['mixed_estimator'] += tock - tick
        tick = time()
        if gth:
            egth,_ = gth_estimator(ke_coul, pos, wf, configs, g,tau, h_ks, f_ks, ks, kcopy,phonon)
        else: egth = np.zeros(eloc.shape)
        tock = time()
        timers['gth_estimator'] += tock - tick
        #syncs internal wf configs object + driver configs object
        f_ks = f2p

        oldwt = np.mean(weight)
        weight = weight* np.exp(-0.5* tau * (elocold + eloc - 2*eref))
        elocold = eloc
        
        # Branch
        tick = time()
        wtot = np.sum(weight)
        wavg = wtot / nconfig
        if elec == True:
            if istep % popstep == 0:
                pos, f_ks,weight, ancestor_indices = popcontrol(pos, weight, f_ks,wavg, wtot)
                wf.update(configs,pos)
        tock = time()
        timers['branch'] += tock - tick

        # Update the reference energy
        Delta = -1./tau* np.log(wavg/oldwt) #need to normalize <w_{n+1}>/<w_n>
        eref = eref + Delta

        if istep % 5000:
            print(istep)

        ''' 
        if istep % savestep == 0 or istep == nstep-1:
            print(
                "iteration",
                istep,
                "avg wt",
                wavg.real,
                "ke_coul",
                np.mean(ke_coul),
                "average energy",
                np.mean(eloc * weight / wavg),
                "eref",
                eref,
                "sig_gth",
                np.std(eloc),
            )
        '''
        if istep % savestep == 0:
            dists = np.sqrt(np.sum((pos[:,0,:]-pos[:,1,:])**2,axis=1))
            avgdists = np.mean(dists) #average over walkers
            d_err = np.std(dists)/nconfig
            
            df["step"].append(istep)
            df["ke_coul"].append(np.mean(ke_coul))
            df["elocal"].append(np.mean(eloc))
            df["egth"].append(np.mean(egth))
            df["eref"].append(eref)
            df['dists'].append(avgdists)
            df['d_err'].append(d_err)
            df["acc_ratio"].append(acc_ratio)
      
        grp = h5file.create_group(h5file.root, 'step%d' % istep)
        big_data = {}
        if (istep % popstep == 0) | (istep == maxsave):
            big_data['ancestor_indices'] = ancestor_indices
        if (istep % arrstep == 0) | (istep == maxsave):
            big_data['pos'] = pos
            big_data['weight'] = weight
        if (istep % save_phonons == 0) | (istep == maxsave):
            big_data['f_ks'] = f_ks
        config_h5.save_dict(big_data, h5file, slab=grp)

    if resume:
          h5file.root.meta.drift_diffusion[0] = timers['drift_diffusion']
          h5file.root.meta.mixed_estimator[0] = timers['mixed_estimator']
          h5file.root.meta.gth_estimator[0] = timers['gth_estimator']
          h5file.root.meta.update_coherent[0] = timers['update_coherent']
          h5file.root.meta.branch[0] = timers['branch']
    else:
        saver = {'kmags': kmag, 'ks':ks, 'h_ks': h_ks}
        config_h5.save_dict(saver, h5file)
 
        #save timings in h5 file also
        config_h5.save_dict(timers, h5file, metagrp)
   
    h5file.close()
    print('Timings:')
    for key, val in timers.items():
      line = '%16s %.4f' % (key, val)
      print(line)
    #plotamps(kcopy,n_ks, N)
    return pd.DataFrame(df)

def simple_vmc(wf, g, tau, pos, nstep=1000, N=10, L=10):
    """
    Force every walker's weight to be 1.0 at every step, and never create/destroy walkers (i.e. no drift, no weights). Uses Metropolis algorithm to accept/reject steps and ensure MC has |psi_T|^2 as its equilibrium distribution.

    In practice, the following two steps should be sufficient for VMC:
    1. keep diffusion term so that electrons move from one step to another R -> R'
    2. use Metropolis criteria to accept/reject according to |Psi_T|^2(R')/|Psi_T|^2(R)
    No weights are needed (a.k.a. set weight=1 for all walkers at every step)

    Inputs:
        L: box length (units of a0)
 
    Outputs:
        A Pandas dataframe with each 

    """
    df = {
        "step": [],
        "r_s": [],
        "tau": [],
        "elocal": [],
        "ke": [],
        "acceptance": [],
    }
    nconfig = pos.shape[0]
    weight = np.ones(nconfig)
    #setup wave function
    configs = wf.setup(pos)
    if nconfig != wf.nconfig:
        print("Incompatible number of walkers: sim nconfig = " + str(nconfig) + ", but wf nconfig = " + str(wf.nconfig) + ". Please re-run step1_opt.py for " + str(nconfig) + " walkers, then try again. Exiting program...")
        return

    #Make a supercell/box
    #k = (nx, ny, nz)*2*pi/L for nx^2+ny^2+nz^2 <= n_c^2 for cutoff value n_c = N, where n_c -> inf is the continuum limit. 
    #A k-sphere cutoff is conventional as it specifies a unique KE cutoff
    ks = 2*np.pi/L* np.array([[nx,ny,nz] for nx,ny,nz in product(range(1,N+1), range(1,N+1), range(1,N+1)) if nx**2+ny**2+nz**2 <= N**2 ])

    kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes
    kcopy = np.array([[ kmag[i] for j in range(nconfig)] for i in range(len(kmag))]) # (# ks) x nconfig matrix

    #initialize f_ks
    f_ks = init_f_k(ks, kmag, g, nconfig)
    h_ks = f_ks #this describes our trial wave fxn coherent state amplitudes

    rho, _ = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy)
    eloc = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)

    eref = np.mean(eloc)
    print(eref)

    for istep in range(nstep):
        wfold=wf.val(pos)
        elocold = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)
        # propose a move
        gauss_move_old = np.random.randn(*pos.shape)
        posnew=pos + np.sqrt(tau)*gauss_move_old

        wfnew=wf.val(posnew)
        # calculate Metropolis-Rosenbluth-Teller acceptance probability
        prob = wfnew**2/wfold**2 # for reversible moves
        # get indices of accepted moves
        acc_idx = (prob + np.random.random_sample(nconfig) > 1.0)
        # update stale stored values for accepted configurations
        pos[acc_idx,:,:] = posnew[acc_idx,:,:]
        wfold[acc_idx] = wfnew[acc_idx]
        acceptance = np.mean(acc_idx) #avg acceptance rate at each step (NOT total, would have to additionally divide by nstep)
        #update coherent state amplitudes
        rho, f2p = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy, phonons=False)
        ke = GetEnergy(wf,configs,pos,'ke') #syncs internal wf configs object + driver configs object
        eloc = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy,phonons=False)
        #syncs internal wf configs object + driver configs object
        f_ks = f2p

        if istep % 10 == 0:
            print(
                "iteration",
                istep,
                "ke", np.mean(ke),
                "average energy",
                np.mean(eloc),
                "acceptance",acceptance
            )

        df["step"].append(istep)
        df["ke"].append(np.mean(ke))
        df["elocal"].append(np.mean(eloc))
        df["acceptance"].append(acceptance)
        df["tau"].append(tau)
        df["r_s"].append(r_s)

    return pd.DataFrame(df)

#####################################

if __name__ == "__main__":
    from updatedjastrow import UpdatedJastrow
    import time
    from argparse import ArgumentParser
    import re
    parser = ArgumentParser()
    parser.add_argument('--rs', type=int, default=4)
    parser.add_argument('--nconf', type=int, default=512)
    parser.add_argument('--seed',type=int,default=0) #random seed
    parser.add_argument('--elec', type=int, default=1) #on/off switch for electrons
    parser.add_argument('--ph', type=int, default=1) #on/off switch for phonons
    parser.add_argument('--Ncut',type=int,default=10) # defines spherical momentum cutoff k_cut = 2pi*N/L
    parser.add_argument('--Nstep',type=int) # number of steps in simulation
    parser.add_argument('--tproj',type=int,default=128) # projection time = tau * nsteps
    parser.add_argument('--arrstep',type=int,default=50) # how frequently to save phonon info + interparticle distances
    parser.add_argument('--popstep',type=int,default=200) # how frequently to branch --> comb through existing walkers
    parser.add_argument('--savestep',type=int,default=5) # how frequently to save energy information
    parser.add_argument('--l',type=np.float64,default=l) #plays the role of the electron coupling strength U 
    parser.add_argument('--eta',type=np.float64,default=eta_STO) 
    parser.add_argument('--gth',type=int,default=1) #on/off switch for growth estimator
    parser.add_argument('--outdir',type=str,default='data') 
    parser.add_argument('--init',type=str,default='bind') 
    parser.add_argument('--tau',type=np.float64)
    parser.add_argument('--diffusion',type=int,default=0) # on/off switch for diffusion (jellium, no Coulomb). Allowed to have diffusion (i.e. 0 Coulomb) + phonons to try and define a baseline for binding energy calc
    parser.add_argument('--resume',type=int,default=0) # whether to resume the simulation from a previous file (if so, give the name of the .h5 file to resume from) 

    args = parser.parse_args()
    # if resume is True, find the name of the file I want to resume from based on the other input arguments
    r_s = args.rs  # inter-electron spacing, controls density
    tau = args.tau
    print(tau)
    if tau is None: tau = r_s/40 
    Nstep=args.Nstep
    popstep = args.popstep
    arrstep = args.arrstep
    savestep = args.savestep
    nconfig = args.nconf #default is 5000
    seed = args.seed
    elec_bool = args.elec > 0
    gth_bool = args.gth > 0
    diffusion = args.diffusion > 0
    if Nstep is None:
        tproj = args.tproj #projection time = tau * nsteps
        Nstep = int(tproj/tau)
    else:
    # if num of sim steps is specified, this determines the sim projection time
        if Nstep % arrstep != 0:
            Nstep = int(arrstep*np.ceil(Nstep/arrstep))
            print('big arrays (position, phonon amplitudes) are only saved every %d steps, rounding to nearest save point: Nstep = %d' %(arrstep, Nstep))
        tproj = Nstep*tau

    #if diffusion:
    #    ph_bool = 0
    #else: ph_bool = args.ph > 0
    
    ph_bool = args.ph > 0
    N = args.Ncut
    init = args.init
    l = args.l
    eta = args.eta
    datadir = args.outdir
    print('diffusion',diffusion)
    print('Nstep',Nstep)

    wf = UpdatedJastrow(r_s,nconfig=nconfig,diffusion=diffusion)
    print(wf.L)

    # Modify the Frohlich coupling constant alpha = (1-eta)*\tilde l
    print('l',l)
    print(datadir)
    if os.path.isdir(datadir) == 0:
        print('Making data directory...')
        os.mkdir(datadir) 

    # if want to resume from a previous file, find said file first. If file not found, start sim from scratch
    resume = args.resume > 0
    if resume:
        filename = 'DMC_%s_diffusion_%d_el%d_ph%d_rs_%d_popsize_%d_seed_%d_N_%d_eta_%.2f_l_%.2f_nstep_[0-9]*_popstep%d_tau_%s.h5' %(init,int(diffusion),int(elec_bool), int(ph_bool), r_s,nconfig, seed, N, eta, l,popstep,str(tau))
        results = [x for x in os.listdir(datadir) if re.match(filename,x) and os.path.exists(os.path.join(datadir,os.path.splitext(x)[0]+'.csv'))]
        print(results)
        if len(results) == 0:
            resumefile = ''
        else:
            nsteps = [h5py.File(os.path.join(datadir,name),'r').get('meta/Nsteps')[0,0] for name in results] 
            print(nsteps)    
            idx = np.argwhere(nsteps == max(nsteps))[0][0]
            resumefile = os.path.join(datadir,results[idx])
            f = h5py.File(resumefile,'r')
            pos = np.array(f.get('step%d/pos' %(arrstep*np.floor(nsteps[idx]/arrstep),)))
            if pos is None: 
                resumefile = ''
            else:
                Nstep = Nstep + nsteps[idx]
            print(resumefile)
            print(Nstep)
    else: 
        resumefile = ''

    if len(resumefile) == 0:
        pos = InitPos(wf,init,d=1) 

    filename = "DMC_{9}_diffusion_{10}_el{8}_ph{7}_rs_{0}_popsize_{1}_seed_{2}_N_{3}_eta_{4:.2f}_l_{5:.2f}_nstep_{6}_popstep{11}_arrstep{12}".format(r_s, nconfig, seed, N,eta,l,Nstep,int(ph_bool),int(elec_bool),init,int(diffusion),popstep,arrstep)
    print(filename)
    print('elec',elec_bool)
    print('ph',ph_bool)
    print('gth',gth_bool)
   
    #LLP = -alpha/(2*l**2) #-alpha hw energy lowering for single polaron
    #feyn = (-alpha -0.98*(alpha/10)**2 -0.6*(alpha/10)**3)/(2*l**2)
    print('N',N)
    #print('LLP',2*LLP)
    #print('Feyn',feyn)
    np.random.seed(seed)
    tic = time.perf_counter()
    filename = os.path.join(datadir,filename + '_tau_' + str(tau))
    h5name = filename + ".h5"
    print(h5name)
    csvname = filename + ".csv"
        
    df = simple_dmc(
        wf,
        pos= pos,
        tau=tau,
        popstep=popstep,
        N=N,
        nstep=Nstep,
        tproj=tproj,
        l=l,
        eta=eta,
        elec=elec_bool,
        phonon=ph_bool,
        gth=gth_bool,
        h5name = h5name,
        arrstep = arrstep,
        savestep = savestep,
        resumeh5 = resumefile,
    )
    df.to_csv(csvname, index=False)
       
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")

