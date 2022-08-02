import pandas as pd
import numpy as np
import sys
import os
import h5py
import qharv.reel.scalar_dat as qharv

def reblock(eloc,warmup,nblocks):
    elocblock=np.array_split(eloc[warmup:],nblocks) #throw out "warmup" number of equilibration steps and split resulting local energy array into nblocks subarrays
    blockenergy=[np.mean(x.real) for x in elocblock]
    return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)

def ReblockEnergies(filenames,tequil=100,blocksize=1):
    dfreblock=[]

    for name in filenames:
        base,extension = os.path.splitext(name)
        df=pd.read_csv(name)
        hfile = h5py.File(base + '.h5','r')
        tau = hfile.get('meta/tau')[0,0]
        r_s = hfile.get('meta/rs')[0,0]
        N = hfile.get('meta/N_cut')[0,0]
        Nw = hfile.get('meta/nconfig')[0,0]
        eta=hfile.get('meta/eta')[0,0]
        ll=hfile.get('meta/l')[0,0]
        diffusion=hfile.get('meta/diffusion')[0,0]
        elec=hfile.get('meta/elec_bool')[0,0]
        ph=hfile.get('meta/ph_bool')[0,0]
        nequil = int(tequil/tau)
        if 'savestep' in hfile.get('meta').keys():
            savestep = hfile.get('meta/savestep')[0,0]
            nequil = int(nequil/savestep) #convert from num sim steps to num of entries in the energy array (which we use to determine how many blocks to split eloc into) 
        if 'alpha' in list(hfile['meta']):
            alpha = hfile.get('meta/alpha')[0,0]
        else:
            alpha = (1-eta)*ll
        eloc=df.sort_values('step')['elocal'].apply(lambda x:complex(x)).values  #mixed estimator
        egth=df.sort_values('step')['egth'].apply(lambda x:complex(x)).values #growth estimator
        autocor = qharv.corr(eloc)
        blocksize = max(blocksize,np.ceil(autocor))
        print(blocksize)
        blocktau=blocksize # max(blocksize/tau,blocksize)
        nblocks=int((len(eloc)-nequil)/blocktau)
        avg,err=reblock(eloc,nequil,nblocks)
        avg_gth,err_gth=reblock(egth,nequil,nblocks)
        print(tau,nblocks)
        dfreblock.append({ 
            'diffusion':diffusion,
            'elec_bool':elec,
            'ph_bool':ph,
            'n_equil': nequil,
            't_equil': tequil,
            'eta':eta,
            'l':ll, 
            'alpha': alpha,
            'r_s':r_s,
            'tau':tau,
            'Ncut':N,
            'nconfig':Nw,
            'tproj':hfile.get('meta/tproj')[0,0],
            'eavg':avg, #in hartrees, I believe
            'err':err,
            'egth':avg_gth,
            'err_gth':err_gth,
            'blocksize': blocksize
        })

    dirname = os.path.dirname(filenames[0]) # take directory of 1st file as output dir
    print(dfreblock)
    pd.DataFrame(dfreblock).to_csv(os.path.join(dirname,"reblocked_eta%.2f_l%.2f_tequil%d.csv" %(eta,ll,tequil)))
    return dfreblock

if __name__ == '__main__':
    #allow multiple input files and concatenate results in same output file
    filenames = sys.argv[1:]
    print(filenames)
    #warmup=1000
    # equilibration time = timestep * (# steps thrown out)
    tequil = 200 
    ReblockEnergies(filenames,tequil)
