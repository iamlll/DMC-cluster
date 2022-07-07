import pandas as pd
import numpy as np
import sys
import os
import h5py

def reblock(eloc,warmup,nblocks):
    elocblock=np.array_split(eloc[warmup:],nblocks) #throw out "warmup" number of equilibration steps and split resulting local energy array into nblocks subarrays
    blockenergy=[np.mean(x.real) for x in elocblock]
    return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)

#allow multiple input files and concatenate results in same output file
filenames = sys.argv[1:]

#warmup=1000
# equilibration time = timestep * (# steps thrown out)
tequil = 20 
blocksize=1.0 # in Hartree energy units

dfreblock=[]

for name in filenames:
    base,extension = os.path.splitext(name)
    print(extension)
    if 'pkl' in extension:
        df=pd.read_pickle(name)
    else:
        df=pd.read_csv(name)
    hfile = h5py.File(base + '.h5','r')
    tau = hfile.get('meta/tau')[0,0]
    r_s = hfile.get('meta/rs')[0,0]
    N = hfile.get('meta/N_cut')[0,0]
    eta=hfile.get('meta/eta')[0,0]
    ll=hfile.get('meta/l')[0,0]
    if 'alpha' in list(hfile['meta']):
        alpha = hfile.get('meta/alpha')[0,0]
    else:
        alpha = (1-eta)*ll
    blocktau= max(blocksize/tau,blocksize)
    eloc=df.sort_values('step')['elocal'].apply(lambda x:complex(x)).values  #mixed estimator
    egth=df.sort_values('step')['egth'].apply(lambda x:complex(x)).values #growth estimator
    nequil = int(tequil/tau)
    nblocks=int((len(eloc)-nequil)/blocktau)
    avg,err=reblock(eloc,nequil,nblocks)
    avg_gth,err_gth=reblock(egth,nequil,nblocks)
    print(tau,nblocks)
    dfreblock.append({ 
        'n_equil': nequil,
        'eta':eta,
        'l':ll, 
        'alpha': alpha,
        'r_s':r_s,
        'tau':tau,
        'Ncut':N,
        'eavg':avg, #in hartrees, I believe
        'err':err,
        'egth':avg_gth,
        'err_gth':err_gth,
        })

dirname = os.path.dirname(filenames[0]) # take directory of 1st file as output dir
pd.DataFrame(dfreblock).to_csv(os.path.join(dirname,"reblocked_eta%.2f_U%.2f_tequil%d.csv" %(eta,ll,tequil)))
