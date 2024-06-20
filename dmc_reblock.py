import pandas as pd
import numpy as np
import sys
import os
import h5py
import qharv.reel.scalar_dat as qhv

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
        #seed = hfile.get('meta/seed')[0,0]
        nequil = int(tequil/tau)
        if 'savestep' in hfile.get('meta').keys():
            savestep = hfile.get('meta/savestep')[0,0]
            nequil = int(nequil/savestep) #convert from num sim steps to num of entries in the energy array (which we use to determine how many blocks to split eloc into) 
        if 'alpha' in list(hfile['meta']):
            alpha = hfile.get('meta/alpha')[0,0]
        else:
            alpha = (1-eta)*ll
        eloc=df.sort_values('step')['elocal'].apply(lambda x:complex(x)).values  #mixed estimator
        #egth=df.sort_values('step')['egth'].apply(lambda x:complex(x)).values #growth estimator
        ke_coul = df.sort_values('step')['ke_coul'].apply(lambda x:complex(x)).values #growth estimator
        elph1 = df.sort_values('step')['H_eph1'].apply(lambda x:complex(x)).values #growth estimator
        elph2 = df.sort_values('step')['H_eph2'].apply(lambda x:complex(x)).values #growth estimator
        eph = df.sort_values('step')['H_ph'].apply(lambda x:complex(x)).values #growth estimator
        autocor = qhv.corr(eloc)
        blocksize = max(blocksize,np.ceil(autocor))
        print(blocksize)
        blocktau=blocksize # max(blocksize/tau,blocksize)
        nblocks=int((len(eloc)-nequil)/blocktau)
        avg,err=reblock(eloc,nequil,nblocks)
        #avg_gth,err_gth=reblock(egth,nequil,nblocks)
        avg_kc,err_kc=reblock(ke_coul,nequil,nblocks)
        avg1,err1=reblock(elph1,nequil,nblocks)
        avg2,err2=reblock(elph2,nequil,nblocks)
        avg_ph,err_ph=reblock(eph,nequil,nblocks)
        print(tau,nblocks)
        dfreblock.append({ 
            #'seed': seed,
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
            'ke_coul':avg_kc,
            'elph1': avg1,
            'elph2': avg2,
            'eph': avg_ph,
            #'egth':avg_gth,
            #'err_gth':err_gth,
            'blocksize': blocksize
        })
    dfreblock = pd.DataFrame(dfreblock)
    print('mean: %.4f, sd: %.4f' %(dfreblock['eavg'].mean(), dfreblock['eavg'].std()))
    dirname = os.path.dirname(filenames[0]) # take directory of 1st file as output dir
    savename = os.path.join(dirname,"reblocked_eta%.2f_l%.2f_tequil%d.csv" %(eta,ll,tequil))
    print(savename)
    dfreblock.to_csv(savename)
    return dfreblock

def CalcBindingEnergy(files):
    filename = files[0]
    df=pd.read_csv(filename)
    df_ph = df[(df['diffusion'] == 0) & (df['ph_bool'] == 1)]
    #df_jell = df[(df['diffusion'] == 0) & (df['ph_bool'] == 0)]
    df_jell = df[(df['diffusion'] == 1) & (df['ph_bool'] == 1)] #no coulomb with phonons to imitate 2 indep polarons in box
    E_ph = df_ph['eavg'].mean()
    E_jell = df_jell['eavg'].mean()
    print(df_ph.head())
    print(df_jell.head())
    print('energies (ha)',E_ph, E_jell, E_ph-E_jell)
    eta = df['eta'].values[0]
    l = df['l'].values[0]
    alpha = (1-eta)*l
    print(eta,l,alpha)
    # alpha hw --> alpha/(2l^2) in hartrees
    # https://faculty.kfupm.edu.sa/phys/aanaqvi/rydberg.pdf
    pred = -alpha/(2*l**2)
    print('predicted GS energy in ha (weak, strong):',2*pred,2*pred*alpha/(3*np.pi))

if __name__ == '__main__':
    #allow multiple input files and concatenate results in same output file
    filenames = sys.argv[1:]
    #warmup=1000
    # equilibration time = timestep * (# steps thrown out)
    tequil = 50 
    ReblockEnergies(filenames,tequil)
    #CalcBindingEnergy(filenames) #read in reblocked csv 
