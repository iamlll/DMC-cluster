import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.optimize import curve_fit
import os
import h5py
import re
sys.path.append('..')
import grscript 

def Viz_Unbound_Dists_etavar(filenames,labels,colors=None,boundd=False):
    '''
        boundd: (bool) whether or not to also plot electron separation distances as calculated for positions confined within the system size
    '''

    fig,ax = plt.subplots(layout='constrained')

    distarr = []
    disterr = []
    teas = []
    if boundd == True:
        boxts = []
        boxds_avg = []
        boxds_err = []
    for i,name in enumerate(filenames):
        df = pd.read_csv(name)
        h = h5py.File(os.path.splitext(name)[0] + '.h5','r')
        steps = df['step'].values
        dists = df['dists'].values
        d_err = df['d_err'].values

        tau = h.get('meta/tau')[0,0]
        Nw = h.get('meta/nconfig')[0,0]
        Nsteps = h.get('meta/Nsteps')[0,0]
        L = h.get('meta/L')[0,0]
        ph = h.get('meta/ph_bool')[0,0]
        r_s = h.get('meta/rs')[0,0]
        diff = h.get('meta/diffusion')[0,0]
        ts = steps*tau
        if i == 0: mint = max(ts)
        else:
            if max(ts) < mint: mint = max(ts)
        distarr.append(dists)
        disterr.append(d_err)
        teas.append(ts)
     
        if boundd == True:
            steps_box,dists_box = grscript.calc_dists_in_box(h,nevery=1,nequil=0)
            ts_box = steps_box*tau
            dbox_err = np.std(dists_box,axis=1)/dists_box.shape[-1] # standard error of walker distn
            dbox_avg = np.mean(dists_box,axis=1) #average over walkers
            boxts.append(ts_box)
            boxds_avg.append(dbox_avg)
            boxds_err.append(dbox_err)

    for i in range(len(distarr)):
        dists = distarr[i]
        d_err = disterr[i]
        ts = teas[i]
        if labels[i] == 'jellium': 
            ls = ':'
            c = 'lightgray'
        else: 
            ls = '-'
            c = None
        tma = steps <= mint
        ts = ts[tma]
        dists = dists[tma]
        d_err = d_err[tma]
        if c is not None:
            line = ax.plot(np.sqrt(ts),dists/L,color=c,label=labels[i],linestyle=ls)
        else:
            line = ax.plot(np.sqrt(ts),dists/L,label=f'{labels[i]} unbound',linestyle=ls)
        ax.fill_between(np.sqrt(ts),(dists+d_err)/L,(dists-d_err)/L,color=line[0].get_color(),alpha=0.4)
        if boundd == True:
            ts_box = boxts[i]
            tmab = ts_box <= mint
            ts_box = ts_box[tmab]
            dbox_avg = boxds_avg[i][tmab]
            dbox_err = boxds_err[i][tmab]
     
            linebox = ax.plot(np.sqrt(ts_box),dbox_avg/L,label='bound',color=line[0].get_color(),linestyle=':')
            ax.fill_between(np.sqrt(ts_box),(dbox_avg+dbox_err)/L,(dbox_avg-dbox_err)/L,color=linebox[0].get_color(),alpha=0.4)

    #ax.legend()
    #ax.axvline(np.sqrt(tequil),color='g',linestyle='--')
    ax.set_xlabel('simulation time$^{1/2}$',fontsize=14)
    ax.set_ylabel('elec. sep. dist. (units of L)',fontsize=14)
    if len(filenames) < 9:
        ax.legend(fontsize=14)
    plt.show()

if __name__=="__main__":
    # Fig 3: fixed r_s and l, vary eta
    file1 = '../rs30_nconfig512_data_eta0_l10_Econstraint_tau0.75/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.00_l10.00_nstep100000_popstep50_arrstep200_tau0.75.csv'
    file2 = '../rs30_nconfig512_data_eta0.2_l10_Econstraint_tau075_bind/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.20_l10.00_nstep80000_popstep50_arrstep200_tau0.75.csv'
    file3 = '../rs30_nconfig512_data_eta0.4_l10_Econstraint_tau075_d1/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.40_l10.00_nstep80000_popstep50_arrstep200_tau0.75.csv'
    file4 = '../rs30_nconfig512_data_eta0.25_l10_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.25_l10.00_nstep80000_popstep50_arrstep200_tau0.75.csv'
    file5 = '../rs30_nconfig512_data_eta0.3_l10_Econstraint_tau075_bind/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.30_l10.00_nstep80000_popstep50_arrstep200_tau0.75.csv'
    file6 = '../rs30_nconfig512_data_eta0.35_l10_Econstraint_tau075/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.35_l10.00_nstep80000_popstep50_arrstep200_tau0.75.csv'
    file7 = '../rs30_nconfig512_data_eta0.9_l10_Econstraint_tau0.75/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.90_l10.00_nstep100000_popstep50_arrstep200_tau0.75.csv'
    reffile = '../rs30_nconfig512_data_eta0_l10_Econstraint_tau0.75/DMC_bind_diffusion0_el1_ph0_rs30_popsize512_seed0_N15_eta0.00_l10.00_nstep100000_popstep50_arrstep200_tau0.75.csv'
    filenames = [file1, file2, file3,file4,file5,file6,reffile]
    labels = ['$\\eta=0$','$\\eta=0.2$','$\\eta=0.4$','$\\eta=0.25$','$\\eta=0.3$','$\\eta=0.9$','jellium']
    
    filenames = sys.argv[1:]
    labels = np.arange(len(filenames))
    n=0
    Viz_Unbound_Dists_etavar(filenames,labels,boundd=True)
