import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.optimize import curve_fit
import os
import h5py

def E_vs_var(filenames,xvar='r_s',labels=[],comp=True):
    ''' Use reblocked csv file (dmc_reblock.py). Plot E vs r_s (sys density) for fixed eta, U (l), etc.

    comp: Whether to compare pure diffusion, jellium, and "full" system (with phonons + electron interactions)
    '''

    fig,ax = plt.subplots(1,1)
    if len(labels) == len(filenames):
        islabel = True
    else: islabel = False
    for i,name in enumerate(filenames):
        data = pd.read_csv(name)
       
        if comp == True:
            data = data.sort_values(['diffusion','ph_bool'])
            df_diff = data[data['diffusion']==1]
            df_jell = data[(data['diffusion']==0) & (data['ph_bool']==0)]
            df_ph = data[(data['diffusion']==0) & (data['ph_bool']==1)]
            
            dflist = [df_diff,df_jell,df_ph]
            labels=['diffusion','jellium','elec+ph']
            islabel=True
        else:
            dflist = [df]
        for n,df in enumerate(dflist):
            if df.empty:
                print('dataframe is empty!')
                continue
        
            if xvar == 'kcut':
                xs = df['Ncut'].values
                Ls = np.full(len(xs),(2*4*np.pi/3)**(1/3) * df['r_s'].values) #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
                xs = 2*np.pi*xs/Ls 
            elif xvar == '1/r_s':
                xs = 1./df['r_s'].values
            if xvar == '1/Nw' or xvar == '1/nconfig':
                xs = 1./df['nconfig'].values
            else:
                xs = df[xvar].values
            idxs = np.argsort(xs)
            xs = xs[idxs]
            eta = df['eta'].values[0]
            l = df['l'].values[0]
            alpha = (1-eta)*l
            E_gs = df['eavg'].values[idxs]
            E_err = df['err'].values[idxs]
            if islabel: 
                ax.errorbar(xs,E_gs,yerr=E_err,label=labels[n])
            else: 
                lab = "$N_{cut}=%d, \\eta=%.2f$" %(df['Ncut'].values[0],eta)
                ax.errorbar(xs,E_gs,yerr=E_err,label=lab)
            print(E_gs)
            print(E_err)
        if xvar == '1/r_s':
            titlename = '$N_w = %d, \\tau = %.2f, (\eta,U)=(%.2f,%.2f)$' %(df['nconfig'].values[0],df['tau'].values[0],eta,2*l)
            xlab = xvar
            if i == len(filenames)-1:
                idx = np.where(xs >= 0.15)[0]
                print(idx)
                fit,txt=FitData(xs[idx],E_gs[idx],guess=[2,2])
                ax.plot(xs[idx],fit,'r--',label='fit',zorder=10)
                ax.text(0.2, 0.4, txt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes) 
        elif xvar == 'r_s':
            xlab = xvar
            titlename = '$N_w = %d, \\tau = %.2f, (\eta,U)=(%.2f,%.2f)$' %(df['nconfig'].values[0],df['tau'].values[0],eta,2*l)
        elif xvar == 'nconfig':
            titlename = '$r_s = %d, \\tau = %.2f, (\eta,U)=(%.2f,%.2f)$' %(df['r_s'].values[0],df['tau'].values[0],eta,2*l)
            xlab = 'N_w'
        elif xvar == '1/nconfig' or xvar=='1/Nw':
            titlename = '$r_s = %d, \\tau = %.2f, (\eta,U)=(%.2f,%.2f)$' %(df['r_s'].values[0],df['tau'].values[0],eta,2*l)
            xlab = '1/N_w'
        elif xvar == 'tau':
            titlename = '$r_s = %d, (\eta,U)=(%.2f,%.2f)$' %(df['r_s'].values[0],eta,2*l)
            xlab = '\\tau'
        elif xvar == 'Ncut':
            xlab = 'N_{cut}'
            titlename = '$r_s = %d, N_w = %d, \\tau=%.2f, (\eta,U)=(%.2f,%.2f)$' %(df['r_s'].values[0],df['nconfig'].values[0],df['tau'].values[0],eta,2*l)
        elif xvar == 'kcut':
            xlab = 'k_{cut}'
            titlename = '$r_s = %d, N_w = %d, \\tau=%.2f, (\eta,U)=(%.2f,%.2f)$' %(df['r_s'].values[0],df['nconfig'].values[0],df['tau'].values[0],eta,2*l)
        else:
            titlename = '$(\eta,U)=(%.2f,%.2f)$' %(eta,2*l)
            xlab = xvar

    ax.set_title(titlename)
    ax.set_xlabel('$%s$' %xlab)
    ax.set_ylabel('$E_{GS}$')
    ax.legend()
    plt.tight_layout()
    plt.show()

def JelliumComp():
    df = pd.read_csv('jellium/jell_rs4_tproj128_nconfig512_data/reblocked_eta0.00_U1.00_tequil20.csv')
    pyqmc_df = pd.read_csv('jellium/pyQMC_reblocked_tequil_20.csv')
    xs = df['tau'].values
    idxs = np.argsort(xs)
    xs = xs[idxs]
    eta = df['eta'].values[0]
    l = df['l'].values[0]
    E_gs = df['eavg'].values[idxs]
    E_err = df['err'].values[idxs]
    pyx = pyqmc_df['tau'].values
    idxs = np.argsort(pyx)
    pyx = pyx[idxs]
    pyE = pyqmc_df['eavg'].values[idxs]
    py_err = pyqmc_df['err'].values[idxs]
    fig,ax = plt.subplots(1,1)
    ax.errorbar(np.sqrt(xs),E_gs,yerr=E_err,label='my jellium DMC')
    ax.errorbar(np.sqrt(pyx),pyE,yerr=py_err,label='pyQMC')
    ax.set_xlabel('$\sqrt{\\tau}$')
    ax.set_ylabel('$E$')
    titlename = '$r_s = %d, N_w = %d$' %(df['r_s'].values[0],df['nconfig'].values[0])
    ax.set_title(titlename)
    ax.legend()
    plt.tight_layout()
    plt.show() 

def FitData(xvals, yvals, yerr=[], fit='lin', extrap=[],guess=[1,1],varlabs=['E','t']):
    def fitlinear(x,a,b):
        f = a*x + b 
        return f

    bnds = ([-20,-20],[20,20]) #bounds for weak coupling fit
    if len(yerr) > 0:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, sigma=yerr) #, p0=guess,bounds=bnds)
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
        r'$%s(%s) = a%s + b$' % (varlabs[0],varlabs[1],varlabs[1]),
        r'$a=%.4f \pm %.4f$' % (a, aerr),
        r'$b=%.6f \pm %.6f$' % (b, berr)
        ))

    print(textstr)
    return ans, textstr

def E_timelapse(filenames,gth=False,tequil=None,labels=[]):
    fig,ax = plt.subplots(1,1,figsize=(7,5))
    for i,name in enumerate(filenames):
        df = pd.read_csv(name)
        h = h5py.File(os.path.splitext(name)[0] + '.h5','r')
        steps = df['step'].values
   
        Eloc = df['elocal'].values
        Eloc= np.array([complex(val) for val in Eloc])
        tau = h.get('meta/tau')[0,0]
        if len(labels)!=0:
            ax.plot(steps*tau,Eloc.real,label=labels[i]) 
        else:    
            ax.plot(steps*tau,Eloc.real,label='$E_{mix}$') 
        if gth:
            ax.plot(steps*tau,df['egth'].values,'r.',label='$E_{gth}$')
        if tequil is None:
            tequil = h.get('meta/Nsteps')[0,0]/3
        ax.axvline(tequil,color='green',linestyle='dotted')
        ax.set_xlabel('sim time')
        ax.set_ylabel('E (ha)')
        if h.get('meta/ph_bool')[0,0] == 0:
            jell=True
        else: jell=False
        ax.set_title('$(\eta,U)=(%.2f,%.2f),\,N_{cut}=%d,\,r_s=%d$, jell=%d' %(h.get('meta/eta')[0,0],2*h.get('meta/l')[0,0],h.get('meta/N_cut')[0,0],h.get('meta/rs')[0,0],jell))
    ax.legend() 
    plt.tight_layout()
    plt.show()
  
def Elec_sep_dist(filenames,tequil=None,labels=[],fit=False,avg=False):
    from h5_plotting import Get_h5_steps
    if len(labels)==0:
        labels = np.full(len(filenames),'',dtype=str)
    fig,ax = plt.subplots(1,1,figsize=(5,7))

    if avg:
        diff_avg = np.zeros(pd.read_csv(filenames[0])['dists'].values.shape)
        print(diff_avg.shape)
        diff_err_avg = np.zeros(diff_avg.shape)
        eph_avg = np.zeros(diff_avg.shape)
        jell_avg = np.zeros(diff_avg.shape)
        dct = 0; ect = 0; jct = 0;

    for i,name in enumerate(filenames):
        df = pd.read_csv(name)
        h = h5py.File(os.path.splitext(name)[0] + '.h5','r')
        steps = df['step'].values
        if 'dists' in df.keys():
            dists = df['dists'].values
            d_err = df['d_err'].values
        else:
            _,_,_,_,dists=Get_h5_steps('',f=h)    
        print(h.get('meta/drift_diffusion')[0])
        tau = h.get('meta/tau')[0,0]
        Nw = h.get('meta/nconfig')[0,0]
        Nsteps = h.get('meta/Nsteps')[0,0]
        print('nsteps',Nsteps)
        ph = h.get('meta/ph_bool')[0,0]
        r_s = h.get('meta/rs')[0,0]
        diff = h.get('meta/diffusion')[0,0]
        ts = steps*tau
        if tequil is None: tequil = Nsteps*tau/3
        print(tequil)
        nequil = int(tequil/tau)
        print(tau)

        if diff == 1:
            labels[i] = 'diffusion'
            color='magenta'
            if avg:
                diff_avg = diff_avg + dists
                diff_err_avg = diff_err_avg + d_err**2
                dct = dct + 1
            if fit==True and avg==False:
                linfit,txt = FitData(np.sqrt(ts),dists,yerr=d_err,guess=[5,5],varlabs=['r','\sqrt{t}'])
                ax.plot(np.sqrt(ts),linfit,'r-',zorder=10,label='diff fit')
                ax.text(0.3,0.75, txt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                print(txt)
        elif diff == 0 and ph == 0: 
            labels[i] = 'jellium'
            if avg:
                jell_avg = jell_avg + dists
                jct = jct + 1
            color='blue'
        else:
            labels[i]= 'elec+ph'
            if avg:
                eph_avg = eph_avg + dists
                ect = ect + 1
            color = 'green'
        ax.plot(np.sqrt(ts),dists,label=labels[i],color=color)

    #ax.legend()
    ax.axvline(np.sqrt(tequil),color='g',linestyle='--')
    ax.set_xlabel('$\sqrt{t}$ (sim time$^{1/2}$)')
    ax.set_ylabel('$|\\vec r_{12}|$')
    ax.set_title('$r_s=%d,N_w=%d$' %(r_s,Nw))
    plt.tight_layout()
    
    if avg:
        print(dct,jct,ect)
        fig2,ax2 = plt.subplots(1,1,figsize=(5,7))
        if dct > 0:
            diff_avg = diff_avg/dct
            diff_err_avg = np.sqrt(diff_err_avg)/dct
        if ect > 0:
            eph_avg = eph_avg/ect
        if jct > 0:
            jell_avg = jell_avg/jct

        ax2.plot(np.sqrt(ts),diff_avg,color='magenta',label='diff (avg)')
        ax2.plot(np.sqrt(ts),jell_avg,color='blue',label='jell (avg)')
        ax2.plot(np.sqrt(ts),eph_avg,color='green',label='eph (avg)')
        if fit:
            linfit,txt = FitData(np.sqrt(ts),diff_avg,yerr=diff_err_avg,guess=[5,5],varlabs=['r','\sqrt{t}'])
            ax2.plot(np.sqrt(ts),linfit,'r-',zorder=10,label='diff fit')
            ax2.text(0.3,0.75, txt, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.axvline(np.sqrt(tequil),color='g',linestyle='--')
        ax2.set_xlabel('$\sqrt{t}$ (sim time$^{1/2}$)')
        ax2.set_ylabel('$<|\\vec r_{12}|>$')
        ax2.set_title('$r_s=%d$, $N_w=$%d walkers, %d trials' %(r_s,Nw,max([dct,ect,jct])))
        ax2.legend()
        plt.tight_layout() 
    plt.show()

def PhononMomDensityTimelapse(filenames,k=1):
    from h5_plotting import Get_h5_steps
    fig,ax = plt.subplots(1,1,figsize=(7,5))
    for i,name in enumerate(filenames):
        df = pd.read_csv(name)
        h = h5py.File(os.path.splitext(name)[0] + '.h5','r')
        steps = df['step'].values
        tau = h.get('meta/tau')[0,0]
        r_s = h.get('meta/rs')[0,0]
        Nw = h.get('meta/nconfig')[0,0]
        if 'n_ks' in df.keys():
            n_ks = df['n_ks'].values
            n_err = df['n_err'].values
            ts = tau*steps
        
            print(n_ks.shape)
            n_ks = np.array([val[k] for val in n_ks])
            print(n_ks.shape)
            ax.plot(ts,n_ks) 
        else:
            _,_,_,n_ks,_=Get_h5_steps('',f=h)    
            savestep = h.get('meta/savestep')[0,0]
            arrstep = h.get('meta/arrstep')[0,0] 
            ts = tau*steps[0::int(arrstep/savestep)]
            sstep = steps[0::int(arrstep/savestep)]
            print(sstep[:5])
            ax.plot(ts,n_ks.real[:,k],'b',label='real')
            ax.plot(ts,n_ks.imag[:,k],'r--',label='imag')
    ax.set_xlabel('t')
    ax.set_ylabel('$n_k$')
    ax.legend()
    ax.set_title('$r_s=%d$, Nconfig=%d, Nsteps=%d' %(r_s,Nw,max(steps)))
    plt.tight_layout()
    plt.show() 

if __name__=="__main__":
    filenames = sys.argv[1:]
    print(filenames)
    #labels=['$t_{proj}=64,N_w=32$','$t_{proj}=128,N_w=32$']
    #labels=['$N=5$','N=10','N=15','N=20']
    #labels=['popstep=10,$\\tau=.75$','popstep=10,$\\tau=5$','popstep=10,$\\tau=2$','popstep=20,$\\tau=.75$','popstep=2,$\\tau=.75$']
    labels=['popstep=2,$\\tau=.75$','popstep=20,$\\tau=.75$','popstep=10,$\\tau=.75$']
    tequil=1800
    #E_timelapse(filenames,tequil=tequil,labels=labels)
    #E_vs_var(filenames,xvar='1/nconfig',comp=True)
    Elec_sep_dist(filenames,fit=True,tequil=tequil,avg=True)
    #JelliumComp()
    #PhononMomDensityTimelapse(filenames,k=1)
