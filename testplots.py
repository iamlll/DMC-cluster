import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.optimize import curve_fit
import os
import h5py

def E_vs_var(filenames,xvar='r_s',labels=[],jellcomp=False):
    ''' Use reblocked csv file (dmc_reblock.py). Plot E vs r_s (sys density) for fixed eta, U (l), etc. '''
    fig,ax = plt.subplots(1,1)
    if len(labels) == len(filenames):
        islabel = True
    else: islabel = False
    print(islabel)
    for i,name in enumerate(filenames):
        df = pd.read_csv(name)
        if xvar == 'kcut':
            xs = df['Ncut'].values
            Ls = np.full(len(xs),(2*4*np.pi/3)**(1/3) * df['r_s'].values) #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
            xs = 2*np.pi*xs/Ls 
        elif xvar == '1/r_s':
            xs = 1./df['r_s'].values
        else:
            xs = df[xvar].values
        idxs = np.argsort(xs)
        xs = xs[idxs]
        eta = df['eta'].values[0]
        l = df['l'].values[0]
        alpha = (1-eta)*l
        E_gs = df['eavg'].values[idxs]
        E_err = df['err'].values[idxs]
        if islabel: ax.errorbar(xs,E_gs,yerr=E_err,label=labels[i])
        else: 
            lab = "$N_{cut}=%d, \\eta=%.2f$" %(df['Ncut'].values[0],eta)
            print(lab)
            ax.errorbar(xs,E_gs,yerr=E_err,label=lab)
        print(E_gs)
        print(E_err)
        if xvar == '1/r_s':
            titlename = '$N_w = %d, \\tau = %.2f, (\eta,U)=(%.2f,%.2f)$' %(df['nconfig'].values[0],df['tau'].values[0],eta,2*l)
            xlab = xvar
            if i == len(filenames)-1:
                idx = np.where(xs >= 0.15)[0]
                print(idx)
                fit,txt=FitData(xs[idx],E_gs[idx])
                ax.plot(xs[idx],fit,'r--',label='fit',zorder=10)
                ax.text(0.2, 0.4, txt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes) 
        elif xvar == 'r_s':
            xlab = xvar
            titlename = '$N_w = %d, \\tau = %.2f, (\eta,U)=(%.2f,%.2f)$' %(df['nconfig'].values[0],df['tau'].values[0],eta,2*l)
        elif xvar == 'nconfig':
            titlename = '$r_s = %d, \\tau = %.2f, (\eta,U)=(%.2f,%.2f)$' %(df['r_s'].values[0],df['tau'].values[0],eta,2*l)
            xlab = 'N_w'
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

def FitData(xvals, yvals, yerr=[], fit='lin', extrap=[]):
    def fitlinear(x,a,b):
        f = a*x + b 
        return f

    bnds = ([-10,-10],[5,5]) #bounds for weak coupling fit
    guess =[-1.5,-3]
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

def E_timelapse(filenames,gth=False,tequil=20):
    for name in filenames:
        df = pd.read_csv(name)
        h = h5py.File(os.path.splitext(name)[0] + '.h5','r')
        steps = df['step'].values
        Eloc = df['elocal'].values
        Eloc= np.array([complex(val) for val in Eloc])
        fig,ax = plt.subplots(1,1)
        ax.plot(steps,Eloc.real,label='$E_{mix}$') 
        if gth:
            ax.plot(steps,df['egth'].values,'r.',label='$E_{gth}$')
        ax.axvline(int(tequil/h.get('meta/tau')[0,0]),color='green',linestyle='dotted')
        ax.set_xlabel('steps')
        ax.set_ylabel('E (ha)')
        if h.get('meta/ph_bool')[0,0] == 0:
            jell=True
        else: jell=False
        ax.set_title('$(\eta,U)=(%.2f,%.2f),\,N_{cut}=%d,\,r_s=%d$, jell=%d' %(h.get('meta/eta')[0,0],h.get('meta/l')[0,0],h.get('meta/N_cut')[0,0],h.get('meta/rs')[0,0],jell))
        plt.tight_layout()
        plt.show()
  
if __name__=="__main__":
    filenames = sys.argv[1:] 
    #labels=['$t_{proj}=64,N_w=32$','$t_{proj}=128,N_w=32$']
    #labels=['$N=5$','N=10','N=15','N=20']
    labels=[]
    E_vs_var(filenames,xvar='tau',labels=labels)
    #E_vs_var(filenames,xvar='1/r_s',labels=labels)
    #E_timelapse(filenames)
    #JelliumComp()
