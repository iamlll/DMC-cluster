import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def E_vs_rs(df):
    ''' Use reblocked csv file (dmc_reblock.py). Plot E vs r_s (sys density) for fixed eta, U (l), etc. '''
    rs = df['r_s'].values
    eta = df['eta'].values[0]
    l = df['l'].values[0]
    alpha = (1-eta)*l
    E_gs = df['eavg'].values
    E_err = df['err'].values
    fig,ax = plt.subplots(1,1)
    ax.errorbar(rs,E_gs,yerr=E_err)
    print(E_err)
    print(np.diff(E_gs))
    ax.set_xlabel('$r_s$')
    ax.set_ylabel('$E_{GS}$')
    plt.show()

if __name__=="__main__":
    df = pd.read_csv(sys.argv[1])
    E_vs_rs(df)
