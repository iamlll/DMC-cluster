import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('energy_chart_for_plotting.csv')
#print(data)
rs = data['rs'].values
L = data['L'].values
avgr = data['<r>'].values # sum of product of distances * weights - independent of choice of bins; sum(r*wt)/sum(wt)
grmax = data['avg_g_r_max'].values # max value of g(r) curves, averaged over different orientational cuts
avg_x_max = data['avg_x_max'].values # x-values corresponding to the maximal g(r) values (peak locations), averaged over different orientational cuts
avg_x_wtd = data['avg_x_wtd'].values # sum(g(r)*x)/sum(g(r))
etas = data['eta'].values
ls = data['l'].values

opt = 0
ma = rs == 30
ma110 = rs == 110

if opt == 0:
    cmat = avgr/L
    lab = '$\\langle r \\rangle=\\frac{1}{L} \\sum_i [w_i r_i] / \\sum_i w_i$'
elif opt == 1:
    cmat = grmax
    lab = '$\\max(g(r))$'
elif opt == 2:
    cmat = avg_x_max/L
    lab = '$\\frac{1}{L} r(\\max(g(r)))$'
elif opt == 3:
    cmat = avg_x_wtd/L
    lab = '$\\frac{1}{L} \\sum [g(r) r] / \\sum g(r)$'
fig,ax = plt.subplots(1,1,layout='constrained')
sc = ax.scatter(etas[ma],ls[ma],s=50,c=cmat[ma])
cb = fig.colorbar(sc,ax=ax)
cb.ax.set_ylabel(lab)
ax.set_xlabel('$\\eta$')
ax.set_ylabel('$l$')

'''
# plot line cut at l = 10, rs = 30
mal10 = ls == 10
f2,a2 = plt.subplots(1,1,layout='constrained')
a2.plot(etas[ma & mal10],cmat[ma & mal10],'o',label='$r_s=30$')
# plot line cut at l = 10, rs = 110
a2.plot(etas[ma110 & mal10],cmat[ma110 & mal10],'o',label='$r_s=110$')
#a2.axhline(np.sqrt(2)/2,linestyle='--',color='k')
a2.set_xlabel('$\\eta$')
a2.set_ylabel(lab)
a2.legend()
'''

# overlay the variational wave function solution's 0-energy contour as contrast.
# Need to fit a line to the bottom to get it to extend all the way to eta = 0
dat = np.load('nakano_fullPD_zerocontour_etaUcoords.npy')
print(dat.shape)


plt.show()

