from ts2vg import HorizontalVG
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

q = .9
r = np.linspace(3.5, 4, 1001)
N_values = 10**3
n_trans = 10**4

def dist_degree_in(g):

    ks, counts = np.unique(g.degrees_in, return_counts=True)
    ps = counts/np.sum(counts)

    return ks, ps

def dist_degree_out(g):

    ks, counts = np.unique(g.degrees_out, return_counts=True)
    ps = counts/np.sum(counts)

    return ks, ps

x0 = .4
x0log = x0*np.ones(len(r)) 
logmap = np.zeros((N_values, len(r)))
n = 0

while n < n_trans:
    x0log = r*x0log*(1-x0log)
    n += 1

logmap[0] = x0log

for i in range(N_values-1):
    logmap[i+1] = r*logmap[i]*(1-logmap[i])

ts = np.around(logmap, decimals=15)

KLD = np.zeros(len(r))
suma = np.zeros(len(r))
lyapunov = np.zeros(len(r))


for k in range(len(r)):

    lyapunov[k] = np.mean(np.log(np.abs((r - 2*r*ts)[:, k])))
    g = HorizontalVG(directed='left_to_right').build(ts[:,k], only_degrees=True)
    ks_in, ps_in = dist_degree_in(g)
    ks_out, ps_out = dist_degree_out(g)

    for i in range(len(ks_in)):
        if ks_in[i] in ks_out:
            j = np.where(ks_out == ks_in[i])
            KLD[k] += ps_out[j] * np.log(ps_out[j]/ps_in[i])
            suma[k] += (ps_out[j]**q)*(ps_in[i]**(1-q))

LVD = (1 - 1/suma)/(q-1)
RD = np.log(suma)/(q-1)
RKD = (1-suma)/(1-q)

peaks_lyapunov, _ = find_peaks(-lyapunov, prominence=.1)
peaks_lvd, _ = find_peaks(LVD, prominence=.2)
coincidences = []

for i in range(len(peaks_lyapunov)):
    if peaks_lyapunov[i] in peaks_lvd:
        coincidences.append(peaks_lyapunov[i])

coincidences = np.array(coincidences)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(211)

ax.plot(r, LVD, label=r'LVD')
ax.plot(r, RKD, label=r'RKD')
ax.plot(r, RD, label=r'RD')
ax.plot(r, KLD, label=r'KLD')
ax.plot(r[coincidences], LVD[coincidences], 'x')
ax.set_xlabel('r')
ax.set_ylabel(r'$D(P_{out}||P_{in})$')
ax.legend(prop={'size': 7})

ax2 = fig.add_subplot(212)
ax2.axhline(0, color='red', ls='--')
ax2.plot(r, lyapunov)
ax2.plot(r[coincidences], lyapunov[coincidences], 'x')
ax2.set_xlabel('r')
ax2.set_ylabel('Lyapunov Exponent')

fig.show()