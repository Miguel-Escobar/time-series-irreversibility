from ts2vg import HorizontalVG
import numpy as np
import matplotlib.pyplot as plt
ts = np.loadtxt('logmap_ts.dat')[:,1]
q = np.concatenate((np.linspace(0, 1, 2000, endpoint=False), -np.linspace(-5, -1, 4000, endpoint=False)[::-1]))

def dist_degree_in(g):

    ks, counts = np.unique(g.degrees_in, return_counts=True)
    ps = counts/sum(counts)

    return ks, ps

def dist_degree_out(g):

    ks, counts = np.unique(g.degrees_out, return_counts=True)
    ps = counts/sum(counts)

    return ks, ps

g = HorizontalVG(directed='left_to_right').build(ts, only_degrees=True)

print('graph generated')
ks_in, ps_in = dist_degree_in(g)
ks_out, ps_out = dist_degree_out(g)

KLD = 0
RKD = np.zeros_like(q)

for i in range(len(ks_in)):
    print(i)
    if ks_in[i] in ks_out:
        j = np.where(ks_out == ks_in[i])
        KLD += ps_out[j] * np.log(ps_out[j]/ps_in[i])
        RKD += (ps_out[j]**q )* (ps_in[i]**(1-q))

LVD = (1 - 1/RKD)/(q-1)
RD = np.log(RKD)/(q-1)
RKD = (1-RKD)/(1-q) 

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111)
ax.plot(ks_in, ps_in, label=r'$P_{in}(k)$', marker='o', ms=2)
ax.plot(ks_out, ps_out, label=r'$P_{out}(k)$', marker='o', ms=2)
ax.set_xlabel('k')
ax.set_ylabel(r'$P(k)$')
ax.set_yscale('log')
ax.legend()
fig.show()

fig2 = plt.figure(2)
fig2.clf()
ax2 = fig2.add_subplot(111)
ax2.plot(q, RKD, label='RKD')
ax2.plot(q, RD, label='RD')
ax2.plot(q, LVD, label='LVD')
ax2.axhline(KLD, label='KLD', color='red')
ax2.set_xlabel('q')
ax2.set_ylabel(r'$D(P_{out}||P_{in})$')
ax2.set_yscale('log')
ax2.legend()
fig2.show()
