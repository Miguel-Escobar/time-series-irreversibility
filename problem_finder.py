from ts2vg import HorizontalVG
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
q = 1.5
r = np.linspace(1, 4, 10**3)
N_values = 1000
n_trans = 10**5
scaling = 1
i_problem = 1

def dist_degree_in(g):

    ks, counts = np.unique(g.degrees_in, return_counts=True)
    ps = counts/np.sum(counts)

    return ks, ps

def dist_degree_out(g):

    ks, counts = np.unique(g.degrees_out, return_counts=True)
    ps = counts/np.sum(counts)

    return ks, ps

x0 = .5
x0log = x0*np.ones_like(r) 
logmap = np.zeros((N_values, len(r)))
n = 0

while n < n_trans:
    x0log = r*x0log*(1-x0log)
    n += 1

logmap[0] = x0log

for i in range(N_values-1):
    logmap[i+1] = r*logmap[i]*(1-logmap[i])

ts = logmap

KLD = np.zeros(len(r))
suma = np.zeros(len(r))

problems = []

for k in range(len(r)):

    g = HorizontalVG(directed='left_to_right').build(ts[:,k])

    ks_in, ps_in = dist_degree_in(g)
    ks_out, ps_out = dist_degree_out(g)

    for i in range(len(ks_in)):

        if ks_in[i] in ks_out:
            j = np.where(ks_out == ks_in[i])
            KLD[k] += ps_out[j] * np.log(ps_out[j]/ps_in[i])
            suma[k] += (ps_out[j]**q)*(ps_in[i]**(1-q))
    
    if (suma[k] < 1 or suma[k]>1.2) and r[k]<3.6:
        print('PROBLEM NUMBER ' + str(len(problems)) + ' IN r=' + str(r[k]))
        problems.append((k, suma[k], g))


LVD = (1 - 1/suma)/(q-1)
RD = np.log(suma)/(q-1)
RKD = (1-suma)/(1-q)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(111)

ax.plot(r, KLD, label=r'KLD')
ax.plot(r, LVD, label=r'LVD')
ax.plot(r, RKD, label=r'RKD')
ax.plot(r, RD, label=r'RD')
ax.set_xlabel('r')
ax.set_ylabel(r'$D(P_{out}||P_{in})$')
ax.legend()
fig.show()

option = input('To stop type "stop", Problem to analyze: ')

while option != 'stop':
    i_problem = int(option)
    graph = problems[i_problem][2]
    k = problems[i_problem][0]
    nxg = graph.as_networkx()
    nodepos = graph.node_positions()

    for i in range(N_values):
        lst = list(nodepos[i])
        lst[1] += -ts[0, k]
        nodepos[i] = tuple(lst)
    
    ts_to_plot = (ts[:, k] - ts[0, k])

    xin , yin = dist_degree_in(graph)
    xout, yout = dist_degree_out(graph)

    print("SUMA = " + str(problems[i_problem][1]))

    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    graph_plot_options = {
        'with_labels': False,
        'node_size': 2,
        'node_color': [(0, 0, 0, 1)],
        'edge_color': [(0, 0, 0, 0.15)],
    }

    nx.draw_networkx(nxg, ax=ax2, pos=nodepos, **graph_plot_options)

    ax2.tick_params(bottom=True, left=True, labelleft=True, labelbottom=True)
    ax2.plot(ts_to_plot, linewidth=.5)
    ax2.set_title('Time Series')
    ax2.set_ylim((np.min(ts_to_plot)*1.5,np.max(ts_to_plot)*1.5))
    fig2.show()

    fig3 = plt.figure("Probability distribution")
    fig3.clf()
    ax3 = fig3.add_subplot(111)
    ax3.plot(xin, yin, label='P(k) in', marker='o', markersize=2)
    ax3.plot(xout, yout, label="P(k) out", marker='o', markersize=2)
    ax3.legend()
    fig3.show()

    option = input('stop? ')