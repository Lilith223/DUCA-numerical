import numpy as np
import networkx as nx
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import cvxpy as cp

def get_metropolis_weight(A):
    '''
    Generate metropolis weight matrix (with eps=1) from adjacency matrix.
    '''
    N = A.shape[0]
    metropolis = np.zeros((N, N))
    d = np.zeros(N)
    for i in range(N):
        d[i] = np.sum(A[i,:])
    for i in range(N):
        for j in range(N):
            if A[i, j]==1:
                metropolis[i,j] = 1/(1+max(d[i],d[j]))
        metropolis[i,i] = 1 - sum(metropolis[i,:])
    return metropolis

def get_lazy_metropolis_weight(A):
    '''
    Generate lazy metropolis weight matrix from adjacency matrix.
    '''
    N = A.shape[0]
    lm = np.zeros((N, N))
    d = np.zeros(N)
    for i in range(N):
        d[i] = np.sum(A[i,:])
    for i in range(N):
        for j in range(N):
            if A[i, j]==1:
                lm[i,j] = 1/(2*max(d[i],d[j]))
        lm[i,i] = 1 - sum(lm[i,:])
    return lm

def get_max_degree_weight(A):
    '''
    Generate max degree weight matrix (with eps=1) from adjacency matrix.
    '''
    N = A.shape[0]
    md = np.zeros((N, N))
    
    d = np.zeros(N)
    for i in range(N):
        d[i] = np.sum(A[i,:])
    d_max = np.max(d)
    
    for i in range(N):
        for j in range(N):
            if A[i, j]==1:
                md[i,j] = 1/(d_max + 1)
        md[i,i] = 1 - sum(md[i,:])
    return md

# def get_doubly_stochastic(metropolis):
#     '''
#     Generate doubly stochastic matrix from metropolis weight matrix.
#     '''
#     return np.identity(metropolis.shape[0]) - metropolis

def graph_gen(num_node, num_edge, B=1):
    ''' Generate graph data, save them in the network folder
    
    Keyword arguments:
    num_node -- number of nodes
    num_edge -- number of edges
    B --  B-connected time-varying graph (default 1)
    '''
    
    adj_subG, G = gen_B_conn_graphs(num_node, num_edge, B)
    adj_G = nx.to_numpy_array(G)
    print(G)

    save_dir = f'data/graph/N{num_node}E{num_edge}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(f'{save_dir}/graph_adj.npy', adj_G)
    np.save(f'{save_dir}/subgraph_adj.npy', adj_subG)

    plt.figure()
    nx.draw_networkx(G, pos=nx.circular_layout(G), arrowsize=3, node_size=50, linewidths=0.2, width=0.2, with_labels=False)
    plt.savefig(f'{save_dir}/network.png')
    plt.savefig('network.png')

    # Now transform the above sub_adjs to doubly stochastic weight matrices.
    W = np.zeros((B, num_node, num_node))
    for b in range(B):  
        metropolis = get_metropolis(adj_subG[b])
        W[b] = get_doubly_stochastic(metropolis)
    np.save(f'{save_dir}/subgraph_W.npy', W[0])
    
    
def L_g(prob, d, i):
    x = cp.Variable(d)
    a_i = prob.a[i]
    c_i = prob.c[i]

    aa_i = prob.aa[i]
    cc_i = prob.cc[i]

    prob = cp.norm(2*(x-aa_i))
    constraint = [cp.norm(x-a_i)**2 <= c_i]
    prob = cp.Problem(cp.Minimize(prob),constraint)
    prob.solve()

    return prob.value



def find_alpha_lower_bound(param, prob):
    N = param['N']
    d = param['d']
    
    max_L_f = 0
    max_L_g = 0
    for i in range(N):
        P_i = prob.P[i]
        L_f = max(np.linalg.eigvals(2*P_i)) 
        if L_f >= max_L_f:
            max_L_f = L_f
        L_g_i = L_g(prob, d, i)
        if L_g_i > max_L_g:
            max_L_g = L_g_i

    lower = max_L_f + 1 + max_L_g**2
    
    return np.ceil(np.real(lower))