import numpy as np
import networkx as nx
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')

def gen_B_conn_graphs(num_of_nodes, num_of_edges, B, ring):
    '''
    Generate undirected graph with B connected subgraphs.
    Return the adjs of these subgraphs.
    '''

    # Generate connected undirected graph
    if ring: # works for even nodes
       G = nx.ring_of_cliques(num_cliques=num_of_nodes>>1, clique_size=2)
    else: 
        G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=False)
        while nx.is_connected(G) == False:
            G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=False)

    # all links in G
    G_links = np.array([e for e in G.edges])

    num_avg_links = math.ceil(num_of_edges / B) # Just a notation. link # of each subgraph is the multiple of it (*2 or *3 ...).
    edge_perm = np.random.permutation(num_of_edges)
    edge_indexSet = np.concatenate((edge_perm, edge_perm)) # repeat twice (may change)

    #subG_links = np.zeros((B, 2*num_avg_links, 2))
    adj_subG = np.zeros((B, num_of_nodes, num_of_nodes))

    for i in range(B): # 0 1 2 3
        start_link_idx = i*num_avg_links   # 0 250 500 750
        end_link_idx = (i+2)*num_avg_links # 500 750 1000 1250
        selected_links_idx = edge_indexSet[start_link_idx:end_link_idx]
        #subG_links[i] = G_links[selected_links_idx]

        for idx in selected_links_idx:
            adj_subG[i, G_links[idx,0], G_links[idx,1]] = 1
            adj_subG[i, G_links[idx,1], G_links[idx,0]] = 1

    return adj_subG, G

def get_metropolis(adjacency_matrix):
    '''
    Generate metropolis weight matrix from adjacency matrix.
    '''
    num_of_nodes = adjacency_matrix.shape[0]
    metropolis=np.zeros((num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if adjacency_matrix[i,j]==1:
                d_i = np.sum(adjacency_matrix[i,:])
                d_j = np.sum(adjacency_matrix[j,:])
                metropolis[i,j]=-1/(1+max(d_i,d_j))
        metropolis[i,i]=-sum(metropolis[i,:])
    return metropolis

def get_doubly_stochastic(metropolis):
    '''
    Generate doubly stochastic matrix from metropolis weight matrix.
    '''
    return np.identity(metropolis.shape[0]) - metropolis

def graph_gen(num_node, num_edge, B=1, ring=False):
    ''' Generate graph data, save them in the network folder
    
    Keyword arguments:
    num_node -- number of nodes
    num_edge -- number of edges
    B --  B-connected time-varying graph (default 1)
    '''
    
    adj_subG, G = gen_B_conn_graphs(num_node, num_edge, B, ring)
    adj_G = nx.to_numpy_array(G)
    print(G)

    save_dir = f'instance/graph/N{num_node}E{num_edge}'
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