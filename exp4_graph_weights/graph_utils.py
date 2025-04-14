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
