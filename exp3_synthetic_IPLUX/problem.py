#%%

import numpy as np
import cvxpy as cp
import os

import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class Synthetic:
    '''
    Synthetic problem in [Wu et al., TAC 23] (IPLUX)
    
    problem:  
        minimize \sum_{i} x_i^T P_i x_i + Q_i.T x_i +(one norm)
        subject to 
                 ||x_i -a_i||^2 <= c_i (h_i as indicator funcion)
                 
                 \sum_{i} ||x_i -a_i'||^2 - c_i' <= 0 
                 \sum_{i} A_ix_i = 0
                 \sum_j\in sparse_in[i] ||x_j - a_ij||^2 -c_ij <= 0, for all i in spare_in
                 \sum_j\in sparse_eq[j] A_ij^s x_j = 0 , for all i in sparse_eq
    '''
    
    prob_tpye = "Synthetic"
    
    def __init__(self, parameters, debug=False):
        '''
        'parameters' is a dictionary, contains following keys:
        N: number of nodes
        d: dimension of xi's
        m: number of inequalities
        p: number of equalities, i.e., shape of A_i's is (p, d)
        
        problem data:
        Q shape: (N, d)
        P shape: (N, d, d)
        A shape: (N, p, d)
        a shape: (N, d)
        c shape: (N, )
        aa shape: (N, d)
        cc shape: (N, )
        
        x_star: optimal solution
        opt_val: optimal value
        '''
        
        self.N = parameters['N']
        self.d = parameters['d']
        self.m = parameters['m']
        self.p = parameters['p']
        
        self.Q = np.zeros((self.N, self.d))
        self.P = np.zeros((self.N, self.d, self.d))
        self.A = np.zeros((self.N, self.p, self.d))
        self.a = np.zeros((self.N, self.d))
        self.c = np.zeros((self.N))
        self.aa = np.zeros((self.N, self.d))
        self.cc = np.zeros((self.N))
        
        self.prob = 0. # to set the cvxpy porblem
        
        self.x_star = np.zeros((self.N, self.d))
        self.opt_val = 0.
        
        # save data to the corresponding folder
        self.prob_name = f"N{self.N}" # name of problem instance
        self.save_dir = f'data/problem/{self.prob_tpye}/{self.prob_name}'
        self.debug = debug
        

    def gen(self):
        ''' Generate a Synthetic problem, then save it in the data folder. '''

        # ============ generate and save problem data ====================
        
        logging.info(f"generating a {self.prob_tpye} problem: N={self.N}, " \
            f"d={self.d}, m={self.m}, p={self.p}")
        
        if self.debug:
            self.generate_debug_exampe()
        else:        
            self.generate_objective()
            self.generate_constraints()
        # print(self.A.shape)
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        np.save(f'{self.save_dir}/Q.npy', self.Q)
        np.save(f'{self.save_dir}/P.npy', self.P)
        # np save does not know captial letter?
        np.save(f'{self.save_dir}/Amat.npy', self.A) 
        np.save(f'{self.save_dir}/a.npy', self.a)
        np.save(f'{self.save_dir}/c.npy', self.c)
        np.save(f'{self.save_dir}/aa.npy', self.aa)
        np.save(f'{self.save_dir}/cc.npy', self.cc)
        # print(A.shape)

        # ============= use cvxpy to solve the problem ===================
        
        # obj = \sum_{i} x_i^T P_i x_i + Q_i.T x_i +(one norm)        
        obj = 0.
        var_x = cp.Variable((self.N, self.d))
        for i in range(self.N):
            obj += cp.quad_form(var_x[i], self.P[i]) \
                    + self.Q[i].T @ var_x[i] \
                    + cp.norm(var_x[i], 1)

        # constraints
        # Local set:          ||x_i -a_i||^2 <= c_i
        # Global inequality:  \sum_{i} ||x_i -a_i'||^2 - c_i' <= 0
        # Global equality:    \sum_{i} A_ix_i = 0
        local_cons = []
        coupling_ineq = 0.
        coupling_eq = 0.
        for i in range(self.N):
            local_cons += [cp.norm(var_x[i] - self.a[i])**2 <= self.c[i]]
            coupling_ineq += cp.norm(var_x[i] - self.aa[i])**2 - self.cc[i]
            coupling_eq += self.A[i] @ var_x[i]
        all_cons = local_cons + [coupling_ineq <= 0, coupling_eq == 0]
        
        self.prob = cp.Problem(cp.Minimize(obj), all_cons)
        assert(self.prob.is_dcp())
        
        # self.prob.solve(solver='MOSEK', verbose=True)
        self.prob.solve(solver='MOSEK')
        self.x_star = var_x.value
        self.opt_val = self.prob.value
        
        logging.info(f'x* {self.x_star.shape}, f* {self.opt_val}')

        # =========================== then save ==============================
        np.save(f'{self.save_dir}/x_star.npy', self.x_star)
        np.save(f'{self.save_dir}/opt_val.npy', [self.opt_val])
        
        logging.info(f"generated problem saved in {self.save_dir}\n")
    
    def load(self):
        print(f"loading a {self.prob_tpye} problem, N={self.N}")
        try:
            self.Q = np.load(f'{self.save_dir}/Q.npy')
            self.P = np.load(f'{self.save_dir}/P.npy')
            self.A = np.load(f'{self.save_dir}/Amat.npy')
            self.a = np.load(f'{self.save_dir}/a.npy')
            self.c = np.load(f'{self.save_dir}/c.npy')
            self.aa = np.load(f'{self.save_dir}/aa.npy')
            self.cc = np.load(f'{self.save_dir}/cc.npy')
            self.x_star = np.load(f'{self.save_dir}/x_star.npy')
            self.opt_val = np.load(f'{self.save_dir}/opt_val.npy')[0]
            
            # reset parameters
            self.N = self.Q.shape[0]
            self.d = self.Q.shape[1]
            self.m = 1
            self.p = self.A.shape[1]
            
            print("problem loaded:")
            logging.info(f'Q: {self.Q.shape}, P: {self.P.shape}')
            logging.info(f'A: {self.A.shape}')
            logging.info(f'a: {self.a.shape}, c: {self.c.shape}')
            logging.info(f'aa: {self.aa.shape}, ca: {self.cc.shape}')
            logging.info(f'x_star {self.x_star.shape}')
            logging.info(f'opt_val {self.opt_val}')
        except:
            print("failed to load data in", self.save_dir)
            raise(ValueError)
        
    def generate_objective(self):
        '''
        P_i: shape = (d, d) = (3, 3)
        Q_i: shape = (d) = (3,)
        '''
        for i in range(self.N):
            Q,R = np.linalg.qr(np.random.rand(self.d,self.d))
            diag_elem = np.random.rand(self.d)
            diag_elem[-1] = 1e-15 
            self.P[i] = Q.T@np.diag(diag_elem)@Q
            # logging.info(f"check semipositivity of P{i}: {min(np.linalg.eigvals(self.P[i]))}")
            self.Q[i] = np.random.rand(self.d)*3

        logging.info(f'Q: {self.Q.shape}, P: {self.P.shape}')
        
    def generate_constraints(self):
        '''
        Inequality:
            Local set:          ||x_i -a_i||^2 <= c_i
            Global inequality:  \sum_{i} ||x_i -a_i'||^2 - c_i' <= 0
                    
                    - a_i: shape = (d, )
                    - c_i: scalar
                    - aa_i: shape = (d, )
                    - cc_i: scalar

        Equality:
            Global equality:    \sum_{i} A_ix_i = 0
                    
                    - A_i: shape = (m, d) = (5, 3)
        '''
        
        for i in range(self.N):
            self.A[i] = np.random.rand(self.p, self.d)
            self.a[i] = np.random.rand(self.d)
            self.c[i] = np.random.rand() + self.a[i].T @ self.a[i]
            self.aa[i] = np.random.rand(self.d)
            self.cc[i] = np.random.rand() + self.aa[i].T @ self.aa[i]
        
        logging.info(f'A: {self.A.shape}')
        logging.info(f'a: {self.a.shape}, c: {self.c.shape}')
        logging.info(f'aa: {self.aa.shape}, cc: {self.cc.shape}')
        
    def generate_debug_exampe(self):
        # N: 1
        # d: 1
        # p: 1
        # m: 1
        self.P[0,0,0] = 1
        self.Q[0,0] = 2
        self.A[0,0,0] = 4
        self.a[0,0] = 1
        self.c[0] = 2
        self.aa[0,0] = 2
        self.cc[0] = 5