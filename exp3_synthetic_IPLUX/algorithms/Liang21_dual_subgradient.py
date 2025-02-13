import numpy as np
import cvxpy as cp
from math import log, sqrt
from time import time
import os
import scipy.linalg

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class dual_subgradient:
    '''
    problem data:
        Q shape: (N, d)
        P shape: (N, d, d)
        A shape: (N, p, d)
        a shape: (N, d)
        c shape: (N, )
        aa shape: (N, d)
        cc shape: (N, )
        
    algorithm parameters:
        alpha (for UDC_prox)
        A, A_half, H1, H2, H2_half, D: (N, N) weight matrices. 
            H1 for H, H2 for H tilde
    '''
    
    name = "dual subgradient"
    
    def __init__(
        self, 
        prob, 
        network, 
        gamma,
        verbose=0):
        '''
        these are set here:
            problem parameters:
                N: number of nodes
                d: dimension of xi's
                m: number of inequalities
                p: number of equalities, i.e., shape of A_i's is (p, d)
                x_star: N dimensional vector
                opt_val: float
            
            problem data:
                Q shape: (N, d)
                P shape: (N, d, d)
                P_sqrt shape: (N, d, d)
                A_data shape: (N, p, d)
                a shape: (N, d)
                c shape: (N, )
                aa shape: (N, d)
                cc shape: (N, )
            
            algorithm parameters:
                gamma: stepsize parameter
                W: doubly stochatic weight matrix
                    
            verbose: displays information
            log_dir
            file_prefix
        
        
        these are set in 'reset()':
            iter_num: iteration number
            self.init_time = time()
            
            x_cur, x_nxt, x_avg: (N, d)
            y_mu_cur, y_mu_nxt: (N, m)
            y_lam_cur, y_lam_nxt: (N, p)
            z_mu_cur, z_mu_nxt: (N, m)
            z_lam_cur, z_lam_nxt: (N, p)
            
            init_time
        '''

        self.N = prob.N
        self.d = prob.d
        self.p = prob.p
        self.m = prob.m
        self.x_star = prob.x_star
        self.opt_val = prob.opt_val
        
        self.Q = prob.Q 
        self.P = prob.P 
        self.P_sqrt = np.zeros(self.P.shape)
        for i in range(self.N):
            self.P_sqrt[i] = scipy.linalg.sqrtm(self.P[i])
        self.A_data = prob.A 
        # print(prob.A.shape)
        self.a = prob.a 
        self.c = prob.c 
        self.aa = prob.aa 
        self.cc = prob.cc 
        
        self.gamma = gamma
        self.W = network
        
        self.verbose = verbose
        self.log_dir = f'log/N{self.N}'
        
        self.file_prefix = f'dual_subgradient_g{self.gamma}'
        
        self._set_argmin_prob()
        self.reset()

    def reset(self):
        '''
        iter_num = 0
        self.init_time = time()
        
        decision vars:
        x_cur, x_nxt, x_avg: (N, d)
        y_mu_cur, y_mu_nxt: (N, m)
        y_lam_cur, y_lam_nxt: (N, p)
        z_mu_cur, z_mu_nxt: (N, m)
        z_lam_cur, z_lam_nxt: (N, p)
        
        log lists:
        obj_err_log = []
        obj_err_avg_log = []
        cons_vio_log = []
        cons_vio_avg_log = []
        x_dis_log = []
        x_dis_avg_log = []
        '''
        
        logging.info('reset')
        
        self.iter_num = 0
        self.init_time = time()
        
        # initial conditions
        self.x_cur = np.zeros((self.N, self.d))
        # self.x_cur = self.x_star.copy()
        self.y_mu_cur = np.zeros((self.N, self.m))
        self.y_lam_cur = np.zeros((self.N, self.p))
        self.z_mu_cur = np.zeros((self.N, self.m))
        self.z_lam_cur = np.zeros((self.N, self.p))
        
        self.x_avg = self.x_cur.copy() # for running average
        
        self.x_nxt = np.zeros((self.N, self.d))
        self.y_mu_nxt = np.zeros((self.N, self.m))
        self.y_lam_nxt = np.zeros((self.N, self.p))
        self.z_mu_nxt = np.zeros((self.N, self.m))
        self.z_lam_nxt = np.zeros((self.N, self.p))
        
        
        # reset logs
        self.obj_err_log = []
        self.obj_err_avg_log = []
        self.cons_vio_log = []
        self.cons_vio_avg_log = []
        self.x_dis_log = []
        self.x_dis_avg_log = []

        self.make_log()
        if self.verbose:
            self.show_status()

    
    def make_log(self):
        ''' save log every 100 iterations '''
        
        # last iterate
        obj_err, cons_vio = self.compute_metrics()
        self.obj_err_log.append(obj_err)
        self.cons_vio_log.append(cons_vio)
        self.x_dis_log.append(np.linalg.norm(self.x_cur-self.x_star))
        
        # running average
        obj_err_avg, cons_vio_avg = self.compute_metrics(avg=True)
        self.obj_err_avg_log.append(obj_err_avg)
        self.cons_vio_avg_log.append(cons_vio_avg)
        self.x_dis_avg_log.append(np.linalg.norm(self.x_avg-self.x_star))
        
        # logging.info(f'iter {self.iter_num}, obj err: {obj_err:.2e}, cons vio: {cons_vio:.2e}')
        
        if self.iter_num%100==0:
            logging.info(f'{self.name} gamma {self.gamma}, iter {self.iter_num}, obj err: {self.obj_err_log[-1]:.2e}, cons vio: {self.cons_vio_log[-1]:.2e}')
            
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            
            # last iterate
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_oe.txt', self.obj_err_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_cv.txt', self.cons_vio_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_xd.txt', self.x_dis_log, delimiter=',')
            # running average
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_oe_avg.txt', self.obj_err_avg_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_cv_avg.txt', self.cons_vio_avg_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_xd_avg.txt', self.x_dis_avg_log, delimiter=',')
            
            logging.info(f"time {time()-self.init_time:.2f}, saved\n")

    def _set_argmin_prob(self):
        '''
        self.var_xi = cp.Variable(self.d)

        self.param_Pi = cp.Parameter((self.d, self.d)) # P_i
        self.param_Qi = cp.Parameter(self.d) # Q_i
        self.param_mu = cp.Parameter(self.m) # \mu_i^k
        self.param_mu_aai = cp.Parameter(self.d) # \mu_i^k aa_i
        self.param_Ai_lam = cp.Parameter(self.d) # A_i^T \lambda_i^k 
        
        self.param_aai = cp.Parameter(self.d)
        self.param_cci = cp.Parameter(1)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        self.param_Ai = cp.Parameter((self.p, self.d))
        '''
        
        self.var_xi = cp.Variable(self.d)

        self.param_Pi = cp.Parameter((self.d, self.d)) # P_i
        self.param_Qi = cp.Parameter(self.d) # Q_i
        self.param_mu = cp.Parameter(self.m, nonneg=True) # \mu_i^k
        self.param_mu_aai = cp.Parameter(self.d) # \mu_i^k aa_i
        self.param_Ai_lam = cp.Parameter(self.d) # A_i^T \lambda_i^k 
        
        self.param_aai = cp.Parameter(self.d)
        self.param_cci = cp.Parameter(1)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        self.param_Ai = cp.Parameter((self.p, self.d))
        
        # set objective
        
        # obj = (x_i^k^T P_i x_i^k +　Q_i^T x_i + ||x_i||_1) 
        #       + \mu_i^k x_i^T x_i - 2\mu_i^k aa_i^T x_i
        #       + < A_i^T\lambda_i^k, x_i >
        
        # f part 
        obj = 0.0
        obj += cp.sum_squares(self.param_Pi @ self.var_xi)
        obj += self.param_Qi.T @ self.var_xi 
        obj += cp.norm(self.var_xi, 1)
        # g part
        obj += self.param_mu[0] * cp.sum_squares(self.var_xi)
        obj += -2*self.param_mu_aai.T @ self.var_xi
        # h part
        obj += self.param_Ai_lam.T @ self.var_xi

        # set constraint
        cons = [
            cp.quad_form(self.var_xi-self.param_ai, np.identity(self.d)) \
                <= self.param_ci[0]
        ]
        
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        logging.info(f'self.prob {self.prob}')
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, i, y_mu_k, y_lam_k):
        '''
        i: agent number
        
        y_mu_k: (m, )
        y_lam_k: (p, )
        
        -> x_^{k+1}: (d, )
        '''
        
        '''
        NEED TO SET:
        self.param_Pi = cp.Parameter((self.d, self.d)) # P_i
        self.param_Qi = cp.Parameter(self.d) # Q_i
        self.param_mu = cp.Parameter(self.m) # \mu_i^k
        self.param_mu_aai = cp.Parameter(self.d) # \mu_i^k aa_i
        self.param_Ai_lam = cp.Parameter(self.d) # A_i^T \lambda_i^k 
        
        self.param_aai = cp.Parameter(self.d)
        self.param_cci = cp.Parameter(1)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        self.param_Ai = cp.Parameter((self.p, self.d))
        '''
        
        self.param_Pi.value = self.P[i]
        self.param_Qi.value = self.Q[i]
        self.param_mu.value = y_mu_k
        self.param_mu_aai.value = y_mu_k[0] * self.aa[i]
        self.param_Ai_lam.value = self.A_data[i].T @ y_lam_k
        
        
        self.param_aai.value = self.aa[i]
        self.param_cci.value = [self.cc[i]]
        self.param_ai.value = self.a[i]
        self.param_ci.value = [self.c[i]]
        self.param_Ai.value = self.A_data[i]
        
        if self.verbose:
            print()
            print(f'param_PxQ.value {self.param_PxQ.value}')
            print(f'param_xik.value {self.param_xik.value}')
            print(f'param_Ai.value {self.param_Ai.value}')
            print(f'param_Awuz.value {self.param_Awuz.value}')
            print(f'param_qGy.value {self.param_qGy.value}')
            print(f'param_qGyaa.value {self.param_qGyaa.value}')
            print(f'param_aai.value {self.param_aai.value}')
            print(f'param_ai.value {self.param_ai.value}')
            print(f'param_ci.value {self.param_ci.value}')
        
        # self.prob.solve(solver='MOSEK')
        self.prob.solve(solver='ECOS', reltol=1e-8)
        
        return self.var_xi.value
    
    def fi(self, i, xi): 
        ''' i: int, xi: (d, ) -> float'''
        return xi.T@self.P[i]@xi + self.Q[i].T@xi + np.linalg.norm(xi,1)
    
    def gi(self, i, xi):
        ''' i: int, xi: (d, ) -> (m, )'''
        res = np.zeros(self.m)
        res[0] = (xi-self.aa[i]).T @ (xi-self.aa[i]) - self.cc[i]
        return res
    
    def hi(self, i, xi):
        ''' i: int, xi: (d, ) -> (p, )'''
        res = self.A_data[i] @ xi
        return res
    
    def local_set_violation(self, i, xi):
        ''' xi: (d, ) -> float'''
        return max(0, (xi-self.a[i]).T@(xi-self.a[i]) - self.c[i])

    def compute_metrics(self, avg=False):
        ''' -> obj_err: float, cons_vio: float'''
        
        fun_val = 0.
        cons_ineq_val = np.zeros(self.m)  # inequality constraint values
        cons_eq_val = np.zeros(self.p)  # equality constraint values
        # constraint violation, including local set violation
        cons_vio = 0.
        
        x = self.x_cur
        if avg:
            x = self.x_avg 
        
        for i in range(self.N):            
            fun_val += self.fi(i, x[i])            
            cons_ineq_val += self.gi(i, x[i]) # m dimensional vector
            # print(self.A_data[i].shape, x[i].shape)
            cons_eq_val += self.A_data[i]@x[i]  # p dimensional vector                
            cons_vio += self.local_set_violation(i, x[i]) 
                        
        cons_vio += np.sum(np.max([cons_ineq_val, np.zeros(self.m)], axis=0))
        cons_vio += np.linalg.norm(cons_eq_val)
        obj_err = abs(fun_val - self.opt_val)
        # obj_err = fun_val - self.opt_val
        
        return obj_err, cons_vio

    def step(self):
        k = self.iter_num
        self.iter_num += 1  # equal to k+1 in this step
        stepsize = self.gamma/sqrt(k+1)
        
        # prepare z_nxt
        # \sum_j (W_{ij}(z_\mu^k)_j
        Wz_mu = np.zeros((self.N, self.m))
        # \sum_j (W_{ij}(z_\lam^k)_j
        Wz_lam  = np.zeros((self.N, self.p))
        for i in range(self.N):
            #   (m, )    =     (m, N)      @ (N, ) 
            Wz_mu[i] = self.z_mu_cur.T @ self.W[i]
            Wz_lam[i] = self.z_lam_cur.T @ self.W[i]
        
        for i in range(self.N):
            xik = self.x_cur[i] # current avg
            # update of x_nxt
            self.x_nxt[i] = self._solve_argmin_prob(i, 
                             self.y_mu_cur[i], self.y_lam_cur[i])
            self.x_avg[i] = (self.x_avg[i]*k 
                             + self.x_nxt[i]) / (k+1) # next avg
            self.z_mu_nxt[i] = Wz_mu[i] \
                + (k+1) * self.gi(i, self.x_avg[i]) \
                - k * self.gi(i, xik)
            self.z_lam_nxt[i] = Wz_lam[i] \
                + (k+1) * self.hi(i, self.x_avg[i],) \
                - k * self.hi(i, xik)
            
            # update of y_nxt (dual variable)
            # projection
            self.z_mu_nxt[i] = np.max([np.zeros(self.m), 
                                       self.z_mu_nxt[i]], axis=0) 
            self.y_mu_nxt[i] = (k*self.y_mu_cur[i] 
                                + stepsize*self.z_mu_nxt[i])/(k+1)
            self.y_lam_nxt[i] = (k*self.y_lam_cur[i] 
                                + stepsize*self.z_lam_nxt[i])/(k+1)

        self.x_cur = self.x_avg.copy()
        self.y_mu_cur = self.y_mu_nxt.copy()
        self.y_lam_cur = self.y_lam_nxt.copy()
        self.z_mu_cur = self.z_mu_nxt.copy()
        self.z_lam_cur = self.z_lam_nxt.copy()
        
        
        self.make_log()
        
        if self.verbose:
            self.show_status()
                    
            
    
    def show_status(self):
        # self.iter_num = 0
        # self.init_time = time()
        
        # # initial conditions
        # self.x_cur = np.zeros((self.N, self.d))
        # self.t_cur = np.zeros((self.N, self.p))
        # self.u_cur = np.zeros((self.N, (self.m+self.p)))
        # self.z_cur = np.zeros((self.N, (self.m+self.p)))
        # self.q_cur = np.zeros((self.N, self.p))
        logging.info(f'iteration {self.iter_num}')
        logging.info(f'objective error: {self.obj_err_log[-1]:.2e}, constraint violation: {self.cons_vio_log[-1]:.2e}')
        logging.info(f'x({self.iter_num}): {self.x_cur}')
        # logging.info(f't({self.iter_num}): {self.t_cur}')
        # logging.info(f'u({self.iter_num}): {self.u_cur}')
        # logging.info(f'z({self.iter_num}): {self.z_cur}')
        # logging.info(f'q({self.iter_num}): {self.q_cur}')
        logging.info(f'\n')
        