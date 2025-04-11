import numpy as np
import cvxpy as cp
from math import log
from time import time
import os

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class IPLUX:
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
        alpha
        rho
        W, H: (N, N) weight matrices
    '''
    
    name = "IPLUX"
    
    def __init__(
        self, 
        prob, 
        network, 
        alpha, 
        rho, 
        no_ineq=False, 
        verbose=0):
        '''
        these are set here:
        
        problem parameters:
        N: number of nodes
        d: dimension of xi's
        p: number of inequalities
        m: number of equalities, i.e., shape of A_i's is (m, d)
        x_star: N dimensional vector
        opt_val: float
        
        problem data:
        Q shape: (N, d)
        P shape: (N, d, d)
        A shape: (N, p, d)
        a shape: (N, d)
        c shape: (N, )
        aa shape: (N, d)
        cc shape: (N, )
        
        algorithm parameters:
        alpha
        rho
        W, H: (N, N) weight matrices
        
        
        these are set in 'reset()':
        iter_num: iteration k
        self.init_time = time()
        
        x_cur, x_nxt, x_avg: (N, d)
        t_cur, t_nxt: (N, m)
        u_cur, u_nxt: (N, m+p)
        z_cur, z_nxt: (N, m+p)
        q_cur, q_nxt: (N, m)
        
        no_ineq: if True, then there is no inequality
        verbose: displays information
        log_dir
        file_prefix
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
        self.A_data = prob.A
        # print(prob.A.shape)
        self.a = prob.a 
        self.c = prob.c 
        self.aa = prob.aa 
        self.cc = prob.cc 
        
        self.alpha = alpha
        self.rho = rho
        self.W = (np.identity(prob.N) + network) * 0.5
        self.H = (np.identity(prob.N) - network) * 0.5
        # self.W = network
        # self.H = np.identity(prob.N) - network
        
        self.no_ineq = no_ineq
        self.verbose = verbose
        self.log_dir = f'log/N{self.N}'
        self.file_prefix = f'{self.name}_a{self.alpha}_r{self.rho}'
        
        self._set_argmin_prob(no_ineq=no_ineq)
        self.reset()

    def reset(self):
        '''
        iter_num = 0
        self.init_time = time()
        
        decision vars:
        x_cur, x_nxt, x_avg: (N, d)
        t_cur, t_nxt: (N, m)
        u_cur, u_nxt: (N, m+p)
        z_cur, z_nxt: (N, m+p)
        q_cur, q_nxt: (N, m)
        
        log lists:
        obj_err_log = []
        obj_err_avg_log = []
        cons_vio_log = []
        cons_vio_avg_log = []
        x_dis_log = []
        x_dis_avg_log = []
        '''
        
        print('reset')
        
        self.iter_num = 0
        self.init_time = time()
        
        # initial conditions
        self.x_cur = np.zeros((self.N, self.d))
        # self.x_cur = self.x_star.copy()
        self.t_cur = np.zeros((self.N, self.m))
        self.u_cur = np.zeros((self.N, (self.m+self.p)))
        self.z_cur = np.zeros((self.N, (self.m+self.p)))
        self.q_cur = np.zeros((self.N, self.m))
        
        self.x_avg = self.x_cur.copy() # for running average
        
        self.x_nxt = np.zeros((self.N, self.d))
        self.t_nxt = np.zeros((self.N, self.m))
        self.u_nxt = np.zeros((self.N, (self.m+self.p)))
        self.z_nxt = np.zeros((self.N, (self.m+self.p)))
        self.q_nxt = np.zeros((self.N, self.m))
        
        for i in range(self.N):
            Gi_yi0 = self.gi(i, self.x_cur[i]) - self.t_cur[i]
            self.q_cur[i] = np.max([np.zeros(self.m), -Gi_yi0], axis=0)
        
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
            
            logging.info(f'{self.name} alpha {self.alpha} rho {self.rho}, iter {self.iter_num}, obj err: {self.obj_err_log[-1]:.2e}, cons vio: {self.cons_vio_log[-1]:.2e}')
            
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

    def _set_argmin_prob(self, no_ineq=False):
        '''
        var_xi: (d, )

        self.param_PxQ = cp.Parameter(self.d) # P_i^T x_i^k + Q_i
        self.param_xik = cp.Parameter(self.d) # x_i^k
        self.param_Ai = cp.Parameter((self.m, self.d))
        # A_i^T (\sum_j w_{ij}(u_x^k)_j - 1/\rho(z_x^k)_i)
        self.param_Awuz = cp.Parameter(self.d) 
        self.param_qGy = cp.Parameter(self.p, nonneg=True) # (q_i^k + G_i(y_i^k))
        self.param_qGyaa = cp.Parameter(self.d) # (q_i^k + G_i(y_i^k)) * a_i'
        self.param_aai = cp.Parameter(self.d)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        '''
        
        self.var_xi = cp.Variable(self.d)

        self.param_PxQ = cp.Parameter(self.d) # 2*P_i^T x_i^k + Q_i
        self.param_xik = cp.Parameter(self.d) # x_i^k
        self.param_Ai = cp.Parameter((self.p, self.d))
        # self.param_ATA = cp.Parameter((self.d, self.d), PSD=True) # A_i^T A_i
        
        # A_i^T (\sum_j w_{ij}(u_x^k)_j - 1/\rho(z_x^k)_i)
        self.param_Awuz = cp.Parameter(self.d) 
        # q_i^k + G_i(y_i^k)
        self.param_qGy = cp.Parameter(self.m, nonneg=True) 
        # (q_i^k + G_i(y_i^k)) * a_i'
        self.param_qGyaa = cp.Parameter(self.d) 
        self.param_aai = cp.Parameter(self.d)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        
        # obj = <2*P_i^T x_i^k +　Q_i, x_i> + ||x_i||_1 
        #       + \alpha/2 * ||x_i-x_i^k||^2
        #       + 1/(2\rho) * ||A_i x_i||^2 
        #       + <\sum_j w_{ij}(u_x^k)_j - 1/\rho(z_x^k)_i, A_ix_i>
        #       + <q_i^k + g_i(x_i^k) - t_i^k, g_i(x_i)>
        
        # quad_form(x, para) is not DPP
        # obj += (0.5 / self.rho) * \
        #         cp.quad_form(self.var_xi, self.param_ATA)
        # sum_squares(para) is OK
        
        # obj += self.param_qGy[0] * \
        #         cp.quad_form(self.var_xi-self.param_aai, np.identity(self.d))
        # above is not DPP, since DPP forbits para * para
        
        
        obj = 0.0
        obj += self.param_PxQ.T @ self.var_xi
        obj += cp.norm(self.var_xi, 1)
        obj += self.alpha * 0.5 * \
                cp.quad_form(self.var_xi-self.param_xik, np.identity(self.d))
        obj += (0.5 / self.rho) * cp.sum_squares(self.param_Ai @ self.var_xi)
        obj += self.param_Awuz.T @ self.var_xi
        
        obj += self.param_qGy[0] * cp.quad_form(self.var_xi, np.identity(self.d))
        obj += -2 * self.param_qGyaa.T @ self.var_xi

        cons = [
            cp.quad_form(self.var_xi-self.param_ai, np.identity(self.d)) \
                <= self.param_ci[0]
        ]
        
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        logging.info(f'self.prob {self.prob}')
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, i, xik, tik, wuik, zik, qik):
        '''
        i: agent number
        xik: x_i^k (d, )
        tik: t_i^k (m, )
        wuik: \sum_j w_{ij}(u_x^k)_j (m+p, )
        zik: z_i^k (m+p, )
        
        -> x_^{k+1}: (d, )
        '''
        
        # NEED TO SET:
        # self.param_PxQ = cp.Parameter(self.d) # 2*P_i^T x_i^k + Q_i
        # self.param_xik = cp.Parameter(self.d) # x_i^k
        # self.param_Ai = cp.Parameter((self.m, self.d))
        # # A_i^T (\sum_j w_{ij}(u_x^k)_j - 1/\rho(z_x^k)_i)
        # self.param_Awuz = cp.Parameter(self.d) 
        # self.param_qGy = cp.Parameter(self.p, nonneg=True) # (q_i^k + G_i(y_i^k))
        # self.param_qGyaa = cp.Parameter(self.d) # (q_i^k + G_i(y_i^k)) * a_i'
        # self.param_aai = cp.Parameter(self.d)
        # self.param_ai = cp.Parameter(self.d)
        # self.param_ci = cp.Parameter(1)
        
        p = self.p
        self.param_PxQ.value = 2*self.P[i].T@xik + self.Q[i]
        self.param_xik.value = xik
        self.param_Ai.value = self.A_data[i]
        self.param_Awuz.value = self.A_data[i].T @ (wuik[:p] - zik[:p]/self.rho)
        qGy = qik + self.gi(i, xik) - tik
        self.param_qGy.value = qGy # shape (p, ) = (1, )
        self.param_qGyaa.value = qGy[0] * self.aa[i] 
        self.param_aai.value = self.aa[i]
        self.param_ai.value = self.a[i]
        self.param_ci.value = [self.c[i]]
        
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
        # self.prob.solve(solver='CPLEX', cplex_params={'simplex.tolerances.optimality':1e-7})
        self.prob.solve(solver='ECOS', reltol=1e-11)
        # self.prob.solve(solver='CVXOPT', reltol=1e-6)
        
        return self.var_xi.value
    
    def fi(self, i, xi): 
        ''' i: int, xi: (d, ) -> float'''
        return xi.T@self.P[i]@xi + self.Q[i].T@xi + np.linalg.norm(xi,1)
    
    def gi(self, i, xi):
        ''' i: int, xi: (d, ) -> (m, )'''
        res = np.zeros(self.m)
        res[0] = (xi-self.aa[i]).T @ (xi-self.aa[i]) - self.cc[i]
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
            cons_ineq_val += self.gi(i, x[i]) # p dimensional vector
            cons_eq_val += self.A_data[i]@x[i]  # m dimensional vector                
            cons_vio += self.local_set_violation(i, x[i]) 
                        
        cons_vio += np.sum(np.max([cons_ineq_val, np.zeros(self.m)], axis=0))
        cons_vio += np.linalg.norm(cons_eq_val)
        obj_err = abs(fun_val - self.opt_val)
        # obj_err = fun_val - self.opt_val
        
        return obj_err, cons_vio

    def step(self):
        self.iter_num += 1
        
        # weighted average of u by W
        u_wavg = np.zeros((self.N, (self.m+self.p))) 
        for i in range(self.N):
            u_wavg[i] = self.u_cur.T @ self.W[i]
            #   (6,)  =     (6,3)   @ (3,) 
        # print(f'u_wavg {u_wavg}')
        
        for i in range(self.N):
            xik = self.x_cur[i]
            tik = self.t_cur[i]
            zik = self.z_cur[i]
            qik = self.q_cur[i]
            Gi_yik = self.gi(i, xik) - tik
            # print(f'Gi_yik {Gi_yik}')
            p = self.p
            
            # updates
            self.x_nxt[i] = self._solve_argmin_prob(i, xik, 
                                                    tik, u_wavg[i], zik, qik)
            self.t_nxt[i] = (
                self.alpha * tik 
                + zik[p:] / self.rho - u_wavg[i,p:]
                + qik + Gi_yik 
                ) / (self.alpha + 1 / self.rho)
            
            self.u_nxt[i] = u_wavg[i] + (
                np.r_[self.A_data[i]@self.x_nxt[i], self.t_nxt[i]] - zik) / self.rho
            # z_nxt needs all u_nxt[j]
            Gi_yi_nxt = self.gi(i, self.x_nxt[i]) - self.t_nxt[i]
            self.q_nxt[i] = max(-Gi_yi_nxt, qik + Gi_yi_nxt)
            
        # z_nxt needs all u_nxt[j]
        u_havg = np.zeros((self.N, (self.m+self.p))) 
        for i in range(self.N):
            u_havg[i] = self.u_nxt.T @ self.H[i]
        self.z_nxt = self.z_cur + self.rho * u_havg
        
        # logging.info(f'dx {self.x_nxt-self.x_cur}, dt {self.t_nxt-self.t_cur}, du {self.u_nxt-self.u_cur}, dz {self.z_nxt-self.z_cur}, dq {self.q_nxt-self.q_cur}')
        
        self.x_cur = self.x_nxt.copy()
        self.t_cur = self.t_nxt.copy()
        self.u_cur = self.u_nxt.copy()
        self.z_cur = self.z_nxt.copy()
        self.q_cur = self.q_nxt.copy()
        
        self.x_avg = (self.x_avg * self.iter_num + self.x_cur) / (self.iter_num + 1)
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
        logging.info(f't({self.iter_num}): {self.t_cur}')
        logging.info(f'u({self.iter_num}): {self.u_cur}')
        logging.info(f'z({self.iter_num}): {self.z_cur}')
        logging.info(f'q({self.iter_num}): {self.q_cur}')
        logging.info(f'\n')
        
    
    