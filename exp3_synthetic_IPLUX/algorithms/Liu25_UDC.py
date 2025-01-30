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

class UDC:
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
    
    name = "UDC"
    
    def __init__(
        self, 
        prob, 
        network, 
        rho,
        alpha=0,
        param_setting='proximal_tracking',
        prox=False, 
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
                rho
                alpha (positve for UDC_prox, 0 for UDC)
                A_weight, H1, H2, H2_half, D: (N, N) weight matrices. 
                    A: P_A
                    A_half: {P_A}^{1/2}
                    H1: P_H
                    H2: P_{\tilde{H}}
                    H2_half: P_{\tilde{H}}^{1/2}
                    D: P_D
                    
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
        
        self.alpha = alpha
        self.rho = rho

        self.A_weight, self.A_half, self.H1, self.H2, self.H2_half, self.D \
                = self.set_weight_param(param_setting, network, rho)
        # self.W = (np.identity(prob.N) + network) * 0.5
        # self.H = (np.identity(prob.N) - network) * 0.5
        # self.W = network
        # self.H = np.identity(prob.N) - network
        
        self.verbose = verbose
        self.log_dir = f'log/N{self.N}'
        
        self.file_prefix = f'{self.name}_a{self.alpha}_r{self.rho}'
        
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

    def _set_argmin_prob(self):
        '''
        self.var_xi = cp.Variable(self.d)

        # (P_i * (P_D)_i)^{1/2}
        self.param_DiPi_sqrt = cp.Parameter((self.d, self.d)) 
        self.param_DiQi = cp.Parameter(self.d) # Q_i * (P_D)_i
        self.param_Di = cp.Parameter(1, nonneg=True) # (P_D)_i
        self.param_Di_sqrt = cp.Parameter(1, nonneg=True) # ((P_D)_i)^{1/2}
        self.param_xik_sqrt_Di = cp.Parameter(self.d) # x_i^k * ((P_D)_i)^{1/2}
        
        # \sum_j A_{ij}(y_\mu^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\mu^k)_j
        self.param_AyHz_mu = cp.Parameter(self.m) 
        # \sum_j A_{ij}(y_\lam^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\lam^k)_j
        self.param_AyHz_lam = cp.Parameter(self.p)
        
        self.param_aai = cp.Parameter(self.d)
        self.param_cci = cp.Parameter(1)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        self.param_Ai = cp.Parameter((self.p, self.d))
        '''
        
        self.var_xi = cp.Variable(self.d)

        # (P_D)_i^{1/2} * (P_i)^{1/2} 
        self.param_DiPi_sqrt = cp.Parameter((self.d, self.d)) 
        self.param_DiQi = cp.Parameter(self.d) # Q_i * (P_D)_i
        self.param_Di = cp.Parameter(1, nonneg=True) # (P_D)_i
        self.param_Di_sqrt = cp.Parameter(1, nonneg=True) # ((P_D)_i)^{1/2}
        self.param_xik_sqrt_Di = cp.Parameter(self.d) # x_i^k * ((P_D)_i)^{1/2}
        
        # \sum_j A_{ij}(y_\mu^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\mu^k)_j
        self.param_AyHz_mu = cp.Parameter(self.m) 
        # \sum_j A_{ij}(y_\lam^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\lam^k)_j
        self.param_AyHz_lam = cp.Parameter(self.p)
        
        self.param_aai = cp.Parameter(self.d)
        self.param_cci = cp.Parameter(1)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        self.param_Ai = cp.Parameter((self.p, self.d))
        
        # set objective
        
        # obj = (x_i^k^T P_i x_i^k +　Q_i^T x_i + ||x_i||_1) * (P_D)_i 
        #       + 0.5 * \alpha * ||x_i-x_i^k||^2 * (P_D)_i
        #       + ||[ \sum_j A_{ij}(y_\mu^k)_j 
        #            -\sum_j (\tilde{H}^{1/2})_{ij}(z_\mu^k)_j 
        #            +||x_i - aa_i||^2-cc_i ]_+||^2/(2)
        #       + ||  \sum_j A_{ij}(y_\lam^k)_j 
        #            -\sum_j (\tilde{H}^{1/2})_{ij}(z_\lam^k)_j 
        #            +A_i x_i ||^2/(2)
        
        # f part and proximal term
        obj = 0.0
        obj += cp.sum_squares(self.param_DiPi_sqrt @ self.var_xi)
        obj += self.param_DiQi.T @ self.var_xi 
        obj += self.param_Di[0] * cp.norm(self.var_xi, 1)
        obj += self.alpha * 0.5 * cp.sum_squares(
            self.param_Di_sqrt[0] * self.var_xi - self.param_xik_sqrt_Di)
                        
        # g part
        gixi = cp.sum_squares(self.var_xi-self.param_aai) - self.param_cci
        obj += 0.5*cp.sum_squares(cp.pos(self.param_AyHz_mu + gixi))
        
        # h part
        obj += 0.5*cp.sum_squares(self.param_AyHz_lam 
                                  + self.param_Ai@self.var_xi)
        

        # set constraint
        cons = [
            cp.quad_form(self.var_xi-self.param_ai, np.identity(self.d)) \
                <= self.param_ci[0]
        ]
        
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        logging.info(f'self.prob {self.prob}')
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, i, xik, AyHz_mu_k, AyHz_lam_k):
        '''
        i: agent number
        
        xik: (d, )
        
        AyHz_mu_k: (m, ), 
        \sum_j A_{ij}(y_\mu^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\mu^k)_j
        
        AyHz_lam_k: (p, ),
        \sum_j A_{ij}(y_\lam^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\lam^k)_j
        
        -> x_^{k+1}: (d, )
        '''
        
        '''
        NEED TO SET:
        
        self.param_Di = cp.Parameter(1, nonneg=True) # (P_D)_i
        self.param_Di_sqrt = cp.Parameter(1, nonneg=True) # ((P_D)_i)^{1/2}
        # (P_i * (P_D)_i)^{1/2}
        self.param_DiPi_sqrt = cp.Parameter((self.d, self.d)) 
        self.param_DiQi = cp.Parameter(self.d) # Q_i * (P_D)_i
        self.param_xik_sqrt_Di = cp.Parameter(self.d) # x_i^k * ((P_D)_i)^{1/2}
        
        # \sum_j A_{ij}(y_\mu^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\mu^k)_j
        self.param_AyHz_mu = cp.Parameter(self.m) 
        # \sum_j A_{ij}(y_\lam^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\lam^k)_j
        self.param_AyHz_lam = cp.Parameter(self.p)
        
        self.param_aai = cp.Parameter(self.d)
        self.param_cci = cp.Parameter(1)
        self.param_ai = cp.Parameter(self.d)
        self.param_ci = cp.Parameter(1)
        self.param_Ai = cp.Parameter((self.p, self.d))
        '''
        
        Di = self.D[i, i]
        self.param_Di.value = [Di]
        self.param_Di_sqrt.value = [sqrt(Di)]
        self.param_DiPi_sqrt.value = sqrt(Di) * self.P_sqrt[i]
        self.param_DiQi.value = Di * self.Q[i]
        self.param_xik_sqrt_Di.value = xik * sqrt(Di)
        self.param_AyHz_mu.value = AyHz_mu_k
        self.param_AyHz_lam.value = AyHz_lam_k
        
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
        
        self.prob.solve(solver='MOSEK')
        # self.prob.solve()
        
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
        self.iter_num += 1  # equal to k+1 in this step
        
        # weighted averages
        
        # AyHz_mu_k: (m, ), 
        # \sum_j A_{ij}(y_\mu^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\mu^k)_j
        
        # AyHz_lam_k: (p, ),
        # \sum_j A_{ij}(y_\lam^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\lam^k)_j
        
        AyHz_mu_k = np.zeros((self.N, self.m))
        AyHz_lam_k = np.zeros((self.N, self.p))
        for i in range(self.N):
            #   (m, )    =     (m, N)      @ (N, ) 
            AyHz_mu_k[i] = self.y_mu_cur.T @ self.A_weight[i] \
                - self.z_mu_cur.T @ self.H2_half[i]
            AyHz_lam_k[i] = self.y_lam_cur.T @ self.A_weight[i] \
                - self.z_lam_cur.T @ self.H2_half[i]
        # print(f'u_wavg {u_wavg}')
        
        for i in range(self.N):
            xik = self.x_cur[i]
            Di = self.D[i, i]
            
            # update of x_nxt
            self.x_nxt[i] = self._solve_argmin_prob(i, xik, 
                                    AyHz_mu_k[i], AyHz_lam_k[i])
            
            # update of y_nxt
            self.y_mu_nxt[i] = np.max(
                np.c_[AyHz_mu_k[i] + self.gi(i, self.x_nxt[i]), 
                      np.zeros(self.m)], 
                axis=1) / Di
            self.y_lam_nxt[i] = (AyHz_lam_k[i] + self.hi(i, self.x_nxt[i])) / Di

        # update of z_nxt needs all y_nxt[j]
        # \sum_j (\tilde{H}^{1/2})_{ij}(y_\mu^{k+1})_j
        Hy_mu_nxt = np.zeros((self.N, self.m))
        # \sum_j (\tilde{H}^{1/2})_{ij}(y_\lam^{k+1})_j
        Hy_lam_nxt = np.zeros((self.N, self.p))
        for i in range(self.N):
            #   (m, )    =     (m, N)      @ (N, ) 
            Hy_mu_nxt[i] = self.y_mu_nxt.T @ self.H2_half[i]
            Hy_lam_nxt[i] = self.y_lam_nxt.T @ self.H2_half[i]
        self.z_mu_nxt = self.z_mu_cur + self.rho * Hy_mu_nxt
        self.z_lam_nxt = self.z_lam_cur + self.rho * Hy_lam_nxt
        
        # logging.info(f'dx {self.x_nxt-self.x_cur}, dt {self.t_nxt-self.t_cur}, du {self.u_nxt-self.u_cur}, dz {self.z_nxt-self.z_cur}, dq {self.q_nxt-self.q_cur}')
        
        self.x_cur = self.x_nxt.copy()
        self.y_mu_cur = self.y_mu_nxt.copy()
        self.y_lam_cur = self.y_lam_nxt.copy()
        self.z_mu_cur = self.z_mu_nxt.copy()
        self.z_lam_cur = self.z_lam_nxt.copy()
        
        self.x_avg = (self.x_avg*self.iter_num + self.x_cur) / (self.iter_num+1)
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
        
    
    def set_weight_param(self, param_setting: str, W, rho):
        '''
        W: (N, N) doubly stochastic
        -> A, A_half, H1, H2, H2_half, D
        '''
        print(f'UDC setting: {param_setting}')
        
        N = W.shape[0]
        I = np.identity(N)
        
        if param_setting == 'proximal_tracking':
            # A = \rho W^2
            # H = I - W^2
            # \tilde{H} = (I-W)^2
            # D = \rho I
            self.name += '_pt'
            A_half = sqrt(rho) * W
            A = A_half @ A_half
            D = rho * I
            H1 = I - W @ W
            H2_half = I - W
            H2 = H2_half @ H2_half
            
        elif param_setting == 'PEXTRA':
            # A = \rho/2 * (I+W)
            # H = 1/2 * (I-W)
            # \tilde{H} = 1/2 * (I-W)
            # D = \rho I
            self.name += '_PEXTRA'    
            A = 0.5 * rho * (I+W)
            A_half = scipy.linalg.sqrtm(A)
            D = rho * I
            H1 = 0.5 * (I-W)
            H2 = 0.5 * (I-W)
            H2_half = scipy.linalg.sqrtm(H2)
        
        
        return A, A_half, H1, H2, H2_half, D
    