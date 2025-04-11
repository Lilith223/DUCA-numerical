import numpy as np
import cvxpy as cp
from math import log, sqrt
from time import time
import os
import scipy.linalg
from graph_utils import get_max_degree_weight, get_metropolis_weight, get_lazy_metropolis_weight

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class UDC:
    '''
    problem data:
        c: (N,)
        d: (N,) 
        b: (1,) 
        
    algorithm parameters:
        alpha (for UDC_prox)
        A, A_half, H1, H2, H2_half, D: (N, N) weight matrices. 
            H1 for H, H2 for H tilde
    '''
    
    name = "UDC"
    
    def __init__(
        self, 
        prob, 
        log_dir,
        adj_mat, 
        rho,
        alpha=0,
        param_setting='pt',
        theta=-1,
        verbose=0):
        '''
        these are set here:
            problem parameters:
                N: number of nodes
                m=1: number of inequalities
                x_star: N dimensional vector
                opt_val: float
            
            problem data:
                c: (N,)
                d: (N,) 
                b: (1,) 
            
            algorithm parameters:
                rho
                alpha (positve for UDC_prox, 0 for UDC)
                theta (only for DPMM)
                A_weight, H1, H2, H2_half, D, D_inv: (N, N) weight matrices. 
                    A_weight: P_A
                    A_half: {P_A}^{1/2}
                    H1: P_H
                    H2: P_{\tilde{H}}
                    H2_half: P_{\tilde{H}}^{1/2}
                    D: P_D
                    D_inv: (P_D)^{-1}
                    
            verbose: displays information
            log_dir
            file_prefix
        
        
        these are set in 'reset()':
            iter_num: iteration number
            self.init_time = time()
            
            x_cur, x_nxt, x_avg: (N, d)
            y_mu_cur, y_mu_nxt: (N, m)
            z_mu_cur, z_mu_nxt: (N, m)

            # y_lam_cur, y_lam_nxt: (N, p=0)
            # z_lam_cur, z_lam_nxt: (N, p=0)
            
            init_time
        '''

        self.N = prob.N
        self.m = 1
        self.x_star = prob.x_star
        self.opt_val = prob.opt_val

        # problem data        
        self.c = prob.c     # (N,) 
        self.d = prob.d     # (N,)
        self.b = prob.b     # (1,)
        
        self.alpha = alpha
        self.rho = rho
        self.theta = theta

        # 'adj_mat' is an 0/1 adjacent matrix
        self.A_weight, self.A_half, self.H1, self.H2, self.H2_half, self.D \
                = self.set_weight_param(param_setting, adj_mat, rho)
        self.D_inv = np.diag(1/np.diag(self.D))
        
        self.verbose = verbose
        self.log_dir = log_dir
        
        if self.theta != -1: # for DPMM
            self.file_prefix = f'{self.name.replace("-", "_")}_a{self.alpha}_r{self.rho}_t{self.theta}'
        else:
            self.file_prefix = f'{self.name.replace("-", "_")}_a{self.alpha}_r{self.rho}'
        
        self._set_argmin_prob()
        self.reset()

    def reset(self):
        '''
        iter_num = 0
        self.init_time = time()
        
        decision vars:
        x_cur, x_nxt, x_avg: (N,)
        y_mu_cur, y_mu_nxt: (N,)
        z_mu_cur, z_mu_nxt: (N,)
        
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
        self.x_cur = np.zeros(self.N)
        # self.x_cur = self.x_star.copy()
        self.y_mu_cur = np.zeros(self.N)
        self.z_mu_cur = np.zeros(self.N)
        
        self.x_avg = self.x_cur.copy() # for running average
        
        self.x_nxt = np.zeros(self.N)
        self.y_mu_nxt = np.zeros(self.N)
        self.z_mu_nxt = np.zeros(self.N)

        
        
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
            
            if self.theta != -1: # only for DPMM, show theta
                logging.info(f'{self.name} alpha {self.alpha} rho {self.rho} theta {self.theta}, iter {self.iter_num}, obj err: {self.obj_err_log[-1]:.2e}, cons vio: {self.cons_vio_log[-1]:.2e}')
            else:
                logging.info(f'{self.name} alpha {self.alpha} rho {self.rho}, iter {self.iter_num}, obj err: {self.obj_err_log[-1]:.2e}, cons vio: {self.cons_vio_log[-1]:.2e}')
            
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            
            # last iterate
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_oe.txt', self.obj_err_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_cv.txt', self.cons_vio_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_xd.txt', self.x_dis_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_oe_cv.txt', self.obj_err_log + self.cons_vio_log, delimiter=',')
            # running average
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_oe_avg.txt', self.obj_err_avg_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_cv_avg.txt', self.cons_vio_avg_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_xd_avg.txt', self.x_dis_avg_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/{self.file_prefix}_oe_cv_avg.txt', self.obj_err_avg_log + self.cons_vio_avg_log, delimiter=',')
            
            logging.info(f"time {time()-self.init_time:.2f}, saved\n")

    def _set_argmin_prob(self):
        '''
        self.var_x = cp.Variable(self.N)

        self.param_Dxk = cp.Parameter(self.N) # D x^k
        # Ay^k - \tilde{H}^{1/2}z^k + b/N 1_N 
        self.param_AyHzb = cp.Parameter(self.N) 
        '''
        
        self.var_x = cp.Variable(self.N)

        self.param_Dxk = cp.Parameter(self.N) # D x^k
        # Ay^k - \tilde{H}^{1/2}z^k + b/N 1_N 
        self.param_AyHzb = cp.Parameter(self.N) 

        
        # set objective
        
        # obj = (Dc)^T x 
        #       + 0.5 * \alpha * || x^T D x - 2 * ((x^k)^TD) x ||^2
        #       + 0.5 * ||[ Ay^k - \tilde{H}^{1/2}z^k + b/N * 1_N
        #            - d \dot \log(1+x) ]_+||^2

        
        # f part and proximal term
        obj = 0.0
        obj += self.c.T @ self.D @ self.var_x
        obj += self.alpha * 0.5 * (cp.quad_form(self.var_x, self.D) 
                                   - 2 * self.param_Dxk.T @ self.var_x)
        
        print(f'd: {self.d.shape}')        
        # g part
        gx = cp.multiply(self.d, cp.log(self.var_x+1))
        obj += 0.5 * cp.sum_squares(cp.pos(self.param_AyHzb - gx))
        
        # set constraint
        cons = [np.zeros(self.N) <= self.var_x, 
                self.var_x <= np.ones(self.N)] 
        
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        # logging.info(f'self.prob {self.prob}')
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, xk, AyHz_mu_k):
        '''
        i: agent number
        
        xk: (N, )
        
        AyHz_mu_k: (N, ), 
        Ay_\mu^k - \tilde{H}^{1/2}z_\mu^k
        
        -> x^{k+1}: (N, )
        '''
        
        '''
        NEED TO SET:

        # D x^k
        self.param_Dxk = cp.Parameter(self.N) 
        
        # Ay^k - \tilde{H}^{1/2}z^k + b/N 1_N 
        self.param_AyHzb = cp.Parameter(self.N) 
        '''
        
        self.param_Dxk.value = self.D @ xk
        self.param_AyHzb.value = AyHz_mu_k + self.b[0]*np.ones(self.N)/self.N
        
        if self.verbose:
            print()
            print(f'param_Dxk.value {self.param_Dxk.value}')
            print(f'param_AyHzb.value {self.param_AyHzb.value}')
            print()
        
        self.prob.solve(solver='MOSEK')
        # self.prob.solve(solver='CPLEX', cplex_params={'parameters.simplex.tolerances.optimality':1e-7})
        # self.prob.solve(solver='ECOS', reltol=1e-8)
        # self.prob.solve(solver='CVXOPT', reltol=1e-6)
        
        return self.var_x.value

    def compute_metrics(self, avg=False):
        ''' -> obj_err: float, cons_vio: float'''
        
        x = self.x_cur
        if avg:
            x = self.x_avg 
        
        # objective function value
        fun_val = self.c.T @ x
        # inequality constraint value
        cons_ineq_val = max(self.b[0] - self.d.T@np.log(x+1), 0)
        # local set constraint violations, shape: (N,)
        cons_loc_vio = np.max([-x, np.zeros(self.N)], axis=0) \
            + np.max([x-1, np.zeros(self.N)], axis=0) 
        
        obj_err = abs(fun_val - self.opt_val)
        cons_vio = np.sum(cons_loc_vio) + cons_ineq_val
        # print(f'obj err {fun_val-self.opt_val}, cons vio: {np.sum(cons_loc_vio)+cons_ineq_val}')
        return obj_err, cons_vio

    def step(self):
        self.iter_num += 1  # equal to k+1 in this step
        
        # weighted averages
        
        # AyHz_mu_k: (m, ), 
        # \sum_j A_{ij}(y_\mu^k)_j - \sum_j (\tilde{H}^{1/2})_{ij}(z_\mu^k)_j
        
        AyHz_mu_k = self.A_weight @ self.y_mu_cur \
                - self.H2_half @ self.z_mu_cur

        # update of x_nxt
        self.x_nxt = self._solve_argmin_prob(self.x_cur, AyHz_mu_k)

        # update of y_nxt
        self.y_mu_nxt = self.D_inv @ np.max(
            [AyHz_mu_k + self.b[0]*np.ones(self.N)/self.N  
                            - np.multiply(self.d, np.log(1+self.x_nxt)), 
             np.zeros(self.N)], axis=0
        )
        # print(f'y_nxt {self.y_mu_nxt.shape}')
        
        # for i in range(self.N):
        #     xik = self.x_cur[i]
        #     Di = self.D[i, i]
            
        #     # update of x_nxt
        #     self.x_nxt[i] = self._solve_argmin_prob(i, xik, 
        #                             AyHz_mu_k[i], AyHz_lam_k[i])
            
        #     # update of y_nxt
        #     self.y_mu_nxt[i] = np.max(
        #         np.c_[AyHz_mu_k[i] + self.gi(i, self.x_nxt[i]), 
        #               np.zeros(self.m)], 
        #         axis=1) / Di
        #     self.y_lam_nxt[i] = (AyHz_lam_k[i] + self.hi(i, self.x_nxt[i]))/Di

        # update of z_nxt needs all y_nxt[j]
        # \sum_j (\tilde{H}^{1/2})_{ij}(y_\mu^{k+1})_j
        # Hy_mu_nxt = np.zeros((self.N, self.m))
        # \sum_j (\tilde{H}^{1/2})_{ij}(y_\lam^{k+1})_j
        # Hy_lam_nxt = np.zeros((self.N, self.p))
        # for i in range(self.N):
        #     #   (m, )    =     (m, N)      @ (N, ) 
        #     Hy_mu_nxt[i] = self.y_mu_nxt.T @ self.H2_half[i]
        #     Hy_lam_nxt[i] = self.y_lam_nxt.T @ self.H2_half[i]
        
        if self.rho_only_in_mat: # these methods use rho=1
            self.z_mu_nxt = self.z_mu_cur + self.H2_half @ self.y_mu_nxt
        else:
            self.z_mu_nxt = self.z_mu_cur \
                            + self.rho * self.H2_half @ self.y_mu_nxt
        
        # logging.info(f'iter {self.iter_num}, x_nxt {self.x_nxt}, y {self.y_mu_nxt}, {self.y_lam_nxt}, z {self.z_mu_nxt, self.z_lam_nxt}')
        
        if self.theta != -1: # only for DPMM
            self.x_cur = (1-self.theta)*self.x_cur + self.theta*self.x_nxt
        else:
            self.x_cur = self.x_nxt.copy()
        self.y_mu_cur = self.y_mu_nxt.copy()
        self.z_mu_cur = self.z_mu_nxt.copy()
        
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
        
    
    def set_weight_param(self, param_setting: str, adj_mat, rho):
        '''
        adj_mat: (N, N) adjcent matrix
        -> A, A_half, H1, H2, H2_half, D
        '''
        print(f'UDC setting: {param_setting}')
        
        N = adj_mat.shape[0]
        I = np.identity(N)
        edge_num = 0
        for i in range(N):
            for j in range(N):
                if i!=j and adj_mat[i,j]:
                    edge_num += 1
        edge_num >>= 1
        degrees = np.sum(adj_mat, axis=1)
        
        L_mh = I - get_metropolis_weight(adj_mat)
        L_lm = I - get_lazy_metropolis_weight(adj_mat)
        L_const = np.diag(degrees) - adj_mat
        
        
        if param_setting == 'New1':
            # A = \rho/2 * (2I - L_mh)
            # H = \tilde{H} = 1/2 * L_mh
            # D = \rho * Lam_mh
            
            self.rho_only_in_mat = False
            self.name += '_New1'    
            Lam_mh = np.diag(np.diag(L_mh))
            A = rho * (Lam_mh - 0.5 * L_mh)
            D = rho * Lam_mh
            H1 = 0.5 * L_mh
            H2 = 0.5 * L_mh
            
            A_half = np.real(scipy.linalg.sqrtm(A))
            H2_half = np.real(scipy.linalg.sqrtm(H2))
        
        elif param_setting == 'PEXTRA':
            # A = \rho/2 * (2I - L_mh)
            # H = \tilde{H} = 1/2 * L_mh
            # D = \rho * I
            
            self.rho_only_in_mat = False
            self.name += '_PEXTRA'    
            A = 0.5 * rho * (2*I - L_mh)
            D = rho * I
            H1 = 0.5 * L_mh
            H2 = 0.5 * L_mh
            
            A_half = np.real(scipy.linalg.sqrtm(A))
            H2_half = np.real(scipy.linalg.sqrtm(H2))
        
        elif param_setting == 'PGC':
            # A = \rho (\Lambda_const - 1/2 * L_const)
            # H = \tilde{H} = 1/2 * L_const
            # D = \rho\Lambda_const
            
            self.rho_only_in_mat = False
            self.name += '_PGC'

            Lam_const = np.diag(degrees)
            A = rho * (Lam_const - 0.5 * L_const)
            H1 = 0.5 * L_const
            H2 = 0.5 * L_const
            D = rho * Lam_const
            
            A_half = np.real(scipy.linalg.sqrtm(A))
            H2_half = np.real(scipy.linalg.sqrtm(H2))
            
        elif param_setting == 'DPGA':
            # A = \diag(\gamma|N_i|) - L_1
            # H = \tilde{H} = \gamma * L_const
            # D = \gamma \diag(|N_i|)
            
            self.rho_only_in_mat = True
            self.name += '_DPGA'
            gamma = sqrt(N*rho/(edge_num*np.min(degrees)))
            # print(f'deg {degrees}, edge_nume {edge_num}, gam {gam}')
            
            H1 = 0.5 * gamma * L_const
            H2 = 0.5 * gamma * L_const
            D = gamma * np.diag(degrees)
            A = D - H1
            
            A_half = np.real(scipy.linalg.sqrtm(A))
            H2_half = np.real(scipy.linalg.sqrtm(H2))
            
        elif param_setting == 'DistADMM':
            # H = \tilde{H} = L_mh^2
            # D = rho * diag(sum_j L_mh_{ij}^2 * (deg_j+1))
            # A = D - rho*H
            self.rho_only_in_mat = False
            self.name += '_DistADMM'
            
            H1 = L_mh @ L_mh
            H2 = L_mh @ L_mh
            H2_half = L_mh
            diag = np.zeros(N)
            for i in range(N):
                diag[i] = np.inner(L_mh[i], (degrees+1)*L_mh[i])
            D = rho * np.diag(diag)
            A = D - rho*H1
            A_half = np.real(scipy.linalg.sqrtm(A))
        
        elif param_setting == 'ALT':
            # A = \rho (I - L_lm)^2
            # H = L_lm (2I - L_lm)
            # \tilde{H} = L_lm^2
            # D = \rho I
            
            self.rho_only_in_mat = False
            self.name = 'ALT'
            A_half = sqrt(rho) * (I - L_lm)
            A = A_half @ A_half
            D = rho * I
            H1 = L_lm @ (2*I - L_lm)
            H2_half = L_lm
            H2 = L_lm @ L_lm
        
        elif param_setting == 'DPMM':
            self.rho_only_in_mat = False
            self.name = 'DPMM'
            
            A, A_half, H1, H2, H2_half, D \
                = self.set_weight_param('PEXTRA', adj_mat, rho)
            self.name = self.name[:-7] # delete '_PEXTRA'
        
        
        # A_half = np.real(scipy.linalg.sqrtm(A))
        # H2_half = np.real(scipy.linalg.sqrtm(H2))
        
        return A, A_half, H1, H2, H2_half, D
    