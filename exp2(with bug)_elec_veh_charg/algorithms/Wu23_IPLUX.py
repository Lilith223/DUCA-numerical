import numpy as np
import cvxpy as cp
from math import log
from time import time

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class IPLUX:
    '''
    a, D, R: problem data
    beta: constant for the step size
    '''
    
    name = "IPLUX"
    
    def __init__(
        self, 
        prob, 
        network, 
        alpha, 
        rho, 
        no_ineq=False, 
        verbose=0, 
        log_dir='log'):
        '''
        these are set in 'reset()':
        x_cur, x_avg: N dimensional vector
        t_cur: N dimensional vector
        u_cur: N dimensional vector
        z_cur: N dimensional vector
        q_cur: N dimensional vector

        these are set here:
        c: N dimensional vector
        d: N dimensional vector
        b: 1 dimensional vector
        x_star: N dimensional vector
        opt_val: float
        
        no_ineq: if True, then there is no inequality
        verbose: the bigger this is, more info displays
        init_time
        '''
        
        self.W = (np.identity(prob.N) + network) * 0.5
        self.H = (np.identity(prob.N) - network) * 0.5
        # self.W = network
        # self.H = np.identity(prob.N) - network

        self.N = prob.N
        self.c = prob.c
        self.d = prob.d
        self.b = prob.b
        self.x_star = prob.x_star
        self.opt_val = prob.opt_val
        
        self.alpha = alpha
        self.rho = rho
        
        # prepare the alg to start
        self.iter_num = 0
        
        # logs
        self.x_log = []
        self.obj_err_log = []
        self.cons_vio_log = []
        self.x_avg_log = []
        self.obj_err_avg_log = []
        self.cons_vio_avg_log = []
        
        self.no_ineq = no_ineq
        self.verbose = verbose
        self.log_dir = log_dir
        self.init_time = time()
        
        self._set_argmin_prob(no_ineq=no_ineq)
        self.reset()

    def reset(self):
        '''
        x_cur, x_nxt, x_avg: N dimensional vector
        t_cur, t_nxt: N dimensional vector
        u_cur, u_nxt: N dimensional vector
        z_cur, z_nxt: N dimensional vector
        q_cur, q_nxt: N dimensional vector
        
        iter_num = 0
        '''
        
        print('reset')
        self.x_cur = np.zeros(self.N) + 1
        # self.x_cur = self.x_star
        self.x_nxt = np.zeros(self.N)
        self.x_avg = self.x_cur.copy() # for running average
        
        self.t_cur = np.zeros(self.N)
        # for i in range(self.N):
        #     self.t_cur[i] = self.gi(i, self.x_cur[i])
        self.t_nxt = np.zeros(self.N)
        
        self.u_cur = np.zeros(self.N)
        self.u_nxt = np.zeros(self.N)
        
        self.z_cur = np.zeros(self.N)
        self.z_nxt = np.zeros(self.N)
        
        self.q_cur = np.zeros(self.N)
        for i in range(self.N):
            Gi_yi0 = self.gi(i, self.x_cur[i]) - self.t_cur[i]
            self.q_cur[i] = max(0, -Gi_yi0)
        self.q_nxt = np.zeros(self.N)

        self.iter_num = 0
        
        # reset logs
        self.obj_err_log = []
        self.obj_err_avg_log = []
        self.cons_vio_log = []
        self.cons_vio_avg_log = []

        self.x_log = []
        self.x_avg_log = []

        self.make_log()
    
    
    def make_log(self):
        ''' log infomation, save every 100 iterations '''
        
        # last iterate
        obj_err, cons_vio = self.compute_metrics()
        self.obj_err_log.append(obj_err)
        self.cons_vio_log.append(cons_vio)
        self.x_log.append(self.x_cur)
        
        # running average
        obj_err_avg, cons_vio_avg = self.compute_metrics(avg=True)
        self.obj_err_avg_log.append(obj_err_avg)
        self.cons_vio_avg_log.append(cons_vio_avg)
        self.x_avg_log.append(self.x_avg)
        
        # logging.info(f'iter {self.iter_num}, obj err: {obj_err:.2e}, cons vio: {cons_vio:.2e}')
        
        if self.iter_num>0 and self.iter_num%100==0:
            # last iterate
            np.savetxt(f'{self.log_dir}/obj_err_{self.name}.txt', self.obj_err_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/cons_vio_{self.name}.txt', self.cons_vio_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/x_{self.name}.txt', self.x_log, delimiter=',')
            # running average
            np.savetxt(f'{self.log_dir}/obj_err_avg_{self.name}.txt', self.obj_err_avg_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/cons_vio_avg_{self.name}.txt', self.cons_vio_avg_log, delimiter=',')
            np.savetxt(f'{self.log_dir}/x_avg_{self.name}.txt', self.x_avg_log, delimiter=',')
            print('saved')

    def _set_argmin_prob(self, no_ineq=False):
        '''
        var_xi: 1 dimensional vector

        param_ci: 1 dimesional vector
        param_xik: 1 dimensional vector
        param_qGyd: 1 dimensional vector, (q_i^k + G_i(y_i^k)) * d_i
        '''
        
        self.var_xi = cp.Variable(1)

        self.param_ci = cp.Parameter(1)
        self.param_xik = cp.Parameter(1)
        self.param_qGyd = cp.Parameter(1, nonneg=True) # (q_i^k + G_i(y_i^k)) * d_i
        # print(f'self.param_ci {self.param_ci.shape}')
        
        obj = 0.
        obj += self.param_ci[0] * self.var_xi[0]
        obj += self.alpha * 0.5 * \
                cp.quad_form(self.var_xi - self.param_xik, np.identity(1))
        if not no_ineq:
            obj +=  self.param_qGyd[0] * (-cp.log(self.var_xi[0] + 1))

        cons = [
            self.var_xi >= 0.,
            self.var_xi <= 1.
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        print(f'self.prob {self.prob}')
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, ci, xik, qGyd):
        '''
        ci: float
        xik: float
        qGyd: float, (q_i^k + G_i(y_i^k)) * d_i
        
        -> 1 dimensional vector
        '''
        
        # print(f'para: ci {ci}, xik {xik}, qGyd, {qGyd}')
        self.param_ci.value = [ci]
        self.param_xik.value = [xik]
        self.param_qGyd.value = [qGyd]
        self.prob.solve()
        
        return self.var_xi.value
    
    def fi(self, i, xi): 
        ''' i: int, xi: float -> float'''
        return self.c[i] * xi
    
    def gi(self, i, xi):
        ''' i: int, xi: float -> float'''
        return -self.d[i] * log(xi + 1) + self.b[0]/self.N
    
    def local_set_violation(self, xi):
        ''' xi: float -> float'''
        return max(-xi, 0) + max(xi-1, 0)

    def compute_metrics(self, avg=False):
        ''' -> obj_err: float, cons_vio: float'''
        
        fun_val = 0.
        cons_val = np.zeros(1)  # constraint value
        # constraint violation, including local set violation
        cons_vio = 0.
        
        x = self.x_cur
        if avg:
            x = self.x_avg 
        
        for i in range(self.N):            
            fun_val += self.fi(i, x[i])            
            cons_val += self.gi(i, x[i]) # d dimensional vector            
            cons_vio += self.local_set_violation(x[i]) 
                        
        tmp = np.c_[cons_val, np.zeros(1)]
        cons_vio += np.sum(np.max(tmp, axis=1))
        obj_err = fun_val - self.opt_val
        
        return obj_err, cons_vio

    def step(self):
        if self.verbose >= 1:
            logging.info(f'\nstep {self.iter_num}')
        
        u_wavg = self.W @ self.u_cur # weighted average by W
        # print(f'u_wavg {u_wavg.shape}')
        
        for i in range(self.N):
            ci = self.c[i]
            di = self.d[i]
            xik = self.x_cur[i]
            tik = self.t_cur[i]
            zik = self.z_cur[i]
            qik = self.q_cur[i]
            Gi_yik = self.gi(i, xik) - tik # G_i(y_i^k)
            # u_wavg = self.W[i] @ self.u_cur # weighted average by W
            # u_havg = self.H[i] @ self.u_cur # weighted average by H
            # print(f'u {self.u_cur}, u_wavg {u_wavg}, u_havg {u_havg}')
            
            # (q_i^k + G_i(y_i^k)) * d_i
            qGyd = (qik + Gi_yik) * di
            # print(f'qGyd {qGyd}')
            
            # updates
            self.x_nxt[i] = self._solve_argmin_prob(ci, xik, qGyd)
            self.t_nxt[i] = (
                self.alpha * tik 
                + zik / self.rho - u_wavg[i]
                + qik + Gi_yik 
                ) / (self.alpha + 1 / self.rho)
            self.u_nxt[i] = u_wavg[i] + (self.t_nxt[i] - zik) / self.rho
            # z_nxt needs all u_nxt[j]
            Gi_yi_nxt = self.gi(i, self.x_nxt[i]) - self.t_nxt[i]
            self.q_nxt[i] = max(-Gi_yi_nxt, qik + Gi_yi_nxt)
            
            if self.verbose >= 1:
                logging.info(f'q{i}+G{i}(y{i}) {self.q_nxt[i] + Gi_yi_nxt:.2e}')
            
        # z_nxt needs all u_nxt[j]
        self.z_nxt = self.z_cur + self.rho * (self.H @ self.u_nxt)
        
        if self.verbose >= 1:
            logging.info(f'dx {self.x_nxt-self.x_cur}, dt {self.t_nxt-self.t_cur}, du {self.u_nxt-self.u_cur}, dz {self.z_nxt-self.z_cur}, dq {self.q_nxt-self.q_cur}')
        
        self.x_cur = self.x_nxt.copy()
        self.t_cur = self.t_nxt.copy()
        self.u_cur = self.u_nxt.copy()
        self.z_cur = self.z_nxt.copy()
        self.q_cur = self.q_nxt.copy()
        # print(f'hat_x {self.hat_x.shape}, lambd {self.lambd.shape}')
        
        self.iter_num += 1
        self.x_avg = (self.x_avg * self.iter_num + self.x_cur) / (self.iter_num + 1)
        
        self.make_log()
        if self.iter_num % 100 == 0:
            logging.info(f"running time for {self.name} = {time()-self.init_time:.2f}")
            
        
    
    