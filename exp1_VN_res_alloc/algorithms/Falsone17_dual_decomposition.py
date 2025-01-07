import numpy as np
import cvxpy as cp
import math

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class Falsone17DualDecomp:
    '''
    a, D, R: problem data
    beta: constant for the step size
    '''
    
    name = "Dual_Decomposition"
    
    def __init__(self, prob, network, beta):
        '''
        hat_x, hat_x_kp1, x_tot: N dimensional vector
        lambd, lambd_kp1: N x d matrix

        a: N x 1 matrix
        D: N x d matrx
        R: d dimensional vector
        '''
        
        self.W_subG = network
        self.subG_num = self.W_subG.shape[0] # 1

        self.N = prob.N
        self.d = prob.d
        
        self.a = prob.a
        self.D = prob.D
        self.R = prob.R
        
        self._set_argmin_prob()
        self.reset()
        self.beta = beta

    def _set_argmin_prob(self):
        '''
        var_xi: 1 dimensional vector
        var_gi: d dimensional vector
        param_ai: 1 dimesional vector
        param_lik: d dimensional vector
        param_di: d dimensional vector
        param_Ri: d dimensional vector
        '''
        
        self.var_xi = cp.Variable(1)
        var_gi = cp.Variable(self.d)
        # var_diff_x_a = cp.Variable(1)
        # print(self.var_xi.shape)

        self.param_ai = cp.Parameter(1)
        self.param_lik = cp.Parameter(self.d)
        self.param_di = cp.Parameter(self.d)
        # print(self.param_di.shape)
        self.param_Ri = cp.Parameter(self.d) # R/N
        
        obj = 0.5 * cp.quad_form(self.var_xi - self.param_ai, np.identity(1)) + cp.vdot(self.param_lik, var_gi) # inner product

        cons = [
            var_gi == self.var_xi * self.param_di  - self.param_Ri,
            self.var_xi >= 0.,
            self.var_xi <= 2.
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, lik, di, Ri, ai):
        '''
        param_lik: d dimensional vector
        param_di: d dimensional vector
        param_Ri: d dimensional vector
        param_ai: 1 dimensional vector
        
        -> 1 dimensional vector
        '''
        
        self.param_lik.value = lik
        self.param_di.value = di
        self.param_Ri.value = Ri
        self.param_ai.value = ai
        self.prob.solve(solver=cp.OSQP)
        
        return self.var_xi.value
    
    def fi(self, i, xi): 
        ''' i: int, xi: float -> float'''
        
        return 0.5 * (xi - self.a[i, 0]) * (xi - self.a[i, 0])
    
    def gi(self, i, xi):
        ''' i: int, xi: float -> d dimensional vector'''
        
        # print(self.D[i] @ xi - self.R/self.N)
        # print(np.zeros(self.R.shape))
        # print(np.max(np.c_[self.D[i] @ xi - self.R/self.N, np.zeros(self.R.shape)], axis=1).shape)
        return xi * self.D[i] - self.R/self.N
    
    def local_set_violation(self, xi):
        ''' xi: int'''
        
        return max(-xi, 0) + max(xi-2, 0)
    
    def reset(self):
        # print('reset')
        self.hat_x = np.zeros(self.N) + 2
        self.hat_x_kp1 = np.zeros(self.N)
        self.x_tot = self.hat_x.copy() # for running average
        
        self.lambd = np.zeros((self.N, self.d))
        self.lambd_kp1 = np.zeros((self.N, self.d))

        self.iter_num = 0
        self.sum_c = 0.

    def step(self):
        # print(f'\nstep {self.iter_num}')
        subG_idx = self.iter_num % self.subG_num # for time-varying graph
        
        c_k = self.beta / (self.iter_num+1)
        self.sum_c += c_k
        
        for i in range(self.N):
            lik = self.W_subG[subG_idx, i] @ self.lambd # weighted average
            # print(self.W_subG[subG_idx, i].shape)
            # print(self.lambd.shape)
            # print(f'lik {lik.shape}')
            
            di = self.D[i]
            Ri = self.R/self.N
            ai = self.a[i]
            # print(f'lik {lik.shape}, di {di.shape}, Ri {Ri.shape}, ai {ai.shape}')
            
            xi_kp1 = self._solve_argmin_prob(lik, di, Ri, ai)
            # print(f'xi_kp1 {xi_kp1.shape}')
            lambd_i_kp1 = np.maximum(lik+c_k*self.gi(i, xi_kp1[0]), np.zeros(self.d))
            hat_xi_kp1 = self.hat_x[i] + (c_k/self.sum_c) * (xi_kp1 - self.hat_x[i])
            # print(f'lambd_i_kp1 {lambd_i_kp1.shape}, hat_xi_kp1 {hat_xi_kp1.shape}')
            self.hat_x_kp1[i] = hat_xi_kp1
            self.lambd_kp1[i] = lambd_i_kp1

        self.hat_x = self.hat_x_kp1.copy()
        self.lambd = self.lambd_kp1.copy()
        # print(f'hat_x {self.hat_x.shape}, lambd {self.lambd.shape}')
        
        self.iter_num += 1
        self.x_tot += self.hat_x

    def compute_metrics(self, opt_val):
        # print(f'compute matrics')
        run_avg = self.x_tot / (self.iter_num+1)
        # print(f'run_avg {run_avg}')
        fun_val = 0.
        cons_val = 0. # constraitn value
        cons_vio = 0. # constraint violation (including local set violation)
        for i in range(self.N):
            # print(f'run_avg[{i}] {run_avg[i]}')
            fun_val += self.fi(i, run_avg[i])
            # print(f'fun_val {fun_val}')
            cons_val += self.gi(i, run_avg[i]) # d dimensional vector
            cons_vio += self.local_set_violation(run_avg[i])
            # tmp = np.c_[self.gi(i, run_avg[i]), np.zeros(self.d)]
            # print(tmp)
            # print(np.max(tmp, axis=1))
            # print(np.sum(np.max(tmp, axis=1)))
            # raise ValueError
            # cons_vio += np.sum(np.max(tmp, axis=1)) \
            #             + self.local_set_violation(run_avg[i]) 
                        
        tmp = np.c_[cons_val, np.zeros(self.d)]
        cons_vio += np.sum(np.max(tmp, axis=1))
        # print(f'cons_vio {cons_vio}')
            
        obj_err = fun_val-opt_val
        # print(f'{fun_val}, {opt_val}')
        # print(f'obj_err {obj_err}')
        return obj_err, cons_vio
