import numpy as np
import cvxpy as cp
import math

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class B_DPP:
    '''
    a, D, R: problem data
    C: constant for the step size
    '''
    def __init__(self, network, a, D, R, C):
        self.W_subG = network
        self.subG_num = self.W_subG.shape[0] # 4

        self.N = a.shape[0]
        self.d = a.shape[1]
        self.m = D.shape[1]
        self.a = a
        self.D = D
        self.R = R
        self._set_argmin_prob()
        self.C = C

    def _set_argmin_prob(self):
        self.var_xi = cp.Variable(self.d)
        var_gi = cp.Variable(self.m)
        var_diff_x_x = cp.Variable(self.d)
        var_diff_x_a = cp.Variable(self.d)

        self.param_V_tp1 = cp.Parameter(nonneg=True)
        self.param_eta_tp1 = cp.Parameter(nonneg=True)
        self.param_hat_mu = cp.Parameter(self.m)
        self.param_xit = cp.Parameter(self.d)
        self.param_di = cp.Parameter((self.m, self.d))
        self.param_Ri = cp.Parameter(self.m) # R/N
        self.param_ai = cp.Parameter(self.d)

        obj = self.param_V_tp1 * 0.5 * cp.quad_form(var_diff_x_a, np.identity(self.d)) + self.param_hat_mu @ var_gi + self.param_eta_tp1 * cp.quad_form(var_diff_x_x, np.identity(self.d))

        cons = [
            var_gi == self.param_di @ self.var_xi - self.param_Ri,
            var_diff_x_x == self.var_xi - self.param_xit,
            var_diff_x_a == self.var_xi - self.param_ai,
            self.var_xi >= 0.,
            self.var_xi <= 2.
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, V_tp1, eta_tp1, hat_mu, xit, di, Ri, ai):
        self.param_V_tp1.value = V_tp1
        self.param_eta_tp1.value = eta_tp1
        self.param_hat_mu.value = hat_mu
        self.param_xit.value = xit
        self.param_di.value = di
        self.param_Ri.value = Ri
        self.param_ai.value = ai
        self.prob.solve(solver=cp.MOSEK)
        return self.var_xi.value
    
    def fi(self, i, xi):
        return 0.5 * (xi-self.a[i]) @ (xi-self.a[i])
    
    def gi(self, i, xi):
        return self.D[i] @ xi - self.R/self.N

    def reset(self):
        self.x = np.zeros((self.N, self.d))+2
        self.mu = np.zeros((self.N, self.m))
        self.ite_num = 0
        self.x_tp1 = np.zeros((self.N, self.d))
        self.mu_tp1 = np.zeros((self.N, self.m))

        # for running average
        self.x_tot = self.x.copy()

    def step(self):
        subG_idx = self.ite_num % self.subG_num # 0 1 2 3
        V_tp1 = math.sqrt(self.ite_num+1)
        eta_tp1 = self.ite_num + 1
        gamma_tp1 = self.C / math.sqrt(self.ite_num + 1)
        for i in range(self.N):
            hat_mu_i = self.W_subG[subG_idx, i] @ self.mu
            xit = self.x[i]
            di = self.D[i]
            Ri = self.R/self.N
            ai = self.a[i]
            xi_tp1 = self._solve_argmin_prob(V_tp1, eta_tp1, hat_mu_i, xit, di, Ri, ai)
            mu_i_tp1 = np.maximum(hat_mu_i + self.gi(i, xi_tp1), np.zeros(self.m)) + gamma_tp1 * np.ones(self.m)
            self.x_tp1[i] = xi_tp1
            self.mu_tp1[i] = mu_i_tp1

        self.x = self.x_tp1.copy()
        self.mu = self.mu_tp1.copy()
        self.ite_num += 1
        self.x_tot += self.x

    def compute_metrics(self, opt_val):
        run_avg = self.x_tot / (self.ite_num+1)
        fun_val = 0.
        cons_val = 0.
        for i in range(self.N):
            fun_val += self.fi(i, run_avg[i])
            cons_val += self.gi(i, run_avg[i])
        obj_err = np.abs(fun_val-opt_val)
        return obj_err, cons_val

class Falsone:
    '''
    a, D, R: problem data
    beta: constant for the step size
    '''
    def __init__(self, network, a, D, R, beta):
        self.W_subG = network
        self.subG_num = self.W_subG.shape[0] # 4

        self.N = a.shape[0]
        self.d = a.shape[1]
        self.m = D.shape[1]
        self.a = a
        self.D = D
        self.R = R
        self._set_argmin_prob()
        self.beta = beta

    def _set_argmin_prob(self):
        self.var_xi = cp.Variable(self.d)
        var_gi = cp.Variable(self.m)
        var_diff_x_a = cp.Variable(self.d)

        self.param_lik = cp.Parameter(self.m)
        self.param_di = cp.Parameter((self.m, self.d))
        self.param_Ri = cp.Parameter(self.m) # R/N
        self.param_ai = cp.Parameter(self.d)
        
        obj = 0.5 * cp.quad_form(var_diff_x_a, np.identity(self.d)) + self.param_lik @ var_gi

        cons = [
            var_gi == self.param_di @ self.var_xi - self.param_Ri,
            var_diff_x_a == self.var_xi - self.param_ai,
            self.var_xi >= 0.,
            self.var_xi <= 2.
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dcp(dpp=True)

    def _solve_argmin_prob(self, lik, di, Ri, ai):
        self.param_lik.value = lik
        self.param_di.value = di
        self.param_Ri.value = Ri
        self.param_ai.value = ai
        self.prob.solve(solver=cp.MOSEK)
        return self.var_xi.value
    
    def fi(self, i, xi):
        return 0.5 * (xi-self.a[i]) @ (xi-self.a[i])
    
    def gi(self, i, xi):
        return self.D[i] @ xi - self.R/self.N
    
    def reset(self):
        self.hat_x = np.zeros((self.N, self.d))+2
        self.lambd = np.zeros((self.N, self.m))
        self.hat_x_kp1 = np.zeros((self.N, self.d))
        self.lambd_kp1 = np.zeros((self.N, self.m))

        self.ite_num = 0
        self.sum_c = 0.
        # for running average
        self.x_tot = self.hat_x.copy()

    def step(self):
        subG_idx = self.ite_num % self.subG_num # 0 1 2 3
        c_k = self.beta / (self.ite_num+1)
        self.sum_c += c_k
        for i in range(self.N):
            lik = self.W_subG[subG_idx, i] @ self.lambd
            di = self.D[i]
            Ri = self.R/self.N
            ai = self.a[i]
            xi_kp1 = self._solve_argmin_prob(lik, di, Ri, ai)
            lambd_i_kp1 = np.maximum(lik+c_k*self.gi(i, xi_kp1), np.zeros(self.m))
            hat_xi_kp1 = self.hat_x[i] + (c_k/self.sum_c) * (xi_kp1 - self.hat_x[i])
            self.hat_x_kp1[i] = hat_xi_kp1
            self.lambd_kp1[i] = lambd_i_kp1

        self.hat_x = self.hat_x_kp1.copy()
        self.lambd = self.lambd_kp1.copy()
        self.ite_num += 1
        self.x_tot += self.hat_x

    def compute_metrics(self, opt_val):
        run_avg = self.x_tot / (self.ite_num+1)
        fun_val = 0.
        cons_val = 0.
        for i in range(self.N):
            fun_val += self.fi(i, run_avg[i])
            cons_val += self.gi(i, run_avg[i])
        obj_err = np.abs(fun_val-opt_val)
        return obj_err, cons_val

class C_SP_SG:
    def __init__(self, network, a, D, R, delta_prime):
        self.W_subG = network
        self.subG_num = self.W_subG.shape[0] # 4

        self.N = a.shape[0]
        self.d = a.shape[1]
        self.m = D.shape[1]
        self.a = a
        self.D = D
        self.R = R
        self.delta_prime = delta_prime # any value from 0~1

        self.sigma = self.get_cons_stepsize()
        self.r = self.get_radius()
        self._set_argmin_P_Wi()
        self._set_argmin_P_B()

    def _set_argmin_P_Wi(self):
        self.var_wi = cp.Variable(self.d)
        self.param_wi = cp.Parameter(self.d)
        obj = cp.norm(self.param_wi-self.var_wi)
        cons = [
            self.var_wi >= 0.,
            self.var_wi <= 2.
        ]
        self.prob_PWi = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob_PWi.is_dcp(dpp=True)

    def _set_argmin_P_B(self):
        '''argmin problem for projection on B(0,r)'''
        self.var_zi = cp.Variable(self.m)
        self.param_zi = cp.Parameter(self.m)
        obj = cp.norm(self.param_zi-self.var_zi)
        cons = [
            self.var_zi >= 0.,
            cp.norm(self.var_zi) <= self.r
        ]
        self.prob_PB = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob_PB.is_dcp(dpp=True)

    def get_cons_stepsize(self):
        # get delta
        delta = np.amin(self.W_subG[np.nonzero(self.W_subG)]) - 0.01
        self.delta = delta
    
        # get dmax
        dmax = np.amax(np.sum(self.W_subG, axis=2))
        self.dmax = dmax

        delta_prime = self.delta_prime # any value from 0~1
        delta_tilde = min(delta_prime, (1-delta_prime) * (self.delta/self.dmax))
        lb = delta_tilde / self.delta
        ub = (1-delta_tilde) / self.dmax
        if ub < lb:
            raise ValueError('Upper bound of the step size is smaller than the lower bound!')
        sigma = lb
        return sigma
    
    def get_lr(self):
        m = math.floor(math.log2(self.ite_num))
        return 1 / math.sqrt(2**m)
    
    def gi(self, i, xi):
        return self.D[i] @ xi - self.R/self.N
    
    def grad_gi(self, i):
        return self.D[i]
    
    def fi(self, i, xi):
        return 0.5 * (xi-self.a[i]) @ (xi-self.a[i])
    
    def grad_fi(self, i, xi):
        return xi-self.a[i]
    
    def get_radius(self):
        x_slater = np.ones((self.N, self.d))

        # Check whether x_slater is the Slater point
        sum_gi = 0
        for i in range(self.N):
            sum_gi += self.gi(i, x_slater[i])
        if np.any(sum_gi >= 0):
            raise ValueError("Sum of gi >= 0. Choose another x_slater")

        gamma = np.min(-sum_gi)

        # compute f_slater
        f_slater = 0
        for i in range(self.N):
            f_slater += self.fi(i, x_slater[i])

        q = 0
        r = (f_slater-q) / gamma
        return r
    
    def P_Wi(self, wi):
        self.param_wi.value = wi
        self.prob_PWi.solve(solver=cp.MOSEK)
        return self.var_wi.value
    
    def P_B(self, zi):
        self.param_zi.value = zi
        self.prob_PB.solve(solver=cp.MOSEK)
        return self.var_zi.value
    
    def reset(self):
        self.w = np.zeros((self.N, self.d))+2
        self.z = np.zeros((self.N, self.m))
        self.w_tp1 = np.zeros((self.N, self.d))
        self.z_tp1 = np.zeros((self.N, self.m))

        self.ite_num = 1
        # for running average
        self.w_tot = self.w.copy()

    def step(self):
        subG_idx = (self.ite_num-1) % self.subG_num # 0 1 2 3
        eta_t = self.get_lr()
        for i in range(self.N):
            grad_fi = self.grad_fi(i, self.w[i])
            hat_wi_tp1 = self.w[i] - eta_t * (grad_fi + self.grad_gi(i).T @ self.z[i])
            hat_zi_tp1 = self.z[i] + self.sigma * (self.W_subG[subG_idx, i] @ self.z - np.sum(self.W_subG[subG_idx, i]) * self.z[i]) + eta_t * self.gi(i, self.w[i])
            wi_tp1 = self.P_Wi(hat_wi_tp1)
            zi_tp1 = self.P_B(hat_zi_tp1)
            self.w_tp1[i] = wi_tp1
            self.z_tp1[i] = zi_tp1

        self.w = self.w_tp1.copy()
        self.z = self.z_tp1.copy()
        self.ite_num += 1
        self.w_tot += self.w

    def compute_metrics(self, opt_val):
        run_avg = self.w_tot / self.ite_num
        fun_val = 0.
        cons_val = 0.
        for i in range(self.N):
            fun_val += self.fi(i, run_avg[i])
            cons_val += self.gi(i, run_avg[i])
        obj_err = np.abs(fun_val-opt_val)
        return obj_err, cons_val

class DPD_TV:
    def __init__(self, adj_subG, a, D, R, beta):
        self.adj_subG = adj_subG
        self.subG_num = self.adj_subG.shape[0] # 4

        self.N = a.shape[0]
        self.d = a.shape[1]
        self.m = D.shape[1]
        self.a = a
        self.D = D
        self.R = R
        self.beta = beta

        self.M = self.get_M()

    def gi(self, i, xi):
        return self.D[i] @ xi - self.R/self.N
    
    def fi(self, i, xi):
        return 0.5 * (xi-self.a[i]) @ (xi-self.a[i])

    def get_M(self):
        x_slater = np.ones((self.N, self.d))

        # Check whether x_slater is the Slater point
        sum_gi = 0
        for i in range(self.N):
            sum_gi += self.gi(i, x_slater[i])
        if np.any(sum_gi >= 0):
            raise ValueError("Sum of gi >= 0. Choose another x_slater")
        
        gamma = np.min(-sum_gi)
        # compute f_slater
        f_slater = 0
        for i in range(self.N):
            f_slater += self.fi(i, x_slater[i])
        M = f_slater / gamma
        return M
    
    def reset(self):
        self.y = np.zeros((self.N, self.m))
        self.x = np.zeros((self.N, self.d))
        self.mu = np.zeros((self.N, self.m))

        self.ite_num = 0
        # for running average
        self.x_tot = np.zeros((self.N, self.d))+2

    def step(self):
        subG_idx = self.ite_num % self.subG_num # 0 1 2 3
        alpha_t = self.beta / (self.ite_num+1)**0.6
        for i in range(self.N):
            var_xi = cp.Variable(self.d)
            var_rhoi = cp.Variable()
            obj = 0.5 * cp.quad_form(var_xi-self.a[i], np.identity(self.d)) + self.M * var_rhoi
            cons = [
                self.gi(i, var_xi) <= self.y[i] + var_rhoi * np.ones(self.m),
                var_xi >= 0.,
                var_xi <= 2.,
                var_rhoi >= 0.
            ]
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(solver=cp.MOSEK)
            self.x[i] = var_xi.value
            self.mu[i] = cons[0].dual_value

        for i in range(self.N):
            self.y[i] += alpha_t * (np.sum(self.adj_subG[subG_idx, i]) * self.mu[i] - self.adj_subG[subG_idx, i] @ self.mu)

        self.ite_num += 1
        self.x_tot += self.x

    def compute_metrics(self, opt_val):
        run_avg = self.x_tot / (self.ite_num+1)
        fun_val = 0.
        cons_val = 0.
        for i in range(self.N):
            fun_val += self.fi(i, run_avg[i])
            cons_val += self.gi(i, run_avg[i])
        obj_err = np.abs(fun_val-opt_val)
        return obj_err, cons_val