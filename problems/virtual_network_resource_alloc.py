#%%

import numpy as np
import cvxpy as cp
import os

class VNResAlloc:
    '''Resource allocation problem in 5G virtualized networks [in Wang et al. LCSS 23]'''
    
    prob_tpye = "VNResAlloc"
    
    def __init__(self, N, d=1):
        '''
            self.N = N  # number of nodes
            self.d = d  # dimension of inequalities
        '''
        
        self.N = N
        self.d = d
        
        self.x_star = 0.
        self.opt_val = 0.
        
        self.prob_name = f"N{self.N}_d{self.d}" # name of problem instance
        
        self.save_dir = f'data/problem/{self.prob_tpye}/{self.prob_name}'
        # Save data to corresponding files
        

    def gen(self):
        '''Generate a VN resource allocation problem, then save it in the data folder'''

        # generate problem parameters
        print("generating a VN resource allocation problem")
        self.a = np.random.uniform(low=1., high=2., size=(self.N, 1))
        self.D = np.random.uniform(low=0.5, high=1., size=(self.N, self.d))
        self.R = np.random.uniform(low=5., high=10., size=self.d)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        np.savetxt(f'{self.save_dir}/a.txt', self.a)
        np.savetxt(f'{self.save_dir}/D.txt', self.D)
        np.savetxt(f'{self.save_dir}/R.txt', self.R)
        np.save(f'{self.save_dir}/a.npy', self.a)
        np.save(f'{self.save_dir}/D.npy', self.D)
        np.save(f'{self.save_dir}/R.npy', self.R)


        # use cvxpy to solve the problem 
        var_x = cp.Variable((self.N, 1))
        obj = 0.
        for i in range(self.N):
            obj = obj + 0.5 * cp.quad_form(var_x[i]-self.a[i], np.identity(1))
        # print(obj)

        coupling_g = np.zeros(self.d)
        for i in range(self.N):
            # print(var_x[i].shape)
            coupling_g = coupling_g + var_x[i] * self.D[i]
        # print(coupling_g.shape)
        coupling_cons = [coupling_g <= self.R]
        local_cons1 = [var_x[i] >= 0. for i in range(self.N)]
        local_cons2 = [var_x[i] <= 2. for i in range(self.N)]
        cons = coupling_cons + local_cons1 + local_cons2

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.prob.solve(solver=cp.OSQP)
        self.x_star = var_x.value
        self.opt_val = self.prob.value

        np.savetxt(f'{self.save_dir}/x_star.txt', self.x_star)
        np.savetxt(f'{self.save_dir}/opt_val.txt', [self.opt_val])
        
        print(f"generated problem saved in {self.save_dir}")
    
    def load(self):
        print("loading a VN resource allocation problem")
        try:
            self.a = np.load(f'{self.save_dir}/a.npy')
            self.D = np.load(f'{self.save_dir}/D.npy')
            # print(self.D.shape)
            self.R = np.load(f'{self.save_dir}/R.npy')
            self.x_star = np.loadtxt(f'{self.save_dir}/x_star.txt')
            self.opt_val = np.loadtxt(f'{self.save_dir}/opt_val.txt')
            print("problem loaded:")
            print(f'a {self.a.shape}')
            print(f'D {self.D.shape}')
            print(f'R {self.R.shape}')
        except:
            print("failed to load data in", self.save_dir)
            raise(ValueError)
        
if __name__ == '__main__':
    N = 3
    d = 4
    prob = VNResAlloc(N, d=d)
    # prob.gen()
    prob.load()
