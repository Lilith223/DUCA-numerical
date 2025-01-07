#%%

import numpy as np
import cvxpy as cp
import os

class ElecVehCharg:
    '''Plug-in electric vehicles charging problem [in Wang et al. LCSS 23]'''
    
    prob_tpye = "ElecVehCharg"
    
    def __init__(self, N):
        '''
            problem parameters:
            N: number of nodes
            
            c: N dimensional vector
            d: N dimensional vector
            b: 1 dimensional vector
            
            x_star: 
            opt_val
        '''
        
        self.N = N
        
        self.x_star = 0.
        self.opt_val = 0.
        
        self.prob_name = f"N{self.N}" # name of problem instance
        
        self.save_dir = f'data/problem/{self.prob_tpye}/{self.prob_name}'
        # Save data to corresponding files
        

    def gen(self, no_ineq=False):
        ''' Generate a ElecVehCharg problem, then save it in the data folder. '''

        # generate problem parameters
        print(f"generating a {self.prob_tpye} problem, N={self.N}")
        self.c = np.random.uniform(low=0., high=1., size=self.N)
        self.d = np.random.uniform(low=0., high=1., size=self.N)
        self.b = np.ndarray(1, dtype=np.float64)
        self.b[0] = 0.1*self.N
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        np.savetxt(f'{self.save_dir}/c.txt', self.c)
        np.savetxt(f'{self.save_dir}/d.txt', self.d)
        np.savetxt(f'{self.save_dir}/b.txt', self.b)
        np.save(f'{self.save_dir}/c.npy', self.c)
        np.save(f'{self.save_dir}/d.npy', self.d)
        np.save(f'{self.save_dir}/b.npy', self.b)


        # use cvxpy to solve the problem 
        var_x = cp.Variable(self.N)
        obj = 0.
        for i in range(self.N):
            obj = obj + self.c[i] * var_x[i]
        # print(obj)

        coupling_g = 0.
        for i in range(self.N):
            # print(var_x[i].shape)
            coupling_g = coupling_g - self.d[i] * cp.log(var_x[i] + 1)
        # print(coupling_g.shape)
        coupling_cons = [coupling_g <= -self.b[0]]
        local_cons1 = [var_x[i] >= 0. for i in range(self.N)]
        local_cons2 = [var_x[i] <= 1. for i in range(self.N)]
        
        if no_ineq:
            cons = local_cons1 + local_cons2
        else:
            cons = coupling_cons + local_cons1 + local_cons2

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.prob.solve()
        self.x_star = var_x.value
        self.opt_val = self.prob.value

        print(f'x* {self.x_star}, f* {self.opt_val}')

        np.savetxt(f'{self.save_dir}/x_star.txt', self.x_star)
        np.savetxt(f'{self.save_dir}/opt_val.txt', [self.opt_val])
        
        print(f"generated problem saved in {self.save_dir}")
    
    def load(self):
        print(f"loading a {self.prob_tpye} problem, N={self.N}")
        try:
            self.c = np.load(f'{self.save_dir}/c.npy')
            self.d = np.load(f'{self.save_dir}/d.npy')
            self.b = np.load(f'{self.save_dir}/b.npy')
            self.x_star = np.loadtxt(f'{self.save_dir}/x_star.txt')
            self.opt_val = np.loadtxt(f'{self.save_dir}/opt_val.txt')
            print("problem loaded:")
            print(f'c {self.c.shape}')
            print(f'd {self.d.shape}')
            print(f'b {self.b.shape}')
            print(f'x_star {self.x_star.shape}')
            print(f'opt_val {self.opt_val}')
        except:
            print("failed to load data in", self.save_dir)
            raise(ValueError)
        
if __name__ == '__main__':
    N = 5
    prob = ElecVehCharg(N)
    prob.gen()
    # prob.load()
