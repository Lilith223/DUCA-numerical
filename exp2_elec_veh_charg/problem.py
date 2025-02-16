import numpy as np
import cvxpy as cp
import os

class ElecVehCharg:
    '''
    Plug-in electric vehicles charging problem in 
    [Globally-Constrained Decentralized Optimization with Variable Coupling]
    
    problem:
        minimize \sum_i c_i x_i
        subject to 
            0 <= x_i <= 1
            \sum_i -d_i \log(1 + x_i) <= -b
        
    '''
    
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
        

    def gen(self):
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
        obj = (self.c.T @ var_x)
        coupling_g = (self.d.T @ cp.log(var_x+1))
        cons = [0. <= var_x, var_x <= 1, self.b[0] - coupling_g <= 0.] 

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        
        assert(self.prob.is_dcp())
        self.prob.solve(solver='ECOS', reltol=1e-9)
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
