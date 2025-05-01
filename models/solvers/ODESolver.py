from models.solvers.RK2 import rk2_solver
from models.solvers.RK4 import rk4_solver


class Solver:
    def __init__(self, f, t, dt, method='RK4', D=None, eta=None):
        self.Func = f
        self.t = t
        self.dt = dt
        
        self.D = D
        self.eta = eta

        if method == 'RK2':
            self.solver = rk2_solver
        else:
            self.solver = rk4_solver
        

    def solve(self, X0, U=None):
        return self.solver(self.Func, X0, self.t, U, self.dt, self.D)
