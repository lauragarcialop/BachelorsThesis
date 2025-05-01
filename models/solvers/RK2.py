import torch

def rk2_solver(Func, X0, t, U=None, dt=0.01, D=None):
    X = torch.zeros((X0.shape[0], len(t)))
    X[:, 0] = X0

    eta = torch.randn((X0.shape[0], len(t)), dtype=torch.float64)

    if U is None:
        U = torch.zeros_like(t, dtype=torch.float64)

    for ii in range(len(t)-1):
        K1 = Func(X[:, ii], t[ii], U[ii])
        K2 = Func(X[:, ii] + K1 * dt + torch.sqrt(2 * D * dt) * eta[:, ii], t[ii] + dt, U[ii])
        X[:, ii+1] = X[:, ii] + 0.5 * (K1 + K2) * dt + torch.sqrt(2 * D * dt) * eta[:, ii]

    return X