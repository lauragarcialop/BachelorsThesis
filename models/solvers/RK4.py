import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt


def OU1(t, dt, D, theta=0.01, mu=0.0, sigma=0.3):
    eta = torch.zeros((D.shape[0], len(t)), dtype=torch.float64)

    for ii in range(len(t)-1):
        dW = torch.randn(D.shape[0]) # Wiener process increment
        eta[:, ii+1] = eta[:, ii] + theta * (mu - eta[:, ii]) * dt + sigma * dW

    # plt.plot(t.numpy(), eta[0].numpy())
    
    return eta


def OU2(t, dt, D, tau=100, mu=0, sigma=0.3):
    eta = torch.zeros((D.shape[0], len(t)), dtype=torch.float64)

    for ii in range(len(t)-1):
        eps = torch.randn(D.shape[0])
        eta[:, ii+1] = eta[:, ii] * (1 - dt/tau) + mu / tau * dt + sigma * np.sqrt(2 * dt / tau) * eps
    
    # plt.plot(t.numpy(), eta[0].numpy())

    return eta


def OU_noise(t, dt, D, type=1):
    if type == 1:
        return OU1(t, dt, D)
    else: # if type == 2:
        return OU2(t, dt, D)


def rk4_solver(Func, X0, t, U=None, dt=0.01, D=None):
    X = torch.zeros((X0.shape[0], len(t)))
    X[:, 0] = X0

    noise = OU_noise(t, dt, D, type=2)

    if U is None:
        U = torch.zeros_like(t, dtype=torch.float64)

    for ii in range(len(t)-1):
        K1 = Func(X[:, ii], t[ii], U[ii])
        K2 = Func(X[:, ii] + 0.5 * K1 * dt + torch.sqrt(2 * D * dt) * noise[:, ii], t[ii] + 0.5 * dt, U[ii])
        K3 = Func(X[:, ii] + 0.5 * K2 * dt + torch.sqrt(2 * D * dt) * noise[:, ii], t[ii] + 0.5 * dt, U[ii])
        K4 = Func(X[:, ii] + K3 * dt + torch.sqrt(2 * D * dt) * noise[:, ii], t[ii] + dt, U[ii])
        
        X[:, ii+1] = X[:, ii] + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6 + torch.sqrt(2 * D * dt) * noise[:, ii]

    return X
