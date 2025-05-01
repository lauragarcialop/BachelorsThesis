import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from models.solvers.ODESolver import Solver


def create_time_specs(duration, dt):
    return torch.tensor(np.arange(0, duration + dt, dt), dtype=torch.float64)


def run_IFModel(params, t, dt, D=0, seed=None):
    gL = params[0]
    I = params[1]

    C, EL = 1, -60
    V_th, V_res = -50, -60

    t_on, t_off = dt, t[-1]-dt
    H = torch.heaviside(t-t_on, torch.ones_like(t)) * torch.heaviside(t_off-t, torch.ones_like(t))
    I_app = I * H

    # White noise
    if seed is not None:
        torch.manual_seed(seed)
    
    V = torch.zeros_like(t)
    V[0] = EL

    def F(X, t, U):
        return (- gL * (X - EL) + U) / C
    
    eta = torch.randn(len(t), dtype=torch.float64)
    
    spkcnt, tspk = 0, []

    for ii in range(len(t)-1):
        K1 = F(V[ii], t[ii], I_app[ii])
        K2 = F(V[ii] + K1*dt + torch.sqrt(2 * D * dt) * eta[ii], t[ii], I_app[ii])
        V[ii+1] = V[ii] + 0.5 * (K1 + K2) * dt + torch.sqrt(2 * D * dt) * eta[ii]

        if V[ii+1] >= V_th:
            V[ii], V[ii+1] = 60, V_res
            tspk += [t[ii]]
            spkcnt += 1

    return dict({'t' : t, 'dt' : dt, 'V' : V, 'I_inj' : I_app, 'Spike Count' : spkcnt, 'Spike Times' : torch.tensor(tspk)})


def plot_results(simulations):
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(3, 1, wspace=0.25, hspace=0.25)

    ax1 = fig.add_subplot(gs[0:2])
    ax1.plot(simulations["t"], simulations["V"])
    ax1.set(ylabel='Voltage ($mV$)')
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(simulations["t"], simulations["I_inj"])
    ax2.set(ylabel='$I_{inj}$ (nA)', xlabel = 'Time (ms)')
    ax2.grid(True)
    return


def plot_reconstructions(obs_sim, rec_sim, obs_params, rec_params):
    fig = plt.figure()
    # fig.suptitle(f"lda={rec_params[0]}, b={rec_params[1]} (Ka = {ls['K_a']}, Kf = {ls['K_f']})")
    gs = mpl.gridspec.GridSpec(3, 1, wspace=0.25, hspace=0.25)
    fig.subplots_adjust(hspace=0.5)

    # Trace reconstruction plot
    ax0 = fig.add_subplot(gs[0:2])
    # ax0.set_title(f'Trace {trace}')
    ax0.plot(obs_sim['t'], obs_sim['V'], label='Observation', c='#01016f')
    ax0.plot(rec_sim['t'], rec_sim['V'], label='Estimation', c='#d8031c', ls='dashed')
    ax0.set_ylabel('Voltage')
    ax0.grid(True)
    ax0.legend()

    ax1 = fig.add_subplot(gs[2], sharex=ax0)
    ax1.plot(obs_sim['t'], obs_sim["I_inj"], label='Observation', c='#01016f')
    ax1.plot(rec_sim['t'], rec_sim["I_inj"], label='Estimation', c='#d8031c', ls='dashed')
    ax1.set_xlabel('Time ($ms$)')
    ax1.set_ylabel('$I_{inj}$ ($nA$)')
    ax1.grid(True)
    ax1.legend()

    return 