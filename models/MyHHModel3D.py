import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as spstats

from models.solvers.ODESolver import Solver

"""
This file implements the Hodgkin-Huxley method with the RK2 (Modified Euler) method to solve the ODE.
The parameters of the model are the conductances of the ion channels and the applied current.
"""


def syn_current(duration=100, t_on=10, t_off=90, dt=0.01, I0 = 0, I1=50, n_pulses=1, delay=0, seed=None):
    """Returns the time and current synchronization data

    Parameters
    ----------
        duration : float (ms) - Duration of the signal
        dt : float (ms) - Timestep
        t_on : float (ms) - When the impulse starts
        t_off : float (ms) - When the impulse ends
        I0 : float (muA) - Base intensity
        I1 : float (muA) - Pulse intensity
        n_pulses : int - Number of pulses
        delay : float (ms) - Delay between pulse onsets
        seed : int
    """

    A_soma = 1e-3  # cm2

    t = torch.linspace(0, duration, int(np.round(duration/dt)))
    I_inj = I0 * torch.ones_like(t) / A_soma                                                               # muA/cm2
    
    for k in range(n_pulses):
        I_inj[int(np.round((t_on + k*delay)/dt)):int(np.round((t_off + k*delay)/dt))] = I1 / A_soma     # muA/cm2

    return I_inj, dt, t, A_soma


## Functions for the gating variables
# u : potential difference in the membrane (V)

# Sodium Activation
def alpha_m(u):
    return 0.1 * (-u - 45) / (np.exp((-u - 45) / 10) - 1)


def beta_m(u):
    return 4 * np.exp((-u - 70) / 18)


def tau_m(u):
    return 1 / (alpha_m(u) + beta_m(u))


def m_inf(u):
    return alpha_m(u) * tau_m(u)


# Sodium Inactivation
def alpha_h(u):
    return 7e-2 * np.exp((-u - 70) / 20)


def beta_h(u):
    return 1 / (1 + np.exp((-u - 40) / 10))


def tau_h(u):
    return 1 / (alpha_h(u) + beta_h(u))


def h_inf(u):
    return alpha_h(u) * tau_h(u)


# Potassium activation
def alpha_n(u):
    return 0.01 * (-u - 60) / (np.exp((-u - 60) / 10) - 1)


def beta_n(u):
    return 0.125 * np.exp((-u - 70) / 80)


def tau_n(u):
    return 1 / (alpha_n(u) + beta_n(u))


def n_inf(u):
    return alpha_n(u) * tau_n(u)


def plot_rate_constants(uvec):
    fig, axs = plt.subplots(1, 3, figsize=(8, 2))

    axs[0].plot(uvec, alpha_m(uvec), label='$alpha$')
    axs[0].plot(uvec, beta_m(uvec), label='$beta$')
    axs[0].set(title='m', ylabel='Rate Constants ($ms^{-1}$)', xlabel='Voltage ($mV$)')
    axs[0].legend()

    axs[1].plot(uvec, alpha_h(uvec), label='$alpha$')
    axs[1].plot(uvec, beta_h(uvec), label='$beta$')
    axs[1].set(title='h', xlabel='Voltage ($mV$)')

    axs[2].plot(uvec, alpha_n(uvec), label='$alpha$')
    axs[2].plot(uvec, beta_n(uvec), label='$beta$')
    axs[2].set(title='n', xlabel='Voltage ($mV$)')

    return


def plot_time_and_ss(uvec):
    fig, axs = plt.subplots(1, 2, figsize=(6, 2))

    axs[0].plot(uvec, tau_m(uvec), label = 'm')
    axs[0].plot(uvec, tau_h(uvec), label = 'h')
    axs[0].plot(uvec, tau_n(uvec), label = 'n')
    axs[0].set(xlabel='Voltage ($mV$)', ylabel='$tau_x(u)$ ($ms$)')
    axs[0].legend()

    axs[1].plot(uvec, m_inf(uvec))
    axs[1].plot(uvec, h_inf(uvec))
    axs[1].plot(uvec, n_inf(uvec))
    axs[1].set(xlabel='Voltage ($mV$)', ylabel='$x_0(u)$')

    return


def run_HHModel(params, I, t, dt, A_soma, initial_conditions = False, V0=0, m0=0, h0=0, n0=0, nois_fact=1e-6, method='RK4', seed=None):

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    # "trainable" parameters
    g_leak = params[0]   
    g_leak = g_leak.to(torch.float64)   # mS/cm2
    g_Na = params[1]
    g_Na = g_Na.to(torch.float64)       # mS/cm2
    g_K = params[2]
    g_K = g_K.to(torch.float64)         # mS/cm2

    if initial_conditions:
        X0 = torch.tensor([V0, m0, h0, n0])
    else:
        X0 = torch.tensor([V0, m_inf(V0), h_inf(V0), n_inf(V0)])

    def F(X, t, U=None):
        # fixed parameters
        E_leak = - 60    # mV
        E_Na = 45        # mV
        E_K = - 82       # mV
        C = 1            # muF/cm2

        V, m, h, n = X[0], X[1], X[2], X[3]

        dV = (g_leak * (E_leak - V) 
                + g_Na * (m ** 3) * h * (E_Na - V) 
                + g_K * (n ** 4) * (E_K - V) 
                + U) / C
        dm = (m_inf(V) - m) / tau_m(V)
        dh = (h_inf(V) - h) / tau_h(V)
        dn = (n_inf(V) - n) / tau_n(V)

        return torch.tensor([dV, dm, dh, dn])


    D = torch.tensor([nois_fact, 0, 0, 0])
    solver = Solver(F, t, dt, method=method, D=D)

    X = solver.solve(X0, I)

    return dict({"V" : X[0], "m" : X[1], "h" : X[2], "n" : X[3], "I_inj" : I, "t" : t, "A" : A_soma})


def plot_results(simulations):
    fig, axs = plt.subplots(3)
    axs[0].plot(simulations["t"], simulations["V"])
    axs[0].set(ylabel='Voltage ($mV$)')

    axs[1].plot(simulations["t"], simulations["m"], label="m")
    axs[1].plot(simulations["t"], simulations["h"], label="h")
    axs[1].plot(simulations["t"], simulations["n"], label="n")
    axs[1].set(ylabel='Gating variables')
    axs[1].legend()

    axs[2].plot(simulations["t"], 1e-3 * simulations["A"] * simulations["I_inj"])
    axs[2].set(ylabel='$I_{inj}$ (nA)', xlabel = 'Time (ms)')
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
    ax1.plot(obs_sim['t'], 1e-3 * obs_sim["A"] * obs_sim["I_inj"], label='Observation', c='#01016f')
    ax1.plot(rec_sim['t'], 1e-3 * rec_sim["A"] * rec_sim["I_inj"], label='Estimation', c='#d8031c', ls='dashed')
    ax1.set_xlabel('Time ($ms$)')
    ax1.set_ylabel('$I_{inj}$ ($nA$)')
    ax1.grid(True)
    ax1.legend()

    return 


def reformat_data(x, I, t):
    return dict({"V" : x.reshape(-1, 1), "I_inj" : np.array(I).reshape(-1, 1), "t" : np.array(t),
                 "A" : 1e-3})


def _binarize_voltage_(Vsig):
    # binarize voltage signal
    V_thres = np.zeros_like(Vsig)
    V_thres[Vsig >= -40] = 1

    V_diff = np.array([[0]] + list(V_thres[1:len(Vsig), :] - V_thres[0:len(Vsig)-1, :]))
    
    V_bin = np.zeros_like(Vsig)
    V_bin[V_diff < 0] = 1

    return V_bin


def compute_attributes(x, att_list):
    # take important data
    V, t = x["V"], x["t"]
    dt = t[1] - t[0] # ms

    # binarize voltage signal
    V_bin = _binarize_voltage_(V)
    spike_times = dt * np.where(V_bin == 1)[0]
    # spike_count = np.array([np.sum((spike_times >= 0.9 * 10) & (spike_times <= 1.1 * 90), axis=0)])

    # compute inter spike interval
    inter_spike_intervals = np.diff(spike_times)

    # compute mean inter spike interval
    if np.any(inter_spike_intervals) == False:
        mean_isi = np.array([float('inf')])
        firing_rate = np.array([0])
    else:
        mean_isi = np.array([np.mean(inter_spike_intervals, axis=0)])
        firing_rate = np.reciprocal(mean_isi)

    # entropy of the inter spike interval distribution
    isi_hist, _ = np.histogram(inter_spike_intervals, bins=20)
    isi_hist = isi_hist[np.nonzero(isi_hist)]
    isi_prob = isi_hist / isi_hist.sum()
    isi_entropy = np.array([np.sum(np.multiply(isi_prob, np.log(isi_prob)), axis=0)])

    # compute voltage statistics
    rest_pot = np.array([np.mean(V[t < 10])])                     # resting potential before the input current
    mean_pot = np.mean(V[(t >= 0.9 * 10) & (t <= 1.1 * 90)], axis=0)          # mean potential during the input current
    
    # compute voltage moments during the input current
    rest_pot_std = np.array([np.std(V[(t >= 0.9 * 10) & (t <= 1.1 * 10)])])
    # std_pot = np.std(V[(t >= 0.9 * 10) & (t <= 1.1 * 90)], axis=0)            # standard deviation of the potential during the input current

    # moments
    n_mom = 3
    std_pw = np.power(np.std(V[(t >= 0.9 * 10) & (t <= 1.1 * 90)]), np.linspace(3, n_mom, n_mom-2))
    std_pw = np.concatenate((np.ones(1), std_pw))
    moments = spstats.moment(V[(t >= 0.9 * 10) & (t <= 1.1 * 90)], np.linspace(2, n_mom, n_mom-1)).reshape(1, -1)
    moments = np.squeeze(moments) / std_pw

    # first spike time
    if np.any(spike_times):
        first_spike_time = np.array([spike_times[0]])
    else:
        first_spike_time = np.array([0])

    attributes = np.array([])
    if "FR" in att_list:
        attributes = np.concatenate((attributes, firing_rate))
    
    if "AvgISI" in att_list:
        attributes = np.concatenate((attributes, mean_isi))
    
    if "AvgV" in att_list:
        attributes = np.concatenate((attributes, mean_pot))
    
    if "RestV" in att_list:
        attributes = np.concatenate((attributes, rest_pot))

    if "StdRestV" in att_list:
        attributes = np.concatenate((attributes, rest_pot_std))

    if "Moments" in att_list:
        attributes = np.concatenate((attributes, moments))
    
    if "ISIEntr" in att_list:
        attributes = np.concatenate((attributes, isi_entropy))

    if "SpT1" in att_list:
        attributes = np.concatenate((attributes, first_spike_time))

    return torch.as_tensor(attributes) 