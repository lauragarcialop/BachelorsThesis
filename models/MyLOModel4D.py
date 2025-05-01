import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from models.solvers.ODESolver import Solver

from scipy.stats import wasserstein_distance


def create_time_specs(duration, dt):
    return torch.tensor(np.arange(0, duration + dt, dt), dtype=torch.float64)


def run_LOModel(params, t, dt, Dx=0, Dy=0, method='RK4', seed=None):
    lda, b = params[0].to(torch.float64), params[1].to(torch.float64)
    omega, a = params[2].to(torch.float64), params[3].to(torch.float64)

    lda_hat, omega_hat, a_hat, b_hat = lda, omega, a, b

    # White noise
    if seed is not None:
        torch.manual_seed(seed)
    
    # Define dynamic functions
    def Lambda(r2):
        return lda - b * r2
    
    def Omega(r2):
        return omega + a * r2
    
    def Lambda_hat(r2):
        return lda_hat - b_hat * r2
    
    def Omega_hat(r2):
        return omega_hat + a_hat * r2
    
    def F(X, t, U=None):
        r2 = X[0] ** 2 + X[1] ** 2
        return torch.tensor([Lambda(r2) * X[0] - Omega(r2) * X[1],
                             Omega_hat(r2) * X[0] + Lambda_hat(r2) * X[1]])

    D = torch.tensor([Dx, Dy], dtype=torch.float64)
    solver = Solver(F, t, dt, D=D)
    
    X0 = torch.tensor([1, 0])
    X = solver.solve(X0)

    return dict({'t' : t, 'dt' : dt, 'x' : X[0], 'y' : X[1]})


def find_nullclines(params):
    lda, b = params[0].to(torch.float64), params[1].to(torch.float64)
    omega, a = params[2].to(torch.float64), params[3].to(torch.float64)

    lda_hat, omega_hat, a_hat, b_hat = lda, omega, a, b

    r = torch.tensor(np.arange(0.01, 2.01, 0.01))
    r2 = torch.pow(r, 2)

    alpha = (lda - b * r2) / (omega + a * r2)
    xp = r * torch.sqrt(1 / (1 + torch.pow(alpha, 2)))
    yp = alpha * xp
    xncl_x = torch.cat((torch.flip(-xp, dims=[0]), torch.tensor([0]), xp))
    xncl_y = torch.cat((torch.flip(-yp, dims=[0]), torch.tensor([0]), yp))

    alpha_hat = (lda_hat - b_hat * r2) / (omega_hat + a_hat * r2)
    xp = alpha_hat * r * torch.sqrt(1 / (1 + torch.pow(alpha_hat, 2)))
    yp = -xp / alpha_hat
    yncl_x = torch.cat((torch.flip(-xp, dims=[0]), torch.tensor([0]), xp))
    yncl_y = torch.cat((torch.flip(-yp, dims=[0]), torch.tensor([0]), yp))

    return dict({'xncl_x' : xncl_x, 'xncl_y' : xncl_y, 'yncl_x' : yncl_x, 'yncl_y' : yncl_y})


def plot_traces(sim, save_fig=None):
    plt.figure(figsize=(6, 4))
    plt.plot(sim['t'], sim['x'], label=r'$x$', c='#01016f')
    plt.plot(sim['t'], sim['y'], label=r'$y$', c='#d8031c')
    plt.xlabel(r'Time ($ms$)')
    plt.ylabel('Trace')
    plt.grid(True)
    plt.legend()
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig)
        plt.close()

    return


def plot_trajectory(sim, params=None, save_fig=None):
    plt.figure(figsize=(5, 5))

    if params is not None:
        nullclines = find_nullclines(params)
        plt.plot(nullclines['xncl_x'], nullclines['xncl_y'], label='x Nullcline', c='#01016f')
        plt.plot(nullclines['yncl_x'], nullclines['yncl_y'], label='y Nullcline', c='#d8031c')

    plt.plot(sim['x'], sim['y'], label='Trajectory', c='#5a5a5a')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(f'({params[0]}, {params[1]}, {params[2]}, {params[3]})')
    plt.suptitle(rf'$K_a$={params[0] / params[1]} & $K_f$={params[2] + params[3] * params[0] / params[1]}')
    plt.grid(True)
    plt.legend(loc='upper left')

    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig)
        plt.close()

    return


def _power_spectrum_(t, v):
    timescale, T, L = 2 * np.pi, t[-1], len(v)
    Nfft = int(2 ** np.ceil(np.log2(L)))
    v_centered = v - torch.mean(v)

    V = torch.fft.fft(v_centered, Nfft) / (L / 2)
    Vs = torch.abs(V[:int(Nfft/2)])

    fs = L / (T / timescale)
    freq = fs / 2 * torch.linspace(0, 1, int(Nfft/2))

    power_norm = Vs / Vs.sum()
    
    return dict({'f' : freq, 'Vs' : Vs, 'distr' : power_norm, 'Nfft' : Nfft})


def compute_power_spectrum(sim, trace='x', plot=False):

    ps = _power_spectrum_(sim['t'], sim[trace])
    
    if plot:
        plt.figure(figsize=(6, 4))
        plt.scatter(ps['f'], ps['Vs'], s=20, facecolors='none', edgecolors='#01016f')
        plt.xlabel(r'Frequency ($Hz$)')
        plt.xlim(0, 10)
        plt.ylabel(f'|FFT({trace})|')
        plt.grid(True)
        plt.show()

    return ps


# pdh_type = 1: Only crossing points with positive derivative (for the PWL function)
# pdh_type = 2: Crossing points with positive and negative derivatives (for the PWL funtion)
def _period_density_histogram_(t, v, pdh_type=1, nlevel=100):
    try:
        batch_size, time_steps = v.shape
    except:
        v = v.unsqueeze(0)
        batch_size, time_steps = v.shape

    # Obtener valores mínimos y máximos por batch
    vmax, vmin = v.max(dim=1, keepdim=True)[0], v.min(dim=1, keepdim=True)[0]
    delta_v = vmax - vmin

    # Crear niveles de voltaje vectorizados
    v_level = torch.linspace(0, 1, steps=nlevel, device=v.device).unsqueeze(0) * delta_v + vmin  # [batch, nlevel]

    # Inicializar contadores y cruces
    cntlevel = torch.zeros((batch_size, nlevel), device=v.device)
    tcross = [torch.tensor([]) for _ in range(batch_size)]

    # Encontrar los índices de los niveles de voltaje
    klevel = torch.searchsorted(v_level, v[:, :-1].contiguous()) - 1  # [batch, time_steps-1]
    klevelnext = torch.searchsorted(v_level, v[:, 1:].contiguous()) - 1  # [batch, time_steps-1]

    # Iterar sobre batch
    for i in range(batch_size):
        for jj in range(time_steps - 1):
            if klevelnext[i, jj] > klevel[i, jj]:  # Transición ascendente
                for kk in range(klevel[i, jj] + 1, klevelnext[i, jj] + 1):
                    cntlevel[i, kk] += 1
                    tcross[i] = torch.cat((tcross[i], torch.tensor([(v_level[i, kk] - v[i, jj]) * (t[jj+1] - t[jj]) / (v[i, jj+1] - v[i, jj]) + t[jj]])))
            
            elif klevel[i, jj] == klevelnext[i, jj]:  # Mismo nivel
                if v[i, jj] == v_level[i, klevel[i, jj]]:
                    cntlevel[i, klevel[i, jj]] += 1
                    tcross[i] = torch.cat((tcross[i], torch.tensor([t[i, jj]])))
                elif pdh_type == 2 and v[i, jj+1] == v_level[i, klevel[i, jj]]:
                    cntlevel[i, klevel[i, jj]] += 1
                    tcross[i] = torch.cat((tcross[i], torch.tensor([t[i, jj+1]])))

            elif pdh_type == 2 and klevelnext[i, jj] < klevel[i, jj]:  # Transición descendente
                for kk in range(klevelnext[i, jj] + 1, klevel[i, jj] + 1):
                    cntlevel[i, kk] += 1
                    tcross[i] = torch.cat((tcross[i], torch.tensor([ (v_level[i, kk] - v[i, jj]) * (t[jj+1] - t[jj]) / (v[i, jj+1] - v[i, jj]) + t[jj] ])))
    
    # Convertir listas de tcross a tensores diferenciables
    period_list = [torch.tensor([]) for _ in range(batch_size)]
    for i in range(batch_size):
        if tcross[i].shape[0] > 1:
            period_list[i] = torch.diff(tcross[i])

    return {'v_level': v_level, 'cntlevel': cntlevel, 'tcross': tcross, 'period': period_list, 'nlevel': nlevel}


def __ReLUmod__(x, xhlf, xslp):
    return torch.max(torch.zeros_like(x), (x-xhlf) / xslp * (1 * (x > 0)) - 0.1 * (x <= 0))


def compute_period_density_histogram(sim, trace='x', pdh_type=1, nlevel=100, plot=False):

    pdh = _period_density_histogram_(sim['t'], sim[trace].unsqueeze(0), pdh_type, nlevel)

    if plot:
        ## Plot trace with 
        plt.figure(figsize=(10, 5))

        tmin, tmax = min(sim['t']), max(sim['t'])
        for kk in range(nlevel):
            if kk == 0:
                plt.plot(np.array([tmin, tmax]), np.array(2 * [pdh['v_level'][kk]]), c='#D3D3D3', linestyle='dashed', label='Level bands')
            else:
                plt.plot(np.array([tmin, tmax]), np.array(2 * [pdh['v_level'][kk]]), c='#D3D3D3', linestyle='dashed')
        
        plt.plot(sim['t'], sim[trace], label='trace', c='#01016f')

        for kk in pdh['tcross'].keys():
            plt.scatter(pdh['tcross'][kk], np.array(len(pdh['tcross'][kk]) * [pdh['v_level'][kk]]), s=15, facecolors='none', edgecolors='#d8031c')
        
        plt.xlabel(r'Time ($ms$)')
        plt.xlim(tmin, tmax)
        plt.ylabel(f'{trace}')
        plt.show()

        ## Plot histogram
        binsize, edges_max = 0.1, 5
        edges = torch.arange(0, edges_max, binsize, dtype=torch.float64)
        period_hist, edges = torch.histogram(pdh['period'], bins=edges, density=True)
        period_hist /= torch.sum(period_hist)

        plt.figure(figsize=(6, 4))
        plt.plot(edges[:-1], __ReLUmod__(period_hist, 0, 1), c='#01016f')
        plt.scatter(edges[:-1], __ReLUmod__(period_hist, 0, 1), facecolor = 'none', edgecolors='#01016f')
        plt.xlabel('Horizontal distance (r phase)')
        plt.xlim(0, edges[-1])
        plt.ylabel('Normalized counts')
        plt.ylim(0, 1)
        plt.show()

        ## Plot counts / level
        cntlevel_norm = pdh['cntlevel'][1:-2] / nlevel

        plt.figure(figsize=(6, 4))
        plt.plot(pdh['v_level'][0:len(cntlevel_norm)], cntlevel_norm, c='#01016f')
        plt.scatter(pdh['v_level'][0:len(cntlevel_norm)], cntlevel_norm, facecolor = 'none', edgecolors='#01016f')
        plt.xlabel(f'{trace} levels')
        plt.ylabel('Normalized counts')
        plt.ylim(0, 1)
        plt.show()

    
    return pdh


def compute_sumary_statistics(sim, traces):

    ss = torch.tensor([])
    for trace in traces:
        # # Basic statistics
        # v = sim[trace]
        # v_mean, v_var = torch.tensor([torch.mean(v)]), torch.tensor([torch.var(v)])
        # v_sem = torch.sqrt(v_var/len(v))
        # ss = torch.cat((ss, v_mean, v_var, v_sem))

        # Oscillation statistics 
        v_amp = torch.tensor([torch.max(sim[trace]) - torch.min(sim[trace])])
        # v_amp = torch.tensor([torch.mean(torch.sqrt(torch.pow(sim['x'], 2) + torch.pow(sim['y'], 2)))])
        ss = torch.cat((ss, v_amp))
        ps = _power_spectrum_(sim['t'], sim[trace])
        f0 = torch.tensor([torch.sum(ps['f'] * ps['Vs']) / torch.sum(ps['Vs'])])
        # f0 = torch.tensor([ps['f'][torch.argmax(ps['Vs'])]])
        ss = torch.cat((ss, f0))

        # # Oscillation frequency
        # pdh = _period_density_histogram_(sim['t'], sim[trace])
        # f0 = torch.tensor([1 / torch.mean(pdh['period'])])
        # ss = torch.cat((ss, f0))

        # # Wasserstein Attribute - ISI
        # # ref = v_amp * torch.sin(f0 * sim['t'] + torch.arcsin(sim[trace][0] / v_amp))
        # ref = v_amp * torch.sin(f0 * sim['t'] + torch.pi / 2)
        # w_isi = torch.tensor([isi_distance(sim['t'], sim[trace], sim['t'], ref)])
        # ss = torch.cat((ss, w_isi))

        # # Wasserstein Attribute - Freq
        # w_freq = torch.tensor([frequency_distance(sim['t'], sim[trace], sim['t'], ref)])
        # ss = torch.cat((ss, w_freq))

    return ss


def find_level_set_constants(params):
    lda, b = params[:, 0].to(torch.float64), params[:, 1].to(torch.float64)
    omega, a = params[:, 2].to(torch.float64), params[:, 3].to(torch.float64)

    lda_hat, omega_hat, a_hat, b_hat = lda, omega, a, b

    Ka, Ka_hat = lda / b, lda_hat / b_hat
    Kf, Kf_hat = omega + a * Ka, omega_hat + a_hat * Ka_hat
    
    return dict({'K_a' : Ka, 'K_f' : Kf, 'K_a_hat' : Ka_hat, 'K_f_hat' : Kf_hat})


def plot_reconstructions(obs_sim, rec_sim, obs_params, rec_params, trace):
    ls = find_level_set_constants(rec_params)

    fig = plt.figure()
    fig.suptitle(f"$\\lambda$={rec_params[0, 0]}, b={rec_params[0, 1]} ($K_a$ = {ls['K_a'].item()}, $K_f$ = {ls['K_f'].item()})")
    gs = mpl.gridspec.GridSpec(2, 3, wspace=0.25, hspace=0.25)

    # Trace reconstruction plot
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.set_title(f'Trace {trace}')
    ax0.plot(obs_sim['t'], obs_sim[trace], label='Observation', c='#01016f')
    ax0.plot(rec_sim['t'], rec_sim[trace], label='Estimation', c='#d8031c', ls='dashed')
    ax0.set_xlabel(r'Time ($ms$)')
    ax0.set_ylabel('Trace')
    ax0.grid(True)
    ax0.legend()

    # Trajectory and isoclines
    obs_nullclines = find_nullclines(obs_params.squeeze())
    rec_nullclines = find_nullclines(rec_params.squeeze())

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.set_title('Trajectory & Isoclines')
    ax1.plot(obs_nullclines['xncl_x'], obs_nullclines['xncl_y'], label='Observation Nullcline', c='#A9A9A9')
    ax1.plot(obs_nullclines['yncl_x'], obs_nullclines['yncl_y'], c='#A9A9A9')
    ax1.plot(rec_nullclines['xncl_x'], rec_nullclines['xncl_y'], label='Estimation Nullcline', ls='dashed', c='#36454F')
    ax1.plot(rec_nullclines['yncl_x'], rec_nullclines['yncl_y'], ls='dashed', c='#36454F')
    ax1.plot(obs_sim['x'], obs_sim['y'], label='Observation', c='#01016f')
    ax1.plot(rec_sim['x'], rec_sim['y'], label='Estimation', c='#d8031c', ls='dashed')
    ax1.set_xlabel(r'$x$')
    ax1.set_xlim(1.2 * min(torch.min(obs_sim['x']), torch.min(rec_sim['x'])), 1.2 * max(torch.max(obs_sim['x']), torch.max(rec_sim['x'])))
    ax1.set_ylabel(r'$y$')
    ax1.set_ylim(1.2 * min(torch.min(obs_sim['y']), torch.min(rec_sim['y'])), 1.2 * max(torch.max(obs_sim['y']), torch.max(rec_sim['y'])))
    ax1.grid(True)
    ax1.legend()
    
    # FFT
    obs_ps = _power_spectrum_(obs_sim['t'], obs_sim[trace])
    rec_ps = _power_spectrum_(rec_sim['t'], rec_sim[trace])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(obs_ps['f'], obs_ps['Vs'], s=20, facecolors='none', edgecolors='#01016f', label='Observation')
    ax2.scatter(rec_ps['f'], rec_ps['Vs'], s=20, facecolors='none', edgecolors='#d8031c', label='Estimation')
    ax2.set_xlabel(r'Frequency ($Hz$)')
    ax2.set_xlim(0, 10)
    ax2.set_ylabel(f'|FFT({trace})|')
    ax2.grid(True)
    ax2.legend()

    ## Period density histograms :)
    obs_pdh = _period_density_histogram_(obs_sim['t'], obs_sim[trace].unsqueeze(0))
    rec_pdh = _period_density_histogram_(rec_sim['t'], rec_sim[trace].unsqueeze(0))

    binsize, edges_max = 0.1, 5
    edges = torch.arange(0, edges_max, binsize, dtype=torch.float64)

    # Normalized counts per horizontal distance
    obs_period_hist, obs_edges = torch.histogram(torch.stack(obs_pdh['period']), bins=edges, density=True)
    obs_period_hist /= torch.sum(obs_period_hist)

    rec_period_hist, rec_edges = torch.histogram(torch.stack(rec_pdh['period']), bins=edges, density=True)
    rec_period_hist /= torch.sum(rec_period_hist)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(obs_edges[:-1], __ReLUmod__(obs_period_hist, 0, 1), label='Observation', c='#01016f')
    ax3.scatter(obs_edges[:-1], __ReLUmod__(obs_period_hist, 0, 1), facecolor = 'none', edgecolors='#01016f')
    ax3.plot(rec_edges[:-1], __ReLUmod__(rec_period_hist, 0, 1), label='Estimation', c='#d8031c')
    ax3.scatter(rec_edges[:-1], __ReLUmod__(rec_period_hist, 0, 1), facecolor = 'none', edgecolors='#d8031c')
    ax3.set_xlabel('Horizontal distance (r phase)')
    ax3.set_xlim(0, max(obs_edges[-1], rec_edges[-1]))
    ax3.set_ylabel('Normalized counts')
    ax3.set_ylim(0, 1)
    ax3.grid(True)
    ax3.legend()

    # Normalized counts per level
    obs_cntlevel_norm = obs_pdh['cntlevel'].squeeze()[1:-2] / obs_pdh['nlevel']
    rec_cntlevel_norm = rec_pdh['cntlevel'].squeeze()[1:-2] / rec_pdh['nlevel']

    obs_vlevel = obs_pdh['v_level'].squeeze()[0:len(obs_cntlevel_norm)]
    rec_vlevel = rec_pdh['v_level'].squeeze()[0:len(rec_cntlevel_norm)]

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(obs_vlevel, obs_cntlevel_norm, label='Observation', c='#01016f')
    ax4.scatter(obs_vlevel, obs_cntlevel_norm, facecolor = 'none', edgecolors='#01016f')
    ax4.plot(rec_vlevel, rec_cntlevel_norm, label='Estimation', c='#d8031c')
    ax4.scatter(rec_vlevel, rec_cntlevel_norm, facecolor = 'none', edgecolors='#d8031c')
    ax4.set_xlabel(f'{trace} levels')
    ax4.set_ylabel('Normalized counts')
    ax4.set_ylim(0, 1)
    ax4.grid(True)
    ax4.legend()
    
    fig.set_figheight(8)
    fig.set_figwidth(12)

    return 


def differentiable_histogram(x, bins):
    batch_size, num_bins = x.shape[0], bins.shape[0]
    histograms = torch.zeros_like((batch_size, num_bins), dtype=torch.float32)

    bin_indices = torch.bucketize(x, bins, right=True)
    histograms = torch.scatter_add(histograms, dim=0, index=bin_indices, src=torch.ones_like(bin_indices, dtype=torch.float32))
    return histograms


def frequency_distance(t1, obs1, t2, obs2):
    ps1 = _power_spectrum_(t1, obs1)
    ps2 = _power_spectrum_(t2, obs2)

    power1_norm = ps1['Vs'] / ps1['Vs'].sum()
    power2_norm = ps2['Vs'] / ps2['Vs'].sum()

    return wasserstein_distance(power1_norm, power2_norm)


def isi_distance(t1, obs1, t2, obs2):
    pdh1 = _period_density_histogram_(t1, obs1)
    pdh2 = _period_density_histogram_(t2, obs2)

    binsize, edges_max = 0.1, 5
    edges = torch.arange(0, edges_max, binsize, dtype=torch.float64)
    period_hist1, _ = torch.histogram(torch.stack(pdh1['period']), bins=edges, density=True)
    period_hist1 /= torch.sum(period_hist1)
    period_hist2, _ = torch.histogram(torch.stack(pdh2['period']), bins=edges, density=True)
    period_hist2 /= torch.sum(period_hist2)

    return wasserstein_distance(period_hist1, period_hist2)