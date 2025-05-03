# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

from models.MyLOModel2D import find_level_set_constants
from scipy.stats import levene, wasserstein_distance


def plot_errors(t, err1, err2, params1, params2):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
    
    axes[0].plot(t, err1, label=f"$\\lambda={params1[0]}$")
    axes[0].plot(t, err2, label=f"$\\lambda={params2[0]}$", alpha=0.5)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Error")
    axes[0].set_title("Observation - Theoretical limit cycle")
    axes[0].legend()

    axes[1].plot(t, err1 - err2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Observation 1 - Observation 2")
    plt.show()

    return


def plot_histograms(err1, err2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].hist(err1, bins=50, color='blue', label='Obs 1', alpha=0.7, density=True)
    axes[0].hist(err2, bins=50, color='red', label='Obs 2', alpha=0.5, density=True)
    axes[0].legend()
    axes[0].set_title("Observation - Theoretical limit cycle")

    axes[1].hist(err1 - err2, bins=50, color='green', alpha=0.7, density=True)
    axes[1].set_title("Observation 1 - Observation 2")
    plt.show()

    return


def compute_variance(data):
    return torch.var(data)


def variance_test(err1, err2):
    stats, p = levene(err1, err2)
    return stats, p


def WassDist(sim1, sim2, trace='x'):
    return wasserstein_distance(sim1[trace], sim2[trace])


def WassErrDist(sim1, sim2, params1, params2, trace='x', plot=False):
    ls1, ls2 = find_level_set_constants(params1.unsqueeze(0)), find_level_set_constants(params2.unsqueeze(0))
    ref1, ref2 = torch.sqrt(ls1['K_a']) * torch.cos(ls1['K_f'] * sim1['t']), torch.sqrt(ls2['K_a']) * torch.cos(ls2['K_f'] * sim2['t'])

    err1 = sim1[trace] - ref1
    err2 = sim2[trace] - ref2

    _, p = variance_test(err1, err2)
    print(f"The p-value for the Levene test is {p}. \n The Levene test says the variances of the error distributions {'the variances are different' if p < 0.05 else 'the variances cannot be considered non similar'}.")

    if plot:
        plot_histograms(err1, err2)
        plot_errors(sim1['t'], err1, err2, params1.squeeze(), params2.squeeze())

    return wasserstein_distance(err1, err2)