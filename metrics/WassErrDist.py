# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

from models.MyLOModel2D import create_time_specs, run_LOModel
from scipy.stats import levene, wasserstein_distance


def plot_errors(t, err1, err2, params1, params2):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    
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
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
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


def WassErrDist(sim1, sim2, params1, params2, trace='x', reference=None, plot=False):
    if reference is None:
        A, f = 1, 2
        reference = A * torch.sin(f * sim1['t'] + 0.5 * torch.pi)

    err1 = sim1[trace] - reference
    err2 = sim2[trace] - reference

    if plot:
        plot_histograms(err1, err2)
        plot_errors(sim1['t'], err1, err2, params1, params2)

    return wasserstein_distance(err1, err2)