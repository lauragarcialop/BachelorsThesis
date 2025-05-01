import torch
import torch.nn as nn
import numpy as np

from models.MyLOModel2D import create_time_specs
from models.MyLOModel2D import _period_density_histogram_, _power_spectrum_, frequency_distance



class ISIWassersteinLoss(nn.Module):
    def __init__(self, Tmax=2 * np.pi * 10, dt=0.1):
        super(ISIWassersteinLoss, self).__init__()

        self.t = create_time_specs(Tmax, dt)


    def forward(self, x_rec, x_true):
        pdh_rec = _period_density_histogram_(self.t, x_rec)
        pdh_true = _period_density_histogram_(self.t, x_true)

        binsize, edges_max = 0.1, 5
        edges = torch.arange(0, edges_max, binsize, dtype=torch.float64)

        period_hist_rec = differentiable_histogram(pdh_rec['period'], edges, add_padding=True)
        period_hist_true = differentiable_histogram(pdh_true['period'], edges, add_padding=True)
        
        # Compute Wasserstain Distance in such a way that pytorch can automatically compute the backward propagation
        return wasserstein_distance(period_hist_rec, period_hist_true)
    


class FreqWassersteinLoss(nn.Module):
    def __init__(self, Tmax=2 * np.pi * 10, dt=0.1):
        super(FreqWassersteinLoss, self).__init__()

        self.t = create_time_specs(Tmax, dt)


    def forward(self, x_rec, x_true):
        ps1 = _power_spectrum_(self.t, x_rec)
        ps2 = _power_spectrum_(self.t, x_true)

        power1_norm = ps1['Vs'] / ps1['Vs'].sum()
        power2_norm = ps2['Vs'] / ps2['Vs'].sum()

        return wasserstein_distance(power1_norm, power2_norm)
    


def padded(v, M):
    return torch.tensor(list(v) + ([float('nan')] * (M - len(v))))


def differentiable_histogram(v, bins, add_padding=False):
    print(v.shape, bins.shape)
    try:
        v = v.unsqueeze(0)
        batch_size = v.shape[0]
    except:
        batch_size = len(v)
    histograms = torch.zeros((batch_size, bins.shape[0]), device=v[0].device, dtype=torch.float32, requires_grad=True)

    if add_padding:
        max_length = max([len(vv) for vv in v])
        v_padding = []

        for vv in v:
            v_padding.append(padded(vv, max_length))
        v_padding = torch.stack(v_padding)

        bin_indices = torch.bucketize(v_padding, bins, right=True)
    else:
        bin_indices = torch.bucketize(v, bins, right=True)
    print(histograms.shape, bin_indices.shape)
    histograms = torch.scatter_add(histograms, dim=1, index=bin_indices, src=torch.ones_like(bin_indices, dtype=torch.float32))

    histograms = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-8)
    
    return histograms
    

def wasserstein_distance(hist1, hist2):
    cumulative1 = torch.cumsum(hist1, dim=0)
    cumulative2 = torch.cumsum(hist2, dim=0)
    return torch.sum(torch.abs(cumulative1 - cumulative2))