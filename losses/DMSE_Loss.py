import torch
import torch.nn as nn
import numpy as np

from models.MyLOModel2D import create_time_specs
import torch.nn.functional as F


class DMSELoss(nn.Module):
    def __init__(self, Tmax=2 * np.pi * 10, dt=0.1):
        super(DMSELoss, self).__init__()

        self.t = create_time_specs(Tmax, dt)


    def forward(self, x_rec, x_true):
        diff_t = torch.diff(self.t)
        diff_rec = torch.diff(x_rec, dim=1)
        diff_true = torch.diff(x_true, dim=1)

        return torch.mean(torch.abs((diff_rec - diff_true) / diff_t) ** 2)