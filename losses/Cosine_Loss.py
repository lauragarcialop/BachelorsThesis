import torch
import torch.nn as nn
import numpy as np

from models.MyLOModel2D import create_time_specs
import torch.nn.functional as F


class CosineLoss(nn.Module):
    def __init__(self, Tmax=2 * np.pi * 10, dt=0.1):
        super(CosineLoss, self).__init__()

        self.t = create_time_specs(Tmax, dt)


    def forward(self, x_rec, x_true):
        x_true_norm = F.normalize(x_true, p=2, dim=1)
        x_rec_norm = F.normalize(x_rec, p=2, dim=1)

        cos_sim = torch.sum(x_true_norm * x_rec_norm, dim=1)
        
        return 1 - torch.mean(cos_sim)