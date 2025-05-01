import torch
import torch.nn as nn
import numpy as np

from models.MyLOModel2D import create_time_specs

from losses.Cosine_Loss import CosineLoss
from losses.Wasserstein_Loss import FreqWassersteinLoss, ISIWassersteinLoss
from losses.DMSE_Loss import DMSELoss

# params = {'cos_loss' : {'weight' : 1},
#           'mse_loss' : {'weight' : 1},
#           'wdist_loss' : {'weight' : 1},
#           'dmse_loss' : {'weight' : 1},
#          }


def get_loss(loss_name, params=None, Tmax=2 * np.pi * 10, dt=0.1):
    if loss_name == 'cos_loss':
        return CosineLoss(Tmax=Tmax, dt=dt)
    elif loss_name == 'mse_loss':
        return nn.MSELoss()
    elif loss_name == 'isiwdist_loss':
        return ISIWassersteinLoss(Tmax=Tmax, dt=dt)
    elif loss_name == 'freqwdist_loss':
        return FreqWassersteinLoss(Tmax=Tmax, dt=dt)
    elif loss_name == 'dmse_loss':
        return DMSELoss(Tmax=Tmax, dt=dt)
            
    

class CombinedLoss(nn.Module):
    def __init__(self, params, Tmax=2 * np.pi * 10, dt=0.1, lda=0.1):
        super(CombinedLoss, self).__init__()

        self.t = create_time_specs(Tmax, dt)

        self.weights = []
        self.losses = []
        
        for loss_name in params.keys():
            loss_params = params[loss_name]
            if loss_params['weight'] != 0:
                self.losses.append(get_loss(loss_name, loss_params, Tmax, dt))
                self.weights.append(loss_params['weight'])


    def forward(self, x_rec, x_true):

        total_loss = 0
        for i, loss in enumerate(self.losses):
            this_loss = loss(x_rec, x_true)

            total_loss += (self.weights[i] * this_loss)

        return total_loss