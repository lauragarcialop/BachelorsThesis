import torch
import numpy as np

from sbi import analysis as analysis

# sbi
from sbi import utils as utils
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from models.MyLOModel2D import create_time_specs, run_LOModel
from functions.DS_Functions import write_dataset


Tmax, dt = 2 * np.pi * 50, 0.01
Dx = 0.01

def simulate_LO_model(params):
    t = create_time_specs(Tmax, dt)
    simulation = run_LOModel(params.squeeze(), t, dt, Dx=Dx)
    return simulation["x"]

# with this prior, the params for the other model are also included
prior_min, prior_max = [0.5] * 2, [1.5] * 2

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

# Check prior, simulator, consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulate_LO_model = process_simulator(simulate_LO_model, prior, prior_returns_numpy)
check_sbi_inputs(simulate_LO_model, prior)


train_simulations, valid_simulations, test_simulations = 4000, 500, 500
MODEL_PATH = '/Users/lauragarcialopez/Documents/uni/TFG/06.TFG/codes/data/'

# Generate training dataset
theta, x = simulate_for_sbi(simulate_LO_model, proposal=prior, num_simulations=train_simulations, num_workers=4)
write_dataset(f"{MODEL_PATH}train_dataset__Dx={str(Dx)}_Tmax={Tmax}", theta, x)

# Generate validation dataset
theta, x = simulate_for_sbi(simulate_LO_model, proposal=prior, num_simulations=valid_simulations, num_workers=4)
write_dataset(f"{MODEL_PATH}valid_dataset__Dx={str(Dx)}_Tmax={Tmax}", theta, x)

# Generate test dataset
theta, x = simulate_for_sbi(simulate_LO_model, proposal=prior, num_simulations=test_simulations, num_workers=4)
write_dataset(f"{MODEL_PATH}test_dataset__Dx={str(Dx)}_Tmax={Tmax}", theta, x)