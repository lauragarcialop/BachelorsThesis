import torch
import torch.optim as optim
import numpy as np

import sys, os, json

import matplotlib as mpl

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


# command line in shell
# python3 DimRedAE.py 0 4 0 0 0.5 0.5
model_num = sys.argv[1]
embedding_dim = int(sys.argv[2])
alpha, beta = float(sys.argv[3]), float(sys.argv[4])
p, n = float(sys.argv[5]), float(sys.argv[6])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.MyLOModel2D import create_time_specs
Tmax, dt, Dx, Dy = 2 * np.pi * 10, 0.01, 0.01, 0
t = create_time_specs(Tmax, dt)


### Create folder for the outputs of the model
MODEL_PATH = '/Users/lauragarcialopez/Documents/uni/TFG/06.TFG/codes/trained_models/SBIAE/'
model_dir = f"{MODEL_PATH}SBIAE_{model_num}"
try:
    os.mkdir(model_dir)
    print(f"Directory '{model_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{model_dir}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{model_dir}'.")
except Exception as e:
    print(f"An error occurred: {e}")


### Write specs of the model
file_path = f"{MODEL_PATH}SBIAE_{model_num}/hyperparameters.txt"
with open(file_path, "w") as file:
    file.write(f"Model {model_num}\n")
    file.write(f"-----------\n")
    file.write(f"Noise Factor : (Dx, Dy) = ({Dx}, {Dy})\n")
    file.write(f"Time Specifications : Tmax = {Tmax} & dt = {dt}\n")
    file.write(f"-----------\n")
    file.write(f"Embedding Dimension : {embedding_dim}\n")
    file.write(f"Dataset Ratios : P = {p} & N = {n}\n")
    file.write(f"Regularization Loss Weight (alpha) : {alpha}\n")
    file.write(f"Triplet Loss Weight (beta) : {beta}\n")


### Import datasets
from datasets.MyDataset import Dataset
from datasets.MyTripletDataset import TripletDataset
from torch.utils.data import DataLoader

DATASET_PATH = '/Users/lauragarcialopez/Documents/uni/TFG/06.TFG/codes/data/'

# Load training dataset
train_dataset = TripletDataset(f"{DATASET_PATH}train_dataset__Dx={str(Dx)}_Tmax={Tmax}.json", p, n)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load validation dataset
# valid_dataset = Dataset(f"{DATASET_PATH}valid_dataset__Dx={str(Dx)}_Tmax={Tmax}.json")
valid_dataset = TripletDataset(f"{DATASET_PATH}valid_dataset__Dx={str(Dx)}_Tmax={Tmax}.json", p, n)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# Load test dataset
test_dataset = Dataset(f"{DATASET_PATH}test_dataset__Dx={str(Dx)}_Tmax={Tmax}.json")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Personalize Model
params = {'encoder' : {'linear_layers' : {'input_dim' : len(t),
                                          'output_dim' : embedding_dim,
                                          'hidden_layers' : [1024, 512, 128, 32]}},
          'decoder' : {'linear_layers' : {'input_dim' : embedding_dim,
                                          'output_dim' : len(t),
                                          'hidden_layers' : [32, 128, 512, 1024]}}}

file_path = f"{MODEL_PATH}SBIAE_{model_num}/model_params.json"
with open(file_path, "w") as file:
    json.dump(params, file, indent=4)


from autoencoders.Autoencoder import MyAutoencoder

autoencoder = MyAutoencoder(params).to(device)
autoencoder = autoencoder.to(device)


from sbi.inference import NPE
from sbi.utils import BoxUniform
from models.MyLOModel2D import create_time_specs, run_LOModel

Tmax, dt = 2 * np.pi * 10, 0.01
t = create_time_specs(Tmax, dt)
Dx = 0.01

def simulate_LO_model(params):
    simulation = run_LOModel(params.squeeze(), t, dt, Dx=Dx)
    return simulation["x"]

prior_min, prior_max = [0.5] * 2, [1.5] * 2
low, high = torch.tensor(prior_min), torch.tensor(prior_max)
prior = BoxUniform(low=low, high=high)

inference = NPE(prior)

theta_0 = prior.sample((10, ))
x = []
for theta in theta_0:
    x += [simulate_LO_model(theta)]
x_0 = torch.stack(x)
z_0 = autoencoder.encode(x_0.view((10, -1))).detach()

_ = inference.append_simulations(theta_0, z_0).train(max_num_epochs=0)


from losses.CombinedLoss import CombinedLoss
from torch.optim import lr_scheduler

# Personalize Reconstruction Loss
# possible losses: ['mse_loss', 'cos_loss', isiwdist_loss', 'freqwdist_loss', 'dmse_loss']
loss_params = {'mse_loss' : {'weight' : 1}}

file_path = f"{MODEL_PATH}SBIAE_{model_num}/loss_params.json"
with open(file_path, "w") as file:
    json.dump(loss_params, file, indent=4)

criterion = CombinedLoss(loss_params, Tmax=Tmax, dt=dt)
optimizer = optim.Adam(list(autoencoder.parameters()) + list(inference._neural_net.parameters()), lr=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)


from functions.AE_Functions import joint_training, test, encode

print("The model training starts now")
joint_training(autoencoder, inference, criterion, optimizer, 700, train_loader, valid_loader, patience=50, alpha=alpha, beta=beta, report_epochs=20, plot=True, model_name=f'AE.pth', inference_model_name=f'SBI.pth', model_dir=f"{MODEL_PATH}SBIAE_{model_num}/")

# Ideally this plots can be saved
print("Testing the model is starting")
test(autoencoder, criterion, test_loader, plot=True, model_dir=f"{MODEL_PATH}SBIAE_{model_num}/")
encode(autoencoder, test_loader, p=0, model_dir=f"{MODEL_PATH}SBIAE_{model_num}/")
encode(autoencoder, test_loader, p=0, ls=True, model_dir=f"{MODEL_PATH}SBIAE_{model_num}/")
encode(autoencoder, test_loader, p=1, model_dir=f"{MODEL_PATH}SBIAE_{model_num}/")