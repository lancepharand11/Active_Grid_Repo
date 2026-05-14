# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:05:25 2025

@author: Connor
"""

# Multi Output Neural Network Model for Active Grid
# Author: Lance Pharand, 2025
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code in an academic application)
# Download the following packages below if not installed already
# !! IMPORTANT Set the Turbulence Parameters Class variables in the "Initializations" section !!

###################################################################
## Initializations and Data Loading
###################################################################
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from Turbulence_Parameters_class import Turbulence_Parameters
from train_nn import train_nn
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
import torch
from get_model import get_model
import copy
import mat73
import math

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hearst_ro = mat73.loadmat("Hearst2015_Ro.mat")
hearst_u = mat73.loadmat("Hearst2015_U.mat")

IO_data_ro = pd.DataFrame(hearst_ro)
IO_data_u = pd.DataFrame(hearst_u)

IO_data_ro = IO_data_ro[['Re_M','Ro','omega','Tu','L_ux']]
IO_data_u = IO_data_u[['Re_M','Ro','omega','Tu','L_ux']]

IO_data = pd.concat([IO_data_ro,IO_data_u])

###################################################################
## Preprocessing
###################################################################
X = IO_data.iloc[:, 0:3]
Y = IO_data.iloc[:, 3:5]

X_all = torch.tensor(X.values, dtype=torch.float32)
Y_all = torch.tensor(Y.values, dtype=torch.float32)
###################################################################
## Experiment Simulation Setup
###################################################################

min_train_fraction = 0.2
max_train_fraction = 0.64
n_steps = 2

test_fraction = 0.36

overall_rmse_turb_int = np.zeros(n_steps)
overall_rmse_L_ux = np.zeros(n_steps)
train_data_size = np.zeros(n_steps)

###################################################################
## Model Setup
###################################################################
input_size, output_size = 3, 2
hidden_size = 3
n_hidden_layers = 2
max_epochs = 10000
min_epochs = 200
learning_rate = 1e-3
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

model = get_model(input_size, hidden_size, output_size, n_hidden_layers, device)
model_compiled = torch.compile(model)
model_initial_parameters = copy.deepcopy(model_compiled.state_dict())

for train_fraction_idx, train_fraction in enumerate(np.linspace(min_train_fraction,max_train_fraction,n_steps)):
    
    # Number of experiments to perform for each size of simulated experimental dataset
    n_experiments = np.min([10,math.comb(len(IO_data),np.int64(train_fraction*len(IO_data)))])
    rmse_turb_int = np.zeros(n_experiments)
    rmse_L_ux = np.zeros(n_experiments)
    
    SS = ShuffleSplit(n_splits=n_experiments, train_size=train_fraction, test_size=test_fraction, random_state=0)
    
    # Loop through each experiment
    for experiment, (experiment_train_idx, experiment_test_idx) in enumerate(SS.split(X_all)):
        
        # Get the subset of data for the simulated experiment
        X_experiment = X_all[experiment_train_idx]
        Y_experiment = Y_all[experiment_train_idx]
        
        # Get the subset of data for evaluation of the model
        X_test = X_all[experiment_test_idx]
        Y_test = Y_all[experiment_test_idx]
        
        # Train the model on the data from the simulated experiment
        model_compiled.load_state_dict(model_initial_parameters)
        
        (weights, scaler_x, scaler_y,
         mse_crit) = train_nn(model_compiled, X_experiment, Y_experiment,
                           hidden_size=hidden_size,
                           max_epochs=max_epochs, 
                           min_epochs=min_epochs,
                           learning_rate=learning_rate,
                           batch_size=batch_size, device=device, plot=False)
    
        #
        # Get the relative errors for the validation data
        #
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            
            X_test = scaler_x.transform(X_test)
            Y_test = scaler_y.transform(Y_test)

            X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
            Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
            
            Y_test_pred = model(X_test)
            Y_test_pred_unscaled_np = scaler_y.inverse_transform(Y_test_pred.cpu().numpy())
            Y_test_unscaled_np = scaler_y.inverse_transform(Y_test.numpy())
            residuals = Y_test_unscaled_np - Y_test_pred_unscaled_np
        
            # Relative RMSE
            err_turb_int = (Y_test_pred_unscaled_np[:, 0] - Y_test_unscaled_np[:, 0])
            err_L_ux = (Y_test_pred_unscaled_np[:, 1] - Y_test_unscaled_np[:, 1])
            rmse_turb_int[experiment] = np.sqrt(np.sum(err_turb_int**2) / err_turb_int.size)
            rmse_L_ux[experiment] = np.sqrt(np.sum(err_L_ux**2) / err_L_ux.size)
        
            print(f"Experiment {experiment + 1} of {n_experiments} RMSEs:")
            print(f"    Turbulence Intensity: {rmse_turb_int[experiment]:.4f}")
            print(f"    L_ux / M: {rmse_L_ux[experiment]:.4f}")
    
    # Print training data size
    train_data_size[train_fraction_idx] = np.floor(X_all.size(0)*train_fraction)
    print(f"Number of training data points: {train_data_size[train_fraction_idx]}")
    
    # Print overall rmse for each size of training dataset
    overall_rmse_turb_int[train_fraction_idx] = np.sqrt(np.sum(np.square(rmse_turb_int)) / rmse_turb_int.size)
    print(f"Overall Tu Root-mean-square-error: {overall_rmse_turb_int[train_fraction_idx]}")
    overall_rmse_L_ux[train_fraction_idx] = np.sqrt(np.sum(np.square(rmse_L_ux)) / rmse_L_ux.size)
    print(f"Overall L_ux Root-mean-square-error: {overall_rmse_L_ux[train_fraction_idx]}")


# Plot the results
fig, ax = plt.subplots(1,1)
ax.scatter(train_data_size,overall_rmse_turb_int)
ax.scatter(train_data_size,overall_rmse_L_ux)

# Save the results for plotting later
training_size_data = pd.DataFrame({"Size of Training Data": train_data_size,
                                   "Tu RMS Error": overall_rmse_turb_int,
                                   "L_ux RMS Error": overall_rmse_L_ux})

training_size_data.to_csv(f"./Training Data Size Analysis Results/Neural Network {n_hidden_layers}-Layer-Hearst.csv")
