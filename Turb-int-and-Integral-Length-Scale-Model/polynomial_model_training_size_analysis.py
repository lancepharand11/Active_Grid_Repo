# -*- coding: utf-8 -*-
# Multi Output Neural Network Model for Active Grid
# Author: Lance Pharand, 2025
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code in an academic application)
# Download the following packages below if not installed already
# !! IMPORTANT Set the Turbulence Parameters Class variables in the "Initializations" section !!

###################################################################
## Initializations and Data Loading
###################################################################
import scipy.io
import scipy.stats as stats
import pandas as pd
import numpy as np
import sys
import os

from train_polynomial_model import train_polynomial_model
sys.path.insert(0, os.path.abspath('../'))
from Turbulence_Parameters_class import Turbulence_Parameters
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
import torch

dataDir = Path("/Users/Connor/Nextcloud/Experimental Data/Active_Grid_Data_Lance/")
counter = 0
turb_objects = []
Turbulence_Parameters.fs = 25600
Turbulence_Parameters.N_samples = 6144000
Turbulence_Parameters.overlap = 0.5
Turbulence_Parameters.mesh_length = 0.06096

# for file in list(dataDir.glob('*.mat')):
#     if counter == 0:
#         time_stamps = scipy.io.loadmat(file, variable_names=['timeStamps'], squeeze_me=True, mat_dtype=True)
#     counter += 1
    
#     Ro_string = file.stem.split("_")[3]
#     if Ro_string == '-':
#         continue
#     file_Ro = float(Ro_string)
#     file_shaftSpeedSTD = float(file.stem.split("_")[5])
    
#     mat_u = scipy.io.loadmat(file, variable_names=['u'], squeeze_me=True, mat_dtype=True)
#     mat_v = scipy.io.loadmat(file, variable_names=['v'], squeeze_me=True, mat_dtype=True)
#     temp_turb_obj = Turbulence_Parameters(filename=file.stem, u_velo=mat_u['u'], v_velo=mat_v['v'],
#                                           freestream_velo=np.mean(mat_u['u'][4000000:]), Rossby_num=file_Ro,
#                                           shaft_speed_std_dev=file_shaftSpeedSTD)
#     temp_turb_obj.filter_velo()
#     temp_turb_obj.calc_L_ux()
#     temp_turb_obj.calc_turb_intensity()
#     turb_objects.append(temp_turb_obj)
    
#     print(f"Loading file {counter}")

# IO_data = pd.DataFrame({"Trial Name": (turb_obj.get_trial_name() for turb_obj in turb_objects),
#                         "Grid Re": (turb_obj.get_grid_Re() for turb_obj in turb_objects),
#                         "Rossby Number": (turb_obj.get_Rossby_num() for turb_obj in turb_objects),
#                         "Shaft Speed Standard Deviation * M / U": (turb_obj.get_shaft_speed_std_dev() for turb_obj in turb_objects),
#                         "Turbulence Intensity": (turb_obj.get_turb_int() for turb_obj in turb_objects),
#                         "L_ux / M": (turb_obj.get_L_ux_non_dim() for turb_obj in turb_objects),
#                         })

IO_data_file_path = "../OLD-and-Extra/DataSummary.csv"
IO_data = pd.read_csv(IO_data_file_path)

IO_data = IO_data[["Trial Name",
                   "Grid Re",
                   "Rossby Number",
                   "Shaft Speed Standard Deviation * M / u_inf",
                   "Turbulence Intensity",
                   "L_ux / M",
                   "Turbulence Intensity Uncertainty",
                   "L_ux Uncertainty"]]

###################################################################
## Preprocessing
###################################################################
X_all = IO_data.iloc[:, 1:4]
Y_all = IO_data.iloc[:, 4:6]
Y_Uncertainty = IO_data.iloc[:, 6:8]
# XY = pd.concat([X, Y], axis=1)
# z_scores = np.abs(stats.zscore(XY, nan_policy='omit'))
# threshold = 3  # Threshold z-score
# rows_with_outlier = (z_scores > threshold).any(axis=1)
# XY_filtered = XY[~rows_with_outlier]

# X_filtered = XY_filtered.iloc[:, :X.shape[1]]
# Y_filtered = XY_filtered.iloc[:, X.shape[1]:]

X_all = torch.tensor(X_all.values, dtype=torch.float32)
Y_all = torch.tensor(Y_all.values, dtype=torch.float32)
###################################################################
## Experiment Simulation Setup
###################################################################
polynomial_order = 5
min_train_fraction = 0.1
max_train_fraction = 0.90
n_steps = 40

test_fraction = 0.1

overall_fraction_acceptable = np.zeros(n_steps)
overall_rel_rmse_turb_int = np.zeros(n_steps)
overall_norm_rmse_Lux = np.zeros(n_steps)
train_data_size = np.zeros(n_steps)

for train_fraction_idx, train_fraction in enumerate(np.linspace(min_train_fraction,max_train_fraction,n_steps)):
    
    # Number of experiments to perform for each size of simulated experimental dataset
    n_experiments = 20
    SS = ShuffleSplit(n_splits=n_experiments, train_size=train_fraction, test_size=test_fraction, random_state=25)

    best_overall_rmse = np.inf
    best_overall_model = None
    best_rel_rmse_turb_int = None
    best_rel_rmse_L_ux = None
    best_scaler_x = None
    best_scaler_y = None
    rel_rmse_turb_int = np.zeros(n_experiments)
    rel_rmse_L_ux = np.zeros(n_experiments)

    for experiment, (experiment_train_idx, experiment_test_idx) in enumerate(SS.split(X_all)):
        
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Get the subset of data for the simulated experiment
        X_experiment = X_all[experiment_train_idx]
        Y_experiment = Y_all[experiment_train_idx]
        Y_Uncertainty_experiment = Y_Uncertainty.iloc[experiment_train_idx]
        
        # Get the subset of data for evaluation of the model
        X_test = X_all[experiment_test_idx]
        Y_test = Y_all[experiment_test_idx]
        Y_Uncertainty_test = Y_Uncertainty.iloc[experiment_test_idx]
        
        model, scaler_x, scaler_y, train_rmse = train_polynomial_model(X_experiment, Y_experiment, polynomial_order, device)

        x_test = scaler_x.transform(X_test)
        y_test = scaler_y.transform(Y_test)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        y_val_pred = model.predict(x_test)
        y_val_pred_unscaled_np = scaler_y.inverse_transform(y_val_pred)
        y_val_unscaled_np = Y_all[experiment_test_idx].numpy()
        residuals = y_val_unscaled_np - y_val_pred_unscaled_np

        # Relative RMSE
        rel_err_turb_int = (y_val_pred_unscaled_np[:, 0] - y_val_unscaled_np[:, 0]) / y_val_unscaled_np[:, 0]
        rel_err_L_ux = (y_val_pred_unscaled_np[:, 1] - y_val_unscaled_np[:, 1]) / y_val_unscaled_np[:, 1]
        rel_rmse_turb_int[experiment] = np.sqrt(np.sum(rel_err_turb_int**2) / rel_err_turb_int.size)
        rel_rmse_L_ux[experiment] = np.sqrt(np.sum(rel_err_L_ux**2) / rel_err_L_ux.size)

        print(f"Experiment {experiment + 1} Normalized RMSEs:")
        print(f"    Turbulence Intensity: {rel_rmse_turb_int[experiment]:.4f}")
        print(f"    L_ux / M: {rel_rmse_L_ux[experiment]:.4f}")
    
        
    # Print training data size
    train_data_size[train_fraction_idx] = np.floor(X_all.size(0)*train_fraction)
    print(f"Number of training data points: {train_data_size[train_fraction_idx]}")
    
    # Print overall rmse for each size of training dataset
    overall_rel_rmse_turb_int[train_fraction_idx] = np.mean(rel_rmse_turb_int)
    print(f"Overall Tu Root-mean-square-error: {overall_rel_rmse_turb_int[train_fraction_idx]}")
    overall_norm_rmse_Lux[train_fraction_idx] = np.mean(rel_rmse_L_ux)
    print(f"Overall L_ux Root-mean-square-error: {overall_norm_rmse_Lux[train_fraction_idx]}")


# Plot the results
fig, ax = plt.subplots(1,1)
ax.plot(train_data_size,overall_rel_rmse_turb_int)
ax.plot(train_data_size,overall_norm_rmse_Lux)

# Save the results for plotting later
training_size_data = pd.DataFrame({"Size of Training Data": train_data_size,
                                   "Tu RMS Relative Error": overall_rel_rmse_turb_int,
                                   "L_ux RMS Relative Error": overall_norm_rmse_Lux})

training_size_data.to_csv("./Training Data Size Analysis Results/Polynomial.csv")
