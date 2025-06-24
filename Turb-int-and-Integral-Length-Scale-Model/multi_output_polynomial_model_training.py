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
import scipy.io
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from Turbulence_Parameters_class import Turbulence_Parameters
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from torch import nn
import torch
import copy

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
X = IO_data.iloc[:, 1:4]
Y = IO_data.iloc[:, 4:6]
Y_Uncertainty = IO_data.iloc[:, 6:8]
# XY = pd.concat([X, Y], axis=1)
# z_scores = np.abs(stats.zscore(XY, nan_policy='omit'))
# threshold = 3  # Threshold z-score
# rows_with_outlier = (z_scores > threshold).any(axis=1)
# XY_filtered = XY[~rows_with_outlier]

# X_filtered = XY_filtered.iloc[:, :X.shape[1]]
# Y_filtered = XY_filtered.iloc[:, X.shape[1]:]

X_all = torch.tensor(X.values, dtype=torch.float32)
Y_all = torch.tensor(Y.values, dtype=torch.float32)
###################################################################
## Shuffle-Split CV Setup
###################################################################
n_models = 10
SS = ShuffleSplit(n_splits=n_models, test_size=0.80, random_state=0)  # NOTE: no seed used

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def get_model():
    model = Pipeline([('poly', PolynomialFeatures(degree=5)), 
                      ('linear', LinearRegression(fit_intercept=False))])
    return model

###################################################################
## Shuffle-Split based training Loop
###################################################################
fold_results = []
best_overall_rmse = np.inf
best_overall_model = None
best_norm_rmse_turb_int = None
best_norm_rmse_L_ux = None
best_scaler_x = None
best_scaler_y = None
fraction_acceptable = np.zeros(n_models)

for fold, (train_idx, val_idx) in enumerate(SS.split(X_all)):
    print(f"\nFold {fold + 1}")
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    x_train = scaler_x.fit_transform(X_all[train_idx])
    y_train = scaler_y.fit_transform(Y_all[train_idx])
    x_val = scaler_x.transform(X_all[val_idx])
    y_val = scaler_y.transform(Y_all[val_idx])
    
    y_val_uncertainty = Y_Uncertainty.iloc[val_idx]

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = get_model()
    mse_crit = nn.MSELoss()
    
    best_rmse, best_epoch = np.inf, -1
    best_weights = None
    
    model.fit(x_train,y_train)

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    y_train_pred_unscaled = torch.tensor(scaler_y.inverse_transform(y_train_pred))
    y_val_pred_unscaled = torch.tensor(scaler_y.inverse_transform(y_val_pred))

    y_train_unscaled = torch.tensor(Y_all[train_idx].numpy())
    y_val_unscaled = torch.tensor(Y_all[val_idx].numpy())

    train_rmse = torch.sqrt(mse_crit(y_train_pred_unscaled, y_train_unscaled)).item()
    val_rmse = torch.sqrt(mse_crit(y_val_pred_unscaled, y_val_unscaled)).item()

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_weights = copy.deepcopy(model)

    print(f" RMSE (unscaled): {best_rmse:.4f}")
    fold_results.append(best_rmse)

    #
    # Residuals plot and compare against measurement uncertainty
    #
    model = best_weights
   
    y_val_pred = model.predict(x_val)
    y_val_pred_unscaled_np = scaler_y.inverse_transform(y_val_pred)
    y_val_unscaled_np = Y_all[val_idx].numpy()
    residuals = y_val_unscaled_np - y_val_pred_unscaled_np
    
    # How many predictions are within measurement uncertainty?
    within_Uncertainty = residuals < y_val_uncertainty.to_numpy()
    fraction_acceptable[fold] = within_Uncertainty.sum() / within_Uncertainty.size       

    # Compute per output RMSE
    rmse_turb_int = torch.sqrt(torch.tensor(mse_crit(torch.tensor(y_val_pred_unscaled_np[:, 0]),
                                                      torch.tensor(y_val_unscaled_np[:, 0])))
                               ).item()
    rmse_L_ux = torch.sqrt(torch.tensor(mse_crit(torch.tensor(y_val_pred_unscaled_np[:, 1]),
                                                  torch.tensor(y_val_unscaled_np[:, 1])))
                           ).item()

    # Normalize based on range
    range_turb_int = Y["Turbulence Intensity"].max() - Y["Turbulence Intensity"].min()
    range_L_ux = Y["L_ux / M"].max() - Y["L_ux / M"].min()
    norm_rmse_turb_int = rmse_turb_int / range_turb_int
    norm_rmse_L_ux = rmse_L_ux / range_L_ux

    print(f"Fold {fold + 1} Normalized RMSEs:")
    print(f"    Turbulence Intensity: {norm_rmse_turb_int:.4f}")
    print(f"    L_ux / M: {norm_rmse_L_ux:.4f}")

    temp_inputs = X_all[val_idx].detach().numpy()

    for i, target_name in enumerate(["Turbulence Intensity", "L_ux / M"]):
        # plt.figure(figsize=(10, 8))
        # plt.scatter(y_val_pred_unscaled_np[:, i], residuals[:, i], alpha=0.7, label=f"Residuals for {target_name}")
        # plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Residual Line")
        # plt.xlabel(f"Predicted {target_name} (unscaled)")
        # plt.ylabel(f"Residual {target_name} (unscaled)")
        # plt.title(f"Fold {fold + 1} Residual Plot: {target_name}")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        p1 = ax1.scatter(temp_inputs[:, 0], temp_inputs[:, 1], temp_inputs[:, 2],
                         c=residuals[:, i], cmap='magma',
                         marker='o', s=50, alpha=0.8
                         )
        cbar1 = fig1.colorbar(p1, ax=ax1, shrink=0.5, pad=0.1)
        cbar1.set_label('Residuals - ' + target_name)
        ax1.set_xlabel('Grid Re', labelpad=7)
        ax1.set_ylabel('Rossby Number')
        ax1.set_zlabel('Shaft Speed Std Dev * M / u_inf', labelpad=8, rotation=0)
        ax1.set_title('3D Scatter: ' + target_name)
        plt.show()

    # Track best model across all folds
    if best_rmse < best_overall_rmse:
        best_overall_rmse = best_rmse
        best_overall_train_idx, best_overall_val_idx = train_idx, val_idx
        best_norm_rmse_turb_int = norm_rmse_turb_int
        best_norm_rmse_L_ux = norm_rmse_L_ux
        best_overall_model = copy.deepcopy(best_weights)
        best_scaler_x = copy.deepcopy(scaler_x)
        best_scaler_y = copy.deepcopy(scaler_y)


###################################################################
## Save Best Model
###################################################################
from datetime import datetime
unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")

out_dir = "Models_and_Results"
os.makedirs(out_dir, exist_ok=True)

model_fname = os.path.join(out_dir, f"best_model_{unique_id}.pth")
train_idx_fname = os.path.join(out_dir, f"train_idx_{unique_id}.csv")
val_idx_fname = os.path.join(out_dir, f"val_idx_{unique_id}.csv")
scaler_x_fname = os.path.join(out_dir, f"scaler_x_{unique_id}.pkl")
scaler_y_fname = os.path.join(out_dir, f"scaler_y_{unique_id}.pkl")
log_fname = os.path.join(out_dir, "rmse_results.txt")

torch.save(best_overall_model, model_fname)
joblib.dump(best_scaler_x, scaler_x_fname)
joblib.dump(best_scaler_y, scaler_y_fname)


np.savetxt(train_idx_fname, best_overall_train_idx, delimiter=",", fmt="%f")
np.savetxt(val_idx_fname, best_overall_val_idx, delimiter=",", fmt="%f")

print(f"\nSaved best model weights to: {model_fname}")
print(f"Saved input scaler to: {scaler_x_fname}")
print(f"Saved output scaler to: {scaler_y_fname}")

print(f"Saved training data indices to: {train_idx_fname}")
print(f"Saved validation data indices to: {val_idx_fname}")

# Log results
log_line = (f"{unique_id}\t"
            f"{os.path.basename(model_fname)}\t"
            f"{best_overall_rmse:.4f}\t"
            f"{best_norm_rmse_turb_int:.4f}\t"
            f"{best_norm_rmse_L_ux:.4f}\n"
            )

# Print overall fraction of acceptable predictions
overall_fraction_acceptable = np.mean(fraction_acceptable)
print(f"Overall fraction of acceptable predictions: {overall_fraction_acceptable}")

with open(log_fname, "a") as f:
    f.write(log_line)

print(f"Appended results to {log_fname}")
print(1)
