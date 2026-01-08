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
sys.path.insert(0, os.path.abspath('../'))
from Turbulence_Parameters_class import Turbulence_Parameters
from train_polynomial_model import train_polynomial_model
from pathlib import Path
import joblib
import torch
from get_model_polynomial import get_model_polynomial

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

IO_data_file_path = "../OLD-and-Extra/DataSummaryOutliersRemoved.csv"
IO_data = pd.read_csv(IO_data_file_path)

IO_data = IO_data[["Trial Name",
                   "Grid Re",
                   "Rossby Number",
                   "Shaft Speed Standard Deviation * M^2 / nu",
                   "Turbulence Intensity",
                   "L_ux / M",
                   "Turbulence Intensity Uncertainty",
                   "L_ux Uncertainty"]]

###################################################################
## Preprocessing
###################################################################
X = IO_data.iloc[:, 1:4]
Y = IO_data.iloc[:, 4:6]
# Y_Uncertainty = IO_data.iloc[:, 6:8]
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
## Training Setup
###################################################################

polynomial_order = 2
input_size, output_size = X_all.shape[1], Y_all.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

model = get_model_polynomial(polynomial_order)

# Train the model on the data from the experiment
scaler_x, scaler_y, train_rmse = train_polynomial_model(model, X_all, Y_all, device)

###################################################################
## Save Model
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

joblib.dump(model, model_fname)
joblib.dump(scaler_x, scaler_x_fname)
joblib.dump(scaler_y, scaler_y_fname)


print(f"\nSaved model to: {model_fname}")
print(f"Saved input scaler to: {scaler_x_fname}")
print(f"Saved output scaler to: {scaler_y_fname}")

# # Log results
# log_line = (f"{unique_id}\t"
#             f"{os.path.basename(model_fname)}\t"
#             f"{best_overall_rmse:.4f}\t"
#             f"{best_norm_rmse_turb_int:.4f}\t"
#             f"{best_norm_rmse_L_ux:.4f}\n"
#             )

# with open(log_fname, "a") as f:
#     f.write(log_line)

# print(f"Appended results to {log_fname}")
# print(1)
