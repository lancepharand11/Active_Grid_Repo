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
import sys
import os
# sys.path.insert(0, os.path.abspath('../'))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from Turbulence_Parameters_class import Turbulence_Parameters
from train_nn import train_nn
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
import torch

# dataDir = Path('F:\Lance\Active_Grid_Model_Data')
dataDir = Path('D:/Active_Grid_Data_Lance/Selected Data')
script_dir = Path(__file__).parent
counter = 0
turb_objects = []
Turbulence_Parameters.fs = 25600
Turbulence_Parameters.N_samples = 6144000
Turbulence_Parameters.overlap = 0.5
Turbulence_Parameters.mesh_length = 0.06096
seed = 42

# for file in list(dataDir.glob('*.mat')):
#     if counter == 0:
#      time_stamps = scipy.io.loadmat(file, variable_names=['timeStamps'], squeeze_me=True, mat_dtype=True)
#     counter += 1

#     Ro_string = file.stem.split("_")[3]
#     if Ro_string == '-':
#      continue
#     file_Ro = float(Ro_string)
#     file_shaftSpeedSTD = float(file.stem.split("_")[5])

#     mat_u = scipy.io.loadmat(file, variable_names=['u'], squeeze_me=True, mat_dtype=True)
#     mat_v = scipy.io.loadmat(file, variable_names=['v'], squeeze_me=True, mat_dtype=True)
#     temp_turb_obj = Turbulence_Parameters(filename=file.stem, u_velo=mat_u['u'], v_velo=mat_v['v'],
#                                        freestream_velo=np.mean(mat_u['u'][4000000:]), Rossby_num=file_Ro,
#                                        shaft_speed_std_dev=file_shaftSpeedSTD)
#     temp_turb_obj.filter_velo()
#     temp_turb_obj.calc_L_ux()
#     temp_turb_obj.calc_turb_intensity()
#     turb_objects.append(temp_turb_obj)

#     print(f"Loading file {counter}")

# IO_data = pd.DataFrame({"Trial Name": (turb_obj.get_trial_name() for turb_obj in turb_objects),
#                        "Grid Re": (turb_obj.get_grid_Re() for turb_obj in turb_objects),
#                        "Rossby Number": (turb_obj.get_Rossby_num() for turb_obj in turb_objects),
#                        "Turbulence Intensity": (turb_obj.get_turb_int() for turb_obj in turb_objects),
#                        "L_ux / M": (turb_obj.get_L_ux_non_dim() for turb_obj in turb_objects),
#                        })

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
# %% Training Setup
###################################################################
X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.10, random_state=seed)

input_size, output_size = X_all.shape[1], Y_all.shape[1]
hidden_size = 3
n_hidden_layers = 1
num_epochs = 10000
learning_rate = 1e-3
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Train the model on the data from the experiment
(weights, scaler_x, scaler_y,
        rmse, mse_crit, model) = train_nn(X_all, Y_all,                                                      
                   hidden_size=hidden_size,
                   n_hidden_layers=n_hidden_layers,
                   max_epochs=num_epochs, 
                   min_epochs=100,
                   learning_rate=learning_rate,
                   batch_size=batch_size, device=device, plot=True)
                                                            

with torch.no_grad():
    x_test_scaled = scaler_x.transform(X_val.numpy())
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)
    y_test_pred = model(x_test_tensor)
    y_test_pred_unsc = torch.tensor(scaler_y.inverse_transform(y_test_pred.cpu().numpy()))
    y_test_unsc = torch.tensor(Y_val.numpy())
    test_rmse = torch.sqrt(mse_crit(y_test_pred_unsc, y_test_unsc)).item()
print(f"Test set RMSE: {test_rmse:.4f}")


###################################################################
# %% Save Model
###################################################################
from datetime import datetime
unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")

out_dir = script_dir / "Models_and_Results"
out_dir.mkdir(exist_ok=True)

model_fname = os.path.join(out_dir, f"best_model_{unique_id}.pth")
train_idx_fname = os.path.join(out_dir, f"train_idx_{unique_id}.csv")
val_idx_fname = os.path.join(out_dir, f"val_idx_{unique_id}.csv")
scaler_x_fname = os.path.join(out_dir, f"scaler_x_{unique_id}.pkl")
scaler_y_fname = os.path.join(out_dir, f"scaler_y_{unique_id}.pkl")
log_fname = os.path.join(out_dir, "rmse_results.txt")

torch.save(weights, model_fname)
joblib.dump(scaler_x, scaler_x_fname)
joblib.dump(scaler_y, scaler_y_fname)

print(f"\nSaved model weights to: {model_fname}")
print(f"Saved input scaler to: {scaler_x_fname}")
print(f"Saved output scaler to: {scaler_y_fname}")

# Log results
log_line = (f"{unique_id}\t"
            f"{os.path.basename(model_fname)}\t"
            f"{test_rmse:.4f}\t"  # on the test set
            )

with open(log_fname, "a") as f:
    f.write(log_line)

print(f"Appended results to {log_fname}")
print(1)
