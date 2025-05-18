# Initial Turbulence Intensity Model for Active Grid
# Author: Lance Pharand, 2024
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code)
# Download the following packages below if not installed already
# !! IMPORTANT Set the Turbulence Parameters Class variables in the "Initializations" section !!

import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from Turbulence_Parameters_class import Turbulence_Parameters
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
import copy
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


dataDir = "/Users/lancepharand/Desktop/URA_S24/Experiment_Scripts/Active_Grid_Data_and_Files/Active_Grid_Data/"
counter = 0
turb_objects = []
Turbulence_Parameters.fs = 25600 #Hz
Turbulence_Parameters.N_samples = 6144000
Turbulence_Parameters.overlap = 0.5
Turbulence_Parameters.mesh_length = 0.06096
Turbulence_Parameters.num_sections = 10

# Filter
sos = signal.butter(6, 10000, btype='low', analog=False, fs=Turbulence_Parameters.fs, output='sos')

for file in os.listdir(dataDir):
    if counter == 0:
        time_stamps = scipy.io.loadmat((dataDir + file), variable_names=['timeStamps'], squeeze_me=True, mat_dtype=True)
        counter += 1

    if file == ".DS_Store":
        continue
    name_full = os.path.basename(dataDir + file).split("/")[-1]
    name = name_full.split(".mat")[0]
    mat_u = scipy.io.loadmat((dataDir + file), variable_names=['u'], squeeze_me=True, mat_dtype=True)
    mat_v = scipy.io.loadmat((dataDir + file), variable_names=['v'], squeeze_me=True, mat_dtype=True)
    temp_turb_obj = Turbulence_Parameters(filename=name, u_velo=mat_u['u'][4000000:], v_velo=mat_v['v'][4000000:],
                                          freestream_velo=float(name.split("_")[1]), Rossby_num=float(name.split("_")[3]),
                                          shaft_speed_std_dev=float(name.split("_")[5]))

    u_velo_filtered = signal.sosfilt(sos, temp_turb_obj.get_u_velo())
    temp_turb_obj.set_u_velo(u_velo_filtered)
    temp_turb_obj.calc_turb_psd_spectrum()
    temp_turb_obj.psd_breakaway_freq_inertial()
    temp_turb_obj.psd_breakaway_freq_dissip()
    temp_turb_obj.psd_inertial_range_slope()
    temp_turb_obj.psd_integral_sectioning()
    turb_objects.append(temp_turb_obj)

IO_data = pd.DataFrame({"Trial Name": (turb_obj.get_trial_name() for turb_obj in turb_objects),
                        "Grid Re": (turb_obj.get_grid_Re() for turb_obj in turb_objects),
                        "Rossby Number": (turb_obj.get_Rossby_num() for turb_obj in turb_objects),
                        "Shaft Speed Standard Deviation * M / U": (turb_obj.get_shaft_speed_std_dev() for turb_obj in turb_objects),
                        "E_11 / (U * M) [Non-Dim PSD]": (turb_obj.get_E_u().tolist() for turb_obj in turb_objects),
                        "freq * M / U [Non-Dim Freq]": (turb_obj.get_freq_non_dim().tolist() for turb_obj in turb_objects),
                        "Log(E_11 / (U * M))": (turb_obj.get_log_E_u() for turb_obj in turb_objects),
                        "Log(freq * M / U)": (turb_obj.get_log_freq_non_dim() for turb_obj in turb_objects),
                        "PSD Log Zero Freq": (turb_obj.get_zero_freq() for turb_obj in turb_objects),
                        "PSD Log Value at Zero Freq": (turb_obj.get_e_zero_freq() for turb_obj in turb_objects),
                        "PSD Inertial Breakaway Freq": (turb_obj.get_breakaway_freq_inertial() for turb_obj in turb_objects),
                        "PSD Dissipative Breakaway Freq": (turb_obj.get_breakaway_freq_dissip() for turb_obj in turb_objects),
                        "PSD Inertial Range Slope": (turb_obj.get_e_slope() for turb_obj in turb_objects),
                        "PSD Slope": (turb_obj.get_dE_u_dfreq() for turb_obj in turb_objects)
                        })

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

device = torch.device(device)

# Split the data. NOTE: Using only psd values for the y
X = IO_data.iloc[:, [1, 2, 3]]
Y = IO_data.iloc[:, [8, 9, 10, 11, 12]]
XY = pd.concat([X, Y], axis=1)
z_scores = np.abs(stats.zscore(XY, nan_policy='omit'))
threshold = 3  # Threshold z-score
rows_with_outlier = (z_scores > threshold).any(axis=1)
XY_filtered = XY[~rows_with_outlier]

# Split back into X and Y
X_filtered = XY_filtered.iloc[:, :X.shape[1]]
Y_filtered = XY_filtered.iloc[:, X.shape[1]:]

x_train, x_test, y_train, y_test = train_test_split(X_filtered, Y_filtered,
                                                    stratify=X_filtered["Grid Re"],
                                                    test_size=0.25, random_state=42)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

device = torch.device(device)
input_size = 3
hidden_size = 64
output_size = 5
num_epochs = 700
learning_rate = 0.001
# train_running_loss = 0
# test_running_loss = 0
batch_size = 16

# Transform data for NN
scaler1 = MinMaxScaler(feature_range=(-1, 1))
scaler2 = MinMaxScaler(feature_range=(-1, 1))
x_train_scaled = scaler1.fit_transform(x_train)  # fit scaler1 based on input training set
x_test_scaled = scaler1.transform(x_test)
y_train_scaled = scaler2.fit_transform(y_train)  # fit scaler2 based on output training set
y_test_scaled = scaler2.transform(y_test)

# Setup model
model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.LeakyReLU(),
                      nn.Linear(hidden_size, hidden_size // 2),
                      nn.LeakyReLU(),
                      nn.Linear(hidden_size // 2, output_size)
                      ).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

###################################################################
## Training and Storing Best Model
###################################################################
# Keep the best model
best_rmse = np.inf # init to infinity
best_epoch = -1
best_weights = None

# Init lists to store losses for each epoch
train_loss = []
test_loss = []

# Convert SCALED values to tensors
x_train_tens = torch.tensor(x_train_scaled, dtype=torch.float32, requires_grad=True).to(device)
x_test_tens = torch.tensor(x_test_scaled, dtype=torch.float32, requires_grad=True).to(device)
y_train_tens = torch.tensor(y_train_scaled, dtype=torch.float32, requires_grad=True).to(device)
y_test_tens = torch.tensor(y_test_scaled, dtype=torch.float32, requires_grad=True).to(device)

train_dataset = TensorDataset(x_train_tens, y_train_tens)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0

    #
    # Train
    #
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # Average training loss for this epoch
    epoch_train_loss /= len(train_loader)
    train_loss.append(epoch_train_loss)

    #
    # Test model with gradient updates off
    #
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test_tens)

        # Inverse scaling for test predictions and true values
        y_test_pred_unscaled = torch.tensor(
            scaler2.inverse_transform(y_test_pred.detach().cpu().numpy())
        )
        y_test_unscaled = torch.tensor(y_test.to_numpy())

        # Calculate overall RMSE (unscaled)
        loss_unscaled = float(torch.sqrt(criterion(y_test_pred_unscaled, y_test_unscaled)))

        test_loss.append(loss_unscaled)
        if loss_unscaled < best_rmse:
            best_epoch = epoch
            best_rmse = loss_unscaled
            best_weights = copy.deepcopy(model.state_dict())

# Save the best model
model_path = "turb_spectrum_generation_model.pth"
torch.save(best_weights, model_path)

print(f"Best model and scalers saved to {model_path}")
print(f"Best Epoch: {best_epoch}, Best Test RMSE: {best_rmse} (unscaled error)")

#
# For testing PSD generation (upto end of inertial range)
#
model.load_state_dict(best_weights)
model.eval()

for trial_num in range(X_filtered.shape[0]):
    plt.figure(figsize=(10, 8))

    grid_Re, ro_num, spd_std_dev = (IO_data["Grid Re"][trial_num],
                                    IO_data["Rossby Number"][trial_num],
                                    IO_data["Shaft Speed Standard Deviation * M / U"][trial_num])
    input_np = np.array([[grid_Re, ro_num, spd_std_dev]])
    input_np_scaled = scaler1.transform(input_np)
    input_tens_scaled = torch.tensor(input_np_scaled, dtype=torch.float32).to(device)

    turb_features_scaled = model(input_tens_scaled)
    turb_features_scaled = turb_features_scaled.detach().cpu().numpy()
    turb_features = scaler2.inverse_transform(turb_features_scaled).reshape(-1)

    #
    # From data
    #
    plt.plot(IO_data["Log(freq * M / U)"][trial_num], IO_data["Log(E_11 / (U * M))"][trial_num], 'b-', label='Full PSD')
    x_data1 = np.linspace(IO_data["PSD Inertial Breakaway Freq"][trial_num],
                         IO_data["PSD Dissipative Breakaway Freq"][trial_num], 40)
    b_data1 = IO_data["PSD Log Value at Zero Freq"][trial_num] - (
            IO_data["PSD Inertial Range Slope"][trial_num] * IO_data["PSD Inertial Breakaway Freq"][trial_num])
    y_data1 = IO_data["PSD Inertial Range Slope"][trial_num] * x_data1 + b_data1
    x_data2 = np.linspace(IO_data["PSD Log Zero Freq"][trial_num],
                          IO_data["PSD Inertial Breakaway Freq"][trial_num], 20)
    y_data2 = [IO_data["PSD Log Value at Zero Freq"][trial_num]] * x_data2.shape[0]
    x_data = np.concatenate((x_data2, x_data1), axis=0)
    y_data = np.concatenate((y_data2, y_data1), axis=0)
    plt.plot(x_data, y_data, 'g-', label='Expected PSD Approx')

    #
    # From NN
    #
    x_pred1 = np.linspace(turb_features[2],
                         turb_features[3], 40)
    b_pred1 = turb_features[1] - (
            turb_features[4] * turb_features[2])
    y_pred1 = turb_features[4] * x_pred1 + b_pred1
    x_pred2 = np.linspace(turb_features[0],
                         turb_features[2], 20)
    y_pred2 = [turb_features[1]] * x_pred2.shape[0]
    x_pred = np.concatenate((x_pred2, x_pred1), axis=0)
    y_pred = np.concatenate((y_pred2, y_pred1), axis=0)
    plt.plot(x_pred, y_pred, 'r-', label='Predicted PSD Approx')

    plt.xlabel("Log(freq * M / U)")
    plt.ylabel("Log(E_u / (U * M))")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
print(1)
