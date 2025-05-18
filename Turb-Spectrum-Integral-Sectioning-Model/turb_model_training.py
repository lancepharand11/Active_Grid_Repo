# Turbulence Spectrum Integral Sectioning Model for Active Grid
# Author: Lance Pharand, 2024
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code in an academic application)
# Download the following packages below if not installed already
# !! IMPORTANT Set the Turbulence Parameters Class variables in the "Initializations" section !!

import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from Turbulence_Parameters_class import Turbulence_Parameters
from torch.utils.data import DataLoader, TensorDataset
import joblib

dataDir = "/Users/lancepharand/Desktop/URA_S24/Experiment_Scripts/Active_Grid_Data_and_Files/Active_Grid_Data/"
counter = 0
turb_objects = []
Turbulence_Parameters.fs = 25600 #Hz
Turbulence_Parameters.N_samples = 6144000
Turbulence_Parameters.overlap = 0.5
Turbulence_Parameters.mesh_length = 0.06096
Turbulence_Parameters.num_sections = 4

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

    temp_turb_obj.calc_turb_psd_spectrum()
    temp_turb_obj.calc_L_ux()
    temp_turb_obj.calc_turb_intensity()
    temp_turb_obj.psd_breakaway_freq_inertial()
    temp_turb_obj.psd_breakaway_freq_dissip()
    temp_turb_obj.psd_inertial_range_slope()
    temp_turb_obj.psd_integral_sectioning()
    turb_objects.append(temp_turb_obj)

IO_data = pd.DataFrame({"Trial Name": (turb_obj.get_trial_name() for turb_obj in turb_objects),
                        "Grid Re": (turb_obj.get_grid_Re() for turb_obj in turb_objects),
                        "Rossby Number": (turb_obj.get_Rossby_num() for turb_obj in turb_objects),
                        "Shaft Speed Standard Deviation * M / u_inf": (turb_obj.get_shaft_speed_std_dev() for turb_obj in turb_objects),
                        "Turbulence Intensity": (turb_obj.get_turb_int() for turb_obj in turb_objects),
                        "L_ux / M": (turb_obj.get_L_ux_non_dim() for turb_obj in turb_objects),
                        "E_11 / (M * U) [Non-Dim PSD]": (turb_obj.get_E_u().tolist() for turb_obj in turb_objects),
                        "Freq * M / U [Non-Dim Freq]": (turb_obj.get_freq_non_dim().tolist() for turb_obj in turb_objects),
                        "Log(E_11 / (M * U))": (turb_obj.get_log_E_u() for turb_obj in turb_objects),
                        "Log(Freq * M / U)": (turb_obj.get_log_freq_non_dim() for turb_obj in turb_objects),
                        "PSD Integral Sections": (turb_obj.get_integral_sections() for turb_obj in turb_objects)
                        })

###################################################################
## Preprocessing and Model Setup
###################################################################
from sklearn.model_selection import train_test_split
from torch import nn, optim
import copy
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Split the data. NOTE: Using only Grid Re for the y
Y = IO_data.iloc[:, 1]
col_labels = ["Integral section " + str(s) for s in range(Turbulence_Parameters.num_sections)]
X = pd.DataFrame(IO_data.iloc[:, 10].to_list(), columns=col_labels)
XY = pd.concat([X, Y], axis=1)

# Compute z-scores for each column and filter
z_scores = np.abs(stats.zscore(XY, nan_policy='omit'))
threshold = 3  # Threshold z-score
rows_with_outlier = (z_scores > threshold).any(axis=1)
XY_filtered = XY[~rows_with_outlier]
X_filtered = XY_filtered.iloc[:, :X.shape[1]]
Y_filtered = XY_filtered.iloc[:, X.shape[1]:]

# # Plot correlation matrix
# correl = XY_filtered.corr(numeric_only=True)
# plt.figure(figsize=(10, 8))
# sns.heatmap(correl, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Between Input Features and Outputs')
# plt.tight_layout()
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(X_filtered, Y_filtered,
                                                    test_size=0.25,
                                                    random_state=42)

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device)
input_size = Turbulence_Parameters.num_sections
hidden_size = 128
output_size = Y_filtered.shape[1]
num_epochs = 2000
learning_rate = 2e-4
batch_size = 32

# Transform data for NN
scaler1 = MinMaxScaler(feature_range=(-1, 1))
scaler2 = MinMaxScaler(feature_range=(-1, 1))
x_train_scaled = scaler1.fit_transform(x_train)  # fit scaler1 based on input training set
x_test_scaled = scaler1.transform(x_test)
y_train_scaled = scaler2.fit_transform(y_train)  # fit scaler2 based on output training set
y_test_scaled = scaler2.transform(y_test)

model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_size, hidden_size // 2),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_size // 2, output_size)
                      ).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

###################################################################
## Training and Storing Best Model
###################################################################
# Keep the best model
best_rmse = np.inf  # init to infinity
best_epoch = -1
best_weights = None

# lists to store losses for each epoch
train_loss = []
test_loss = []
output_test_loss = np.zeros((num_epochs, output_size))

# Convert SCALED values to tensors
x_train_tens = torch.tensor(x_train_scaled, dtype=torch.float32, requires_grad=True).to(device)
x_test_tens = torch.tensor(x_test_scaled, dtype=torch.float32, requires_grad=True).to(device)
y_train_tens = torch.tensor(y_train_scaled, dtype=torch.float32, requires_grad=True).to(device)
y_test_tens = torch.tensor(y_test_scaled, dtype=torch.float32, requires_grad=True).to(device)

# Create PyTorch datasets and loaders
train_dataset = TensorDataset(x_train_tens, y_train_tens)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train model
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = torch.sqrt(criterion(y_pred, y_batch))
        loss.backward()
        optimizer.step()

        # Unscale and calculate loss
        y_pred_unscaled = torch.tensor(
            scaler2.inverse_transform(y_pred.detach().cpu().numpy()),
            dtype=torch.float32
        )
        y_batch_unscaled = torch.tensor(
            scaler2.inverse_transform(y_batch.detach().cpu().numpy()),
            dtype=torch.float32
        )
        loss_unscaled = torch.sqrt(criterion(y_pred_unscaled, y_batch_unscaled))

        epoch_train_loss += loss_unscaled.item()

    # Average RMSE UNSCALED training loss for this epoch
    epoch_train_loss /= len(train_loader)
    train_loss.append(epoch_train_loss)

    #
    # Test model
    #
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test_tens)

        # Calculate overall RMSE (unscaled)
        y_test_pred_unscaled = torch.tensor(
            scaler2.inverse_transform(y_test_pred.detach().cpu().numpy()),
            dtype=torch.float32
        )
        y_test_unscaled = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
        loss_unscaled = float(torch.sqrt(criterion(y_test_pred_unscaled, y_test_unscaled)))

        if epoch % 5 == 0:
            print(f"Test set loss for epoch {epoch}: {loss_unscaled}")

        # Calculate NORMALIZED RMSE per output
        for i in range(output_size):
            output_test_loss[epoch, i] = (
                float(torch.sqrt(criterion(y_test_pred_unscaled[:, i], y_test_unscaled[:, i])))
                / scaler2.data_range_[i]
            )

        test_loss.append(loss_unscaled)
        if loss_unscaled < best_rmse:
            best_epoch = epoch
            best_rmse = loss_unscaled
            best_weights = copy.deepcopy(model.state_dict())

# Save the best model and scalers
model_path = "./turb_spectrum_integral_section_model.pth"
torch.save(best_weights, model_path)
joblib.dump(scaler1, "./x_scaler.pkl")
joblib.dump(scaler2, "./y_scaler.pkl")

print(f"Best model and scalers saved to {model_path}")
print(f"Best Epoch: {best_epoch}, Best Test RMSE: {best_rmse} (unscaled error)")
print(f"Normalized RMSE [Grid_Re]: \n{output_test_loss[best_epoch, :]}")

#
# Plot Training vs Test Loss
#
plt.figure(figsize=(10, 8))
plt.plot(range(num_epochs), train_loss, label='Train Loss (Unscaled RMSE)')
plt.plot(range(num_epochs), test_loss, label='Test Loss (Unscaled RMSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Test Loss')
plt.legend()
plt.grid(True)
plt.show()

#
# Plot residuals
#

# Load the best model
model.load_state_dict(best_weights)

# Evaluate the best model using the test data
model.eval()
with torch.no_grad():
    y_test_pred = model(x_test_tens)

y_test_pred_unscaled = scaler2.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test_unscaled = scaler2.inverse_transform(y_test_tens.detach().cpu().numpy())

# Compute residuals
residuals = y_test_unscaled - y_test_pred_unscaled

# Create residual plots for each output
for i, target_name in enumerate(Y_filtered.columns.tolist()):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        y_test_pred_unscaled[:, i], residuals[:, i], alpha=0.7, label=f"Residuals for {target_name}"
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Residual Line")
    plt.xlabel(f"Predicted {target_name} (unscaled)")
    plt.ylabel(f"Residual {target_name} (unscaled)")
    plt.title(f"Residual Plot for {target_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
print(1)
