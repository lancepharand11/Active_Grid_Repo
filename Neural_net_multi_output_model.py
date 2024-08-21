# Multi Output Neural Network Model for Active Grid (predictions for turb int and integral length scale ONLY)
# Author: Lance Pharand, 2024
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code)
# Download the following packages below if not installed already
# !! IMPORTANT Set the Turbulence Parameters Class variables in the "Initializations" section !!

###################################################################
## Initializations and Functions
###################################################################
import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Turbulence_Parameters_class import Turbulence_Parameters

dataDir = "/Users/lancepharand/Desktop/URA_S24/Experiment_Scripts/Active_Grid_Data_and_Files/Active_Grid_Data/"
counter = 0
turb_objects = []
Turbulence_Parameters.fs = 25600 #Hz
Turbulence_Parameters.N_samples = 6144000
Turbulence_Parameters.overlap = 0.5
Turbulence_Parameters.mesh_length = 0.06096

# function to flatten 2D list to 1D list
def flatten(x):
    flatten_list = []
    for nums in x:
        for val in nums:
            flatten_list.append(val)
    return flatten_list

###################################################################
## Read in data
###################################################################

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
    temp_turb_obj = Turbulence_Parameters(filename=name, u_velo=mat_u['u'], v_velo=mat_v['v'],
                                          freestream_velo=float(name.split("_")[1]), Rossby_num=float(name.split("_")[3]),
                                          shaft_speed_std_dev=float(name.split("_")[5]))
    temp_turb_obj.calc_L_ux()
    temp_turb_obj.calc_turb_psd_spectrum()
    temp_turb_obj.calc_turb_intensity()
    turb_objects.append(temp_turb_obj)


# Load Data into data frame for access
IO_data = pd.DataFrame({"Trial Name": (turb_obj.get_trial_name() for turb_obj in turb_objects),
                        "Grid Re": (turb_obj.get_grid_Re() for turb_obj in turb_objects),
                        "Rossby Number": (turb_obj.get_Rossby_num() for turb_obj in turb_objects),
                        "Shaft Speed Standard Deviation * M / u_inf": (turb_obj.get_shaft_speed_std_dev() for turb_obj in turb_objects),
                        "Turbulence Intensity": (turb_obj.get_turb_int() for turb_obj in turb_objects),
                        "L_ux * M": (turb_obj.get_L_ux_non_dim() for turb_obj in turb_objects),
                        "k*E_11 / var(u) [Non-Dim PSD]": (turb_obj.get_E_u().tolist() for turb_obj in turb_objects),
                        "k*M [Non-Dim Freq]": (turb_obj.get_freq_non_dim().tolist() for turb_obj in turb_objects)
                        })

###################################################################
## Preprocessing and Model Setup
###################################################################
from sklearn.model_selection import train_test_split
from torch import nn, optim
import copy
import torch
from sklearn.preprocessing import MinMaxScaler

# Split the data. NOTE: Using only turb intensity & integral length scale for the y
X = IO_data.iloc[:, 1:4]
Y = IO_data.iloc[:, 4:6]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    stratify=X["Grid Re"], random_state=42)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

device = torch.device(device)
input_size = 3
hidden_size = 128
output_size = 2
num_epochs = 500
learning_rate = 0.0001
train_running_loss = 0
test_running_loss = 0
test_loss = []

# Transform data for NN
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
x_train_scaled = scaler1.fit_transform(x_train) # fit scaler1 based on input training set
x_test_scaled = scaler1.transform(x_test)
y_train_scaled = scaler2.fit_transform(y_train) # fit scaler2 based on output training set
y_test_scaled = scaler2.transform(y_test)

# Setup model. And define loss and optimizer
model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, hidden_size // 2),
                      nn.ReLU(),
                      nn.Linear(hidden_size // 2, hidden_size // 4),
                      nn.ReLU(),
                      nn.Linear(hidden_size // 4, output_size)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###################################################################
## Training and Storing Best Model
###################################################################
# Keep the best model
best_rmse = np.inf # init to infinity
best_epoch = -1
best_weights = None

# Convert SCALED values to tensors
x_train_tens = torch.tensor(x_train_scaled, dtype=torch.float32, requires_grad=True).to(device)
x_test_tens = torch.tensor(x_test_scaled, dtype=torch.float32, requires_grad=True).to(device)
y_train_tens = torch.tensor(y_train_scaled, dtype=torch.float32, requires_grad=True).to(device)
y_test_tens = torch.tensor(y_test_scaled, dtype=torch.float32, requires_grad=True).to(device)

# Train model
for epoch in range(num_epochs):
    model.train()
    for batchid, x in enumerate(x_train_tens):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = torch.sqrt(criterion(y_pred, y_train_tens[batchid, :]))  # NOTE: SCALED RMSE

        loss.backward()
        optimizer.step()
        # train_running_loss += loss.item()

    # Test model with gradient updates off
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test_tens)
        loss_unscaled = float(torch.sqrt(criterion(torch.tensor(
            scaler2.inverse_transform(y_test_pred.detach().cpu().numpy())),
                      torch.tensor(y_test.to_numpy()))))  # RMSE; UNSCALED
        # test_running_loss += loss_unscaled
        test_loss.append(loss_unscaled)
        if loss_unscaled < best_rmse:
            best_epoch = epoch
            best_rmse = loss_unscaled
            best_weights = copy.deepcopy(model.state_dict())

print(f"Best Epoch: {best_epoch}, Best Test RMSE: {best_rmse} (unscaled error)")

# print(f'Average Training loss: {train_running_loss / (len(x_train) * num_epochs)}')
# print(f'Average Testing loss: {test_running_loss / (len(x_test) * num_epochs)}')

# Plot the test set loss over the epochs
plt.plot(test_loss, color='b', label='Test Loss')
# plt.plot(train_loss, color='r', label='Training Loss')
# plt.ylim([0, 10])
plt.ylabel("Loss per Epoch")
plt.xlabel("Epoch Number")
plt.title("Test Set - Loss Plot")
plt.show()

# Load the best model
model.load_state_dict(best_weights)

# Evaluate the best model using the test data
model.eval()
with torch.no_grad():
    y_test_pred = model(x_test_tens)

# Plot residuals
residuals_turb_int = []
residuals_integral_length = []
for i in range(y_test_pred.shape[0]):
    residuals_turb_int.append((y_test_tens[i, 0] - y_test_pred[i, 0]).detach().cpu().numpy().item())
    residuals_integral_length.append((y_test_tens[i, 1] - y_test_pred[i, 1]).detach().cpu().numpy().item())

# scaler2.inverse_transform(flatten(y_test_pred[:, 0].view(-1, 1).detach().cpu().numpy().tolist()))
plt1 = pd.DataFrame({
    'Turbulence Intensity - Predicted Values': scaler2.inverse_transform(y_test_pred.detach().cpu().numpy())[:, 0],
    'Turbulence Intensity - Residuals': residuals_turb_int
})

# scaler2.inverse_transform(flatten(y_test_pred[:, 1].view(-1, 1).detach().cpu().numpy().tolist()))
plt2 = pd.DataFrame({
    'Integral Length Scale - Predicted Values': scaler2.inverse_transform(y_test_pred.detach().cpu().numpy())[:, 1],
    'Integral Length Scale - Residuals': residuals_integral_length
})

fig, ax = plt.subplots(ncols=2, figsize=(10, 8))
sns.regplot(x='Turbulence Intensity - Predicted Values', y='Turbulence Intensity - Residuals', data=plt1, ax=ax[0],
            ci=None, color='b')
# plt.xlabel('Turbulence Intensity - Predicted Values')
# plt.ylabel('Turbulence Intensity - Residuals')
sns.regplot(x='Integral Length Scale - Predicted Values', y='Integral Length Scale - Residuals', data=plt2, ax=ax[1],
            ci=None, color='r')
# plt.xlabel('Integral Length Scale - Predicted Values')
# plt.ylabel('Integral Length Scale - Residuals')
plt.tight_layout()
plt.show()
