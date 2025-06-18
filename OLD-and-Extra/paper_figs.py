# Figures for Active Grid Paper
# Author: Lance Pharand, 2024
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code in an academic application)
# Download the following packages below if not installed already
# !! IMPORTANT Set the Turbulence Parameters Class variables in the "Initializations" section !!
# These are only a portion of the figures seen in the paper

import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from Turbulence_Parameters_class import Turbulence_Parameters
from torch.utils.data import DataLoader, TensorDataset
import joblib
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

dataDir = Path("/Users/Connor/Nextcloud/Experimental Data/Active_Grid_Data_Lance/")
counter = 0
turb_objects = []
Turbulence_Parameters.fs = 25600 #Hz
Turbulence_Parameters.N_samples = 6144000
Turbulence_Parameters.overlap = 0.5
Turbulence_Parameters.mesh_length = 0.06096
Turbulence_Parameters.num_sections = 4

for file in list(dataDir.glob('*.mat')):
    if counter == 0:
        time_stamps = scipy.io.loadmat(file, variable_names=['timeStamps'], squeeze_me=True, mat_dtype=True)
        counter += 1

    Ro_string = file.stem.split("_")[3]
    if Ro_string == '-':
        continue
    file_Ro = float(Ro_string)
    file_shaftSpeedSTD = float(file.stem.split("_")[5])
    mat_u = scipy.io.loadmat(file, variable_names=['u'], squeeze_me=True, mat_dtype=True)
    mat_v = scipy.io.loadmat(file, variable_names=['v'], squeeze_me=True, mat_dtype=True)
    temp_turb_obj = Turbulence_Parameters(filename=file.stem, u_velo=mat_u['u'][4000000:], v_velo=mat_v['v'][4000000:],
                                          freestream_velo=np.mean(mat_u['u'][4000000:]), Rossby_num=file_Ro,
                                          shaft_speed_std_dev=file_shaftSpeedSTD)
    
    temp_turb_obj.filter_velo()
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


col_labels = ["Integral section " + str(s) for s in range(Turbulence_Parameters.num_sections)]
turb_sections = pd.DataFrame(IO_data.iloc[:, 10].to_list(), columns=col_labels)

# Select core variables
core_cols = [
    "Grid Re",
    "Rossby Number",
    "Shaft Speed Standard Deviation * M / u_inf",
    "Turbulence Intensity",
    "L_ux / M"
]

# Combine and compute statistics
df_all = pd.concat([IO_data[core_cols], turb_sections], axis=1)
stats = df_all.agg(['mean', 'std', 'min', 'median', 'max']).T
stats.index.name = 'Variable'
stats.rename_axis(columns='Statistic', inplace=True)
print(stats.to_markdown())

###################################################################
## Preprocessing and Model Setup
###################################################################
from scipy import stats

# Split the data and combine it. NOTE: Using only turb intensity & integral length scale for the y
X = IO_data.iloc[:, 1:4]
Y = IO_data.iloc[:, 4:6]
XY = pd.concat([X, Y], axis=1)

# Compute z-scores for each column
z_scores = np.abs(stats.zscore(XY, nan_policy='omit'))
threshold = 3  # Threshold z-score
rows_with_outlier = (z_scores > threshold).any(axis=1)
XY_filtered = XY[~rows_with_outlier]

# Split back into X and Y
X_filtered = XY_filtered.iloc[:, :X.shape[1]]
Y_filtered = XY_filtered.iloc[:, X.shape[1]:]

#######################
# 3D scatter plots
#######################
x = X_filtered.iloc[:, 0].values  # Grid Re
y = X_filtered.iloc[:, 1].values  # Rossby Number
z = X_filtered.iloc[:, 2].values  # Shaft Speed Std Dev

y1 = Y_filtered.iloc[:, 0].values  # turb intensity
y2 = Y_filtered.iloc[:, 1].values  # int length scale

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
p1 = ax1.scatter(x, y, z, c=y1, cmap='magma',
                 marker='o', s=50, alpha=0.8
                 )
cbar1 = fig1.colorbar(p1, ax=ax1, shrink=0.5, pad=0.1)
cbar1.set_label('Turbulence Intensity')
ax1.set_xlabel('Grid Re', labelpad=7)
ax1.set_ylabel('Rossby Number')
ax1.set_zlabel('Shaft Speed Std Dev * M / u_inf', labelpad=8, rotation=0)
ax1.set_title('3D Scatter: Turbulence Intensity')

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')
p2 = ax2.scatter(x, y, z, c=y2, cmap='viridis',
                 marker='^', s=50, alpha=0.8
                 )
cbar2 = fig2.colorbar(p2, ax=ax2, shrink=0.5, pad=0.1)
cbar2.set_label('L_ux / M')
ax2.set_xlabel('Grid Re', labelpad=7)
ax2.set_ylabel('Rossby Number')
ax2.set_zlabel('Shaft Speed Std Dev * M / u_inf', labelpad=8, rotation=0)
ax2.set_title('3D Scatter: Integral Length Scale')
plt.show()

# #######################
# # Scatter Plot of Dataset Matrix
# #######################
# import seaborn as sns
#
# sns.pairplot(XY_filtered)
# plt.show()


# #######################
# # Correlation bar graph
# #######################
# XY_corr = XY_filtered.corr()
# x_cols = X_filtered.columns.tolist()
# y_cols = Y_filtered.columns.tolist()
# corr_XY = XY_corr.loc[x_cols, y_cols]
#
# n_x = len(x_cols)
# n_y = len(y_cols)
# ind = np.arange(n_x)
# width = 0.8 / n_y  # total width 0.8 divided among Y-bars. Leaves a 0.2 gap between features in the graph
#
# fig, ax = plt.subplots(figsize=(10, 8))
# for i, y in enumerate(y_cols):
#     ax.bar(ind + i * width, corr_XY[y].values, width, label=y)
#
# ax.set_xticks(ind + width * (n_y - 1) / 2)
# ax.set_xticklabels(x_cols, rotation=30, ha='right')
# ax.set_ylabel("Pearson r")
# ax.set_title("Correlation of each X-feature with Y-labels")
# ax.legend(title="Y variable", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
