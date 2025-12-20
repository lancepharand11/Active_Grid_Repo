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
from torch.utils.data import DataLoader, TensorDataset
import joblib

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from Turbulence_Parameters_class import Turbulence_Parameters
from dataOverviewPlot import dataOverviewPlot
from shaftSpeedStdPlot import shaftSpeedStdPlot
from inputOutputHeatmapPlot import inputOutputHeatmapPlot

import seaborn as sns

# %%

dataDir = Path("/Users/ctoppings/Nextcloud/Experimental Data/Active_Grid_Data_Lance/Selected Data/")
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
    temp_turb_obj.calc_L_ux_Uncertainty()
    temp_turb_obj.calc_turb_intensity()
    temp_turb_obj.calc_turb_int_uncertainty()
    temp_turb_obj.calc_Re_lambda()
    temp_turb_obj.calc_dissipation_rate()
    temp_turb_obj.calc_anisotropy()
    # temp_turb_obj.psd_breakaway_freq_inertial()
    # temp_turb_obj.psd_breakaway_freq_dissip()
    # temp_turb_obj.psd_inertial_range_slope()
    temp_turb_obj.psd_integral_sectioning()
    turb_objects.append(temp_turb_obj)
    
    print(f"Loaded file {counter}: {file.stem}")

IO_data = pd.DataFrame({"Trial Name": (turb_obj.get_trial_name() for turb_obj in turb_objects),
                        "Grid Re": (turb_obj.get_grid_Re() for turb_obj in turb_objects),
                        "Rossby Number": (turb_obj.get_Rossby_num() for turb_obj in turb_objects),
                        "Shaft Speed Standard Deviation * M / u_inf": (turb_obj.get_shaft_speed_std_dev() for turb_obj in turb_objects),
                        "Turbulence Intensity": (turb_obj.get_turb_int() for turb_obj in turb_objects),
                        "Turbulence Intensity Uncertainty": (turb_obj.get_turb_int_uncertainty() for turb_obj in turb_objects),
                        "L_ux / M": (turb_obj.get_L_ux_non_dim() for turb_obj in turb_objects),
                        "L_ux Uncertainty": (turb_obj.get_L_ux_uncertainty() for turb_obj in turb_objects),
                        "E_11 / (M * U) [Non-Dim PSD]": (turb_obj.get_E_u().tolist() for turb_obj in turb_objects),
                        "Freq * M / U [Non-Dim Freq]": (turb_obj.get_freq_non_dim().tolist() for turb_obj in turb_objects),
                        "Log(E_11 / (M * U))": (turb_obj.get_log_E_u() for turb_obj in turb_objects),
                        "Log(Freq * M / U)": (turb_obj.get_log_freq_non_dim() for turb_obj in turb_objects),
                        "Anisotropy": (turb_obj.get_anisotropy() for turb_obj in turb_objects),
                        "Re_lambda": (turb_obj.get_Re_lambda() for turb_obj in turb_objects),
                        "Epsilon": (turb_obj.get_epsilon() for turb_obj in turb_objects)
                        })

# Normalise the Shaft Speed Standard Deviation by the viscous time scale
IO_data["Shaft Speed Standard Deviation * M^2 / nu"] = np.multiply(IO_data["Shaft Speed Standard Deviation * M / u_inf"], IO_data["Grid Re"])

IO_data_file_path = "./DataSummaryOutliersRemoved.csv"
IO_data_to_save = IO_data[["Trial Name", "Grid Re", "Rossby Number",
                 "Shaft Speed Standard Deviation * M / u_inf",
                 "Shaft Speed Standard Deviation * M^2 / nu",
                 "Turbulence Intensity",
                 "Turbulence Intensity Uncertainty", "L_ux / M",
                 "L_ux Uncertainty","Anisotropy",
                 "Re_lambda",
                 "Epsilon"]]
# Save the DataFrame to a CSV file
IO_data_to_save.to_csv(IO_data_file_path)

###################################################################
# %% Scatter Plot of Dataset Matrix and Comparison with Previous Data
###################################################################

# Load the CSV file into a DataFrame
IO_data = pd.read_csv(IO_data_file_path)

dataComparisonFigure, dataOverviewFigure = dataOverviewPlot(IO_data)
dataComparisonFigure_FileName = "../Figures/dataComparison.eps"
dataOverviewFigure_FileName = "../Figures/dataOverview.eps"
dataOverviewFigure.savefig(dataOverviewFigure_FileName,format="eps")
dataComparisonFigure.savefig(dataComparisonFigure_FileName,format="eps")

###################################################################
# %% Effect of Shaft Speed Standard Deviation
###################################################################

shaftSpeedStdFigure = shaftSpeedStdPlot(IO_data)
shaftSpeedStdFigure_FileName = "../Figures/shaftSpeedStd.eps"
shaftSpeedStdFigure.savefig(shaftSpeedStdFigure_FileName,format="eps")

###################################################################
# %% Model Input and Output Heatmap
###################################################################

inputOutputHeatmapFigure = inputOutputHeatmapPlot(IO_data)
inputOutputHeatmapFigure_FileName = "../Figures/inputOutputHeatmap.eps"
inputOutputHeatmapFigure.savefig(inputOutputHeatmapFigure_FileName,format="eps")

