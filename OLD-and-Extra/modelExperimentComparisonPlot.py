# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:06:30 2025

@author: Connor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mkr
import sys
import os
sys.path.insert(0, os.path.abspath('../Turb-int-and-Integral-Length-Scale-Model'))
from IntensityLengthModelClass import IntensityLengthModel

# Load the CSV file into a DataFrame
IO_data_file_path = "./DataSummary.csv"
IO_data = pd.read_csv(IO_data_file_path)

# %% Load the NN Model

modelPath = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/best_model_20250518_113100.pth"
# Load the scalers
scaler1Path = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_x_20250518_113100.pkl"
scaler2Path = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_y_20250518_113100.pkl"

NNModel = IntensityLengthModel(modelPath, scaler1Path, scaler2Path)

Ro_Re_M_Const = np.linspace(5, 75, num=100)
Re_M_Re_M_Const = np.ones(Ro_Re_M_Const.shape)*30000
sigma = np.ones(Ro_Re_M_Const.shape)*0.0075

Tu_Re_M_Const, L_ux_Re_M_Const = NNModel.evaluate(Re_M_Re_M_Const, Ro_Re_M_Const, sigma)

Re_M_Ro_Const = np.linspace(5000, 50000, num=100)
Ro_Ro_Const = np.ones(Re_M_Ro_Const.shape)*25
sigma = np.ones(Re_M_Ro_Const.shape)*0.0075

Tu_Ro_Const, L_ux_Ro_Const = NNModel.evaluate(Re_M_Ro_Const, Ro_Ro_Const, sigma)

# %% Plot the experimental data

Re_M_Const_Data = IO_data[(IO_data["Grid Re"] > 25000) & (IO_data["Grid Re"] < 35000) & (IO_data["Shaft Speed Standard Deviation * M / u_inf"]*IO_data["Grid Re"] > 100) & (IO_data["Shaft Speed Standard Deviation * M / u_inf"]*IO_data["Grid Re"] < 300)]
Re_M_Const_Data["Turbulence Intensity Percent"] = Re_M_Const_Data["Turbulence Intensity"]*100

Ro_Const_Data = IO_data[(IO_data["Rossby Number"] == 25) & (IO_data["Shaft Speed Standard Deviation * M / u_inf"]*IO_data["Grid Re"] > 100) & (IO_data["Shaft Speed Standard Deviation * M / u_inf"]*IO_data["Grid Re"] < 300)]
Ro_Const_Data["Turbulence Intensity Percent"] = Ro_Const_Data["Turbulence Intensity"]*100

# Create figure with 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(6.375, 6.375*2/3))
axs = axs.flatten()

# Marker Style
present_marker = mkr.MarkerStyle('^',fillstyle="none")

# Tu vs Ro
Re_M_Const_Data.plot(kind="scatter", x="Rossby Number", y="Turbulence Intensity Percent", ax=axs[0], label='Measurement', marker=present_marker, c="k")
axs[0].plot(Ro_Re_M_Const,Tu_Re_M_Const*100, label='NN Prediction')
axs[0].set_xlabel(r"$\textrm{Ro}$")
axs[0].set_ylabel('$Tu$')
axs[0].get_legend().remove()
axs[0].set_title(r"$\textrm{Re}_M=3\times10^5$")
fig.legend(loc='outside upper center')

# Tu vs Re_M
Ro_Const_Data.plot(kind="scatter", x="Grid Re", y="Turbulence Intensity Percent", ax=axs[1], label='Measurement', legend=False, marker=present_marker, c="k")
axs[1].plot(Re_M_Ro_Const,Tu_Ro_Const*100, label='NN Prediction')
axs[1].set_ylabel('$Tu$')
axs[1].set_xlabel(r"$\textrm{Re}_M$")
axs[1].set_title(r"$\textrm{Ro}=25$")

# L_ux vs Ro
Re_M_Const_Data.plot(kind="scatter", x="Rossby Number", y="L_ux / M", ax=axs[2], label='Measurement', legend=False, marker=present_marker, c="k")
axs[2].plot(Ro_Re_M_Const,L_ux_Re_M_Const, label='NN Prediction')
axs[2].set_ylabel('$L_{ux}/M$')
axs[2].set_xlabel(r"$\textrm{Ro}$")

# L_ux vs Re_M
Ro_Const_Data.plot(kind="scatter", x="Grid Re", y="L_ux / M", ax=axs[3], label='Measurement', legend=False, marker=present_marker, c="k")
axs[3].plot(Re_M_Ro_Const,L_ux_Ro_Const, label='NN Prediction')
axs[3].set_ylabel('$L_{ux}/M$')
axs[3].set_xlabel(r"$\textrm{Re}_M$")

# X-Axis Limits
for x in [1,3]:
    axs[x].set_xlim(left=0, right=75000)
    
for x in [0,2]:
    axs[x].set_xlim(left=0, right=100)
    
# Y-Axis Limits
for x in [0,1]:
    axs[x].set_ylim(bottom=10, top=14)
    
for x in [2,3]:
    axs[x].set_ylim(bottom=0, top=6)
    

# Adjust layout
plt.subplots_adjust(top=0.78, hspace=0.45, wspace=0.4)

plt.show()

# Print
modelExperimentComparisonFigure_FileName = "../Figures/modelExperimentComparison.eps"
fig.savefig(modelExperimentComparisonFigure_FileName,format="eps")
