# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:06:30 2025

@author: Connor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mkr
plt.rcParams['text.usetex'] = True
import sys
import os
sys.path.insert(0, os.path.abspath('../Turb-int-and-Integral-Length-Scale-Model'))
from IntensityLengthModelClass import IntensityLengthModel

# Load the CSV file into a DataFrame
IO_data_file_path = "./DataSummary.csv"
IO_data = pd.read_csv(IO_data_file_path)

# %% Load the Model
modelID = "20250708_085121"
modelPath = f"../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/best_model_{modelID}.pth"
# Load the scalers
scaler1Path = f"../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_x_{modelID}.pkl"
scaler2Path = f"../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_y_{modelID}.pkl"

Model = IntensityLengthModel(modelPath, scaler1Path, scaler2Path)

# %% Load the Indices of the validation set

# val_idx = np.loadtxt(f"../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/val_idx_{modelID}.csv", delimiter=',')
# train_idx = np.loadtxt(f"../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/train_idx_{modelID}.csv", delimiter=',')

# Evaluate the model for Re_M = 20000
Ro_Re_M_Const_20000 = np.linspace(5, 75, num=100)
Re_M_Re_M_Const_20000 = np.ones(Ro_Re_M_Const_20000.shape)*20000
sigma = np.ones(Ro_Re_M_Const_20000.shape)*0.0075

Tu_Re_M_Const_20000, L_ux_Re_M_Const_20000 = Model.evaluate(Re_M_Re_M_Const_20000, Ro_Re_M_Const_20000, sigma)

# Evaluate the model for Re_M = 30000
Ro_Re_M_Const_30000 = np.linspace(5, 75, num=100)
Re_M_Re_M_Const_30000 = np.ones(Ro_Re_M_Const_30000.shape)*30000
sigma = np.ones(Ro_Re_M_Const_30000.shape)*0.0075

Tu_Re_M_Const_30000, L_ux_Re_M_Const_30000 = Model.evaluate(Re_M_Re_M_Const_30000, Ro_Re_M_Const_30000, sigma)

# Evaluate the model for Re_M = 40000
Ro_Re_M_Const_40000 = np.linspace(5, 75, num=100)
Re_M_Re_M_Const_40000 = np.ones(Ro_Re_M_Const_40000.shape)*40000
sigma = np.ones(Ro_Re_M_Const_40000.shape)*0.0075

Tu_Re_M_Const_40000, L_ux_Re_M_Const_40000 = Model.evaluate(Re_M_Re_M_Const_40000, Ro_Re_M_Const_40000, sigma)

# Evaluate the model for Ro = 15
Re_M_Ro_Const_15 = np.linspace(5000, 50000, num=100)
Ro_Ro_Const_15 = np.ones(Re_M_Ro_Const_15.shape)*15
sigma = np.ones(Re_M_Ro_Const_15.shape)*0.0075

Tu_Ro_Const_15, L_ux_Ro_Const_15 = Model.evaluate(Re_M_Ro_Const_15, Ro_Ro_Const_15, sigma)

# Evaluate the model for Ro = 25
Re_M_Ro_Const_25 = np.linspace(5000, 50000, num=100)
Ro_Ro_Const_25 = np.ones(Re_M_Ro_Const_25.shape)*25
sigma = np.ones(Re_M_Ro_Const_25.shape)*0.0075

Tu_Ro_Const_25, L_ux_Ro_Const_25 = Model.evaluate(Re_M_Ro_Const_25, Ro_Ro_Const_25, sigma)

# Evaluate the model for Ro = 40
Re_M_Ro_Const_40 = np.linspace(5000, 50000, num=100)
Ro_Ro_Const_40 = np.ones(Re_M_Ro_Const_40.shape)*40
sigma = np.ones(Re_M_Ro_Const_40.shape)*0.0075
Tu_Ro_Const_40, L_ux_Ro_Const_40 = Model.evaluate(Re_M_Ro_Const_40, Ro_Ro_Const_40, sigma)

# %% Select the experimental data

shaft_speed_product = IO_data["Shaft Speed Standard Deviation * M / u_inf"] * IO_data["Grid Re"]

# Boolean masks for selecting data based on conditions
constant_re_30000 = (IO_data["Grid Re"] > 25000) & (IO_data["Grid Re"] < 35000)
constant_re_40000 = (IO_data["Grid Re"] > 35000) & (IO_data["Grid Re"] < 45000)
constant_re_20000 = (IO_data["Grid Re"] > 15000) & (IO_data["Grid Re"] < 25000)
constant_ro_15 = (IO_data["Rossby Number"] == 15)
constant_ro_25 = (IO_data["Rossby Number"] == 25)
constant_ro_40 = (IO_data["Rossby Number"] == 40)
constant_sigma_200 = (shaft_speed_product > 100) & (shaft_speed_product < 300)

# For Re_M = 30000
Re_M_Const_30000_Exp = IO_data[constant_re_30000 & constant_sigma_200].copy()
Re_M_Const_30000_Exp["Turbulence Intensity Percent"] = Re_M_Const_30000_Exp["Turbulence Intensity"]*100
Re_M_Const_30000_Exp["Turbulence Intensity Uncertainty Percent"] = Re_M_Const_30000_Exp["Turbulence Intensity Uncertainty"]*100

# For Re_M = 40000
Re_M_Const_40000_Exp = IO_data[constant_re_40000 & constant_sigma_200].copy()
Re_M_Const_40000_Exp["Turbulence Intensity Percent"] = Re_M_Const_40000_Exp["Turbulence Intensity"]*100
Re_M_Const_40000_Exp["Turbulence Intensity Uncertainty Percent"] = Re_M_Const_40000_Exp["Turbulence Intensity Uncertainty"]*100

# For Re_M = 20000
Re_M_Const_20000_Exp = IO_data[constant_re_20000 & constant_sigma_200].copy()
Re_M_Const_20000_Exp["Turbulence Intensity Percent"] = Re_M_Const_20000_Exp["Turbulence Intensity"]*100
Re_M_Const_20000_Exp["Turbulence Intensity Uncertainty Percent"] = Re_M_Const_20000_Exp["Turbulence Intensity Uncertainty"]*100

# For Ro = 25
Ro_Const_25_Exp = IO_data[constant_ro_25 & constant_sigma_200].copy()
Ro_Const_25_Exp["Turbulence Intensity Percent"] = Ro_Const_25_Exp["Turbulence Intensity"]*100
Ro_Const_25_Exp["Turbulence Intensity Uncertainty Percent"] = Ro_Const_25_Exp["Turbulence Intensity Uncertainty"]*100

# For Ro = 15
Ro_Const_15_Exp = IO_data[constant_ro_15 & constant_sigma_200].copy()
Ro_Const_15_Exp["Turbulence Intensity Percent"] = Ro_Const_15_Exp["Turbulence Intensity"]*100
Ro_Const_15_Exp["Turbulence Intensity Uncertainty Percent"] = Ro_Const_15_Exp["Turbulence Intensity Uncertainty"]*100

# For Ro = 40
Ro_Const_40_Exp = IO_data[constant_ro_40 & constant_sigma_200].copy()
Ro_Const_40_Exp["Turbulence Intensity Percent"] = Ro_Const_40_Exp["Turbulence Intensity"]*100
Ro_Const_40_Exp["Turbulence Intensity Uncertainty Percent"] = Ro_Const_40_Exp["Turbulence Intensity Uncertainty"]*100

# %% Plot the data

# Create figure with 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(6.375, 6.375*2/3))
axs = axs.flatten()

# Marker Style
present_marker = mkr.MarkerStyle('^',fillstyle="none")

# Tu vs Ro
axs[0].errorbar(Re_M_Const_20000_Exp["Rossby Number"],
               Re_M_Const_20000_Exp["Turbulence Intensity Percent"],
               label=r'$\textrm{Re}_M=2\times10^5$',
               yerr=Re_M_Const_20000_Exp["Turbulence Intensity Uncertainty Percent"],
               marker=present_marker, mec="b", ecolor="b", linestyle="none",
               capsize=2, lw = 1)
axs[0].errorbar(Re_M_Const_30000_Exp["Rossby Number"],
               Re_M_Const_30000_Exp["Turbulence Intensity Percent"],
               label=r'$\textrm{Re}_M=3\times10^5$',
               yerr=Re_M_Const_30000_Exp["Turbulence Intensity Uncertainty Percent"],
               marker=present_marker, mec="k", ecolor="k", linestyle="none",
               capsize=2, lw = 1)
axs[0].errorbar(Re_M_Const_40000_Exp["Rossby Number"],
               Re_M_Const_40000_Exp["Turbulence Intensity Percent"],
               label=r'$\textrm{Re}_M=4\times10^5$',
               yerr=Re_M_Const_40000_Exp["Turbulence Intensity Uncertainty Percent"],
               marker=present_marker, mec="r", ecolor="r", linestyle="none",
               capsize=2, lw = 1)

axs[0].plot(Ro_Re_M_Const_20000, Tu_Re_M_Const_20000*100, c="b", lw = 1)
axs[0].plot(Ro_Re_M_Const_30000, Tu_Re_M_Const_30000*100, c="k", lw = 1)
axs[0].plot(Ro_Re_M_Const_40000, Tu_Re_M_Const_40000*100, c="r", lw = 1)
axs[0].set_xlabel(r"$\textrm{Ro}$")
axs[0].set_ylabel('$Tu$ [\%]')
axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)
#axs[0].set_title(r"$\\textrm{Re}_M=3\\times10^5$")

# Tu vs Re_M
axs[1].errorbar(Ro_Const_15_Exp["Grid Re"],
               Ro_Const_15_Exp["Turbulence Intensity Percent"],
               label=r'$\textrm{Ro}=15$',
               yerr=Ro_Const_15_Exp["Turbulence Intensity Uncertainty Percent"],
               marker=present_marker, mec="b", ecolor="b", linestyle="none",
               capsize=2, lw = 1)
axs[1].errorbar(Ro_Const_25_Exp["Grid Re"],
               Ro_Const_25_Exp["Turbulence Intensity Percent"],
               label=r'$\textrm{Ro}=25$',
               yerr=Ro_Const_25_Exp["Turbulence Intensity Uncertainty Percent"],
               marker=present_marker, mec="k", ecolor="k", linestyle="none",
               capsize=2, lw = 1)
axs[1].errorbar(Ro_Const_40_Exp["Grid Re"],
               Ro_Const_40_Exp["Turbulence Intensity Percent"],
               label=r'$\textrm{Ro}=40$',
               yerr=Ro_Const_40_Exp["Turbulence Intensity Uncertainty Percent"],
               marker=present_marker, mec="r", ecolor="r", linestyle="none",
               capsize=2, lw = 1)
axs[1].plot(Re_M_Ro_Const_15,Tu_Ro_Const_15*100, c="b", lw = 1)
axs[1].plot(Re_M_Ro_Const_25,Tu_Ro_Const_25*100, c="k", lw = 1)
axs[1].plot(Re_M_Ro_Const_40,Tu_Ro_Const_40*100, c="r", lw = 1)
axs[1].set_ylabel('$Tu$ [\%]')
axs[1].set_xlabel(r"$\textrm{Re}_M$")
axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)
#axs[1].set_title(r"$\\textrm{Ro}=25$")

# L_ux vs Ro
axs[2].errorbar(Re_M_Const_20000_Exp["Rossby Number"],
               Re_M_Const_20000_Exp["L_ux / M"],
               yerr=Re_M_Const_20000_Exp["L_ux Uncertainty"],
               marker=present_marker, mec="b", ecolor="b", linestyle="none",
               capsize=2, lw = 1)
axs[2].errorbar(Re_M_Const_30000_Exp["Rossby Number"],
               Re_M_Const_30000_Exp["L_ux / M"],
               yerr=Re_M_Const_30000_Exp["L_ux Uncertainty"],
               marker=present_marker, mec="k", ecolor="k", linestyle="none",
               capsize=2, lw = 1)
axs[2].errorbar(Re_M_Const_40000_Exp["Rossby Number"],
               Re_M_Const_40000_Exp["L_ux / M"],
               yerr=Re_M_Const_40000_Exp["L_ux Uncertainty"],
               marker=present_marker, mec="r", ecolor="r", linestyle="none",
               capsize=2, lw = 1)

axs[2].plot(Ro_Re_M_Const_20000, L_ux_Re_M_Const_20000, c="b", lw = 1)
axs[2].plot(Ro_Re_M_Const_30000, L_ux_Re_M_Const_30000, c="k", lw = 1)
axs[2].plot(Ro_Re_M_Const_40000, L_ux_Re_M_Const_40000, c="r", lw = 1)
axs[2].set_xlabel(r"$\textrm{Ro}$")
axs[2].set_ylabel(r'$L_{ux}/M$')
#axs[0].set_title(r"$\\textrm{Re}_M=3\\times10^5$")

# L_ux vs Re_M
axs[3].errorbar(Ro_Const_15_Exp["Grid Re"],
               Ro_Const_15_Exp["L_ux / M"],
               yerr=Ro_Const_15_Exp["L_ux Uncertainty"],
               marker=present_marker, mec="b", ecolor="b", linestyle="none",
               capsize=2, lw = 1)
axs[3].errorbar(Ro_Const_25_Exp["Grid Re"],
               Ro_Const_25_Exp["L_ux / M"],
               yerr=Ro_Const_25_Exp["L_ux Uncertainty"],
               marker=present_marker, mec="k", ecolor="k", linestyle="none",
               capsize=2, lw = 1)
axs[3].errorbar(Ro_Const_40_Exp["Grid Re"],
               Ro_Const_40_Exp["L_ux / M"],
               yerr=Ro_Const_40_Exp["L_ux Uncertainty"],
               marker=present_marker, mec="r", ecolor="r", linestyle="none",
               capsize=2, lw = 1)
axs[3].plot(Re_M_Ro_Const_15, L_ux_Ro_Const_15, c="b", lw = 1)
axs[3].plot(Re_M_Ro_Const_25, L_ux_Ro_Const_25, c="k", lw = 1)
axs[3].plot(Re_M_Ro_Const_40, L_ux_Ro_Const_40, c="r", lw = 1)
axs[3].set_ylabel(r'$L_{ux}/M$')
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
