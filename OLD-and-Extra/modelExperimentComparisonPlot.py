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
import IntensityLengthModelClass
import IntensityLengthPolynomialModelClass

# Text formatting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10;


# Load the CSV file into a DataFrame
IO_data_file_path = "./DataSummaryOutliersRemoved.csv"
IO_data = pd.read_csv(IO_data_file_path)

# %% Neural Network Model
nnModelPath = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/best_model_20260112_171956.pth"
# Load the scalers
nnScaler1Path = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_x_20260112_171956.pkl"
nnScaler2Path = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_y_20260112_171956.pkl"

# %% Polynomial Model
polyModelPath = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/best_model_20260112_172128.pth"
# Load the scalers
polyScaler1Path = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_x_20260112_172128.pkl"
polyScaler2Path = "../Turb-int-and-Integral-Length-Scale-Model/Models_and_Results/scaler_y_20260112_172128.pkl"

# %%
nnModel = IntensityLengthModelClass.IntensityLengthModel(nnModelPath, nnScaler1Path, nnScaler2Path,1)
polyModel = IntensityLengthPolynomialModelClass.IntensityLengthModel(polyModelPath, polyScaler1Path, polyScaler2Path)

# %% Evaluate the model for the constant sigma plot

# Choose Shaft Speed Standard Deviation * M^2 / nu
sigma_value = 620.17

# Evaluate the model for Re_M = 20000
Ro_Re_M_Const_20000 = np.linspace(5, 75, num=100)
Re_M_Re_M_Const_20000 = np.ones(Ro_Re_M_Const_20000.shape)*20000
sigma = np.ones(Ro_Re_M_Const_20000.shape)*sigma_value

Tu_Re_M_Const_20000_NN, L_ux_Re_M_Const_20000_NN = nnModel.evaluate(Re_M_Re_M_Const_20000, Ro_Re_M_Const_20000, sigma)
Tu_Re_M_Const_20000_Poly, L_ux_Re_M_Const_20000_Poly = polyModel.evaluate(Re_M_Re_M_Const_20000, Ro_Re_M_Const_20000, sigma)

# Evaluate the model for Re_M = 30000
Ro_Re_M_Const_30000 = np.linspace(5, 75, num=100)
Re_M_Re_M_Const_30000 = np.ones(Ro_Re_M_Const_30000.shape)*30000
sigma = np.ones(Ro_Re_M_Const_30000.shape)*sigma_value

Tu_Re_M_Const_30000_NN, L_ux_Re_M_Const_30000_NN = nnModel.evaluate(Re_M_Re_M_Const_30000, Ro_Re_M_Const_30000, sigma)
Tu_Re_M_Const_30000_Poly, L_ux_Re_M_Const_30000_Poly = polyModel.evaluate(Re_M_Re_M_Const_30000, Ro_Re_M_Const_30000, sigma)

# Evaluate the model for Re_M = 40000
Ro_Re_M_Const_40000 = np.linspace(5, 75, num=100)
Re_M_Re_M_Const_40000 = np.ones(Ro_Re_M_Const_40000.shape)*40000
sigma = np.ones(Ro_Re_M_Const_40000.shape)*sigma_value

Tu_Re_M_Const_40000_NN, L_ux_Re_M_Const_40000_NN = nnModel.evaluate(Re_M_Re_M_Const_40000, Ro_Re_M_Const_40000, sigma)
Tu_Re_M_Const_40000_Poly, L_ux_Re_M_Const_40000_Poly = polyModel.evaluate(Re_M_Re_M_Const_40000, Ro_Re_M_Const_40000, sigma)

# Evaluate the model for Ro = 15
Re_M_Ro_Const_15 = np.linspace(5000, 50000, num=100)
Ro_Ro_Const_15 = np.ones(Re_M_Ro_Const_15.shape)*15
sigma = np.ones(Re_M_Ro_Const_15.shape)*sigma_value

Tu_Ro_Const_15_NN, L_ux_Ro_Const_15_NN = nnModel.evaluate(Re_M_Ro_Const_15, Ro_Ro_Const_15, sigma)
Tu_Ro_Const_15_Poly, L_ux_Ro_Const_15_Poly = polyModel.evaluate(Re_M_Ro_Const_15, Ro_Ro_Const_15, sigma)

# Evaluate the model for Ro = 25
Re_M_Ro_Const_25 = np.linspace(5000, 50000, num=100)
Ro_Ro_Const_25 = np.ones(Re_M_Ro_Const_25.shape)*25
sigma = np.ones(Re_M_Ro_Const_25.shape)*sigma_value

Tu_Ro_Const_25_NN, L_ux_Ro_Const_25_NN = nnModel.evaluate(Re_M_Ro_Const_25, Ro_Ro_Const_25, sigma)
Tu_Ro_Const_25_Poly, L_ux_Ro_Const_25_Poly = polyModel.evaluate(Re_M_Ro_Const_25, Ro_Ro_Const_25, sigma)

# Evaluate the model for Ro = 40
Re_M_Ro_Const_40 = np.linspace(5000, 50000, num=100)
Ro_Ro_Const_40 = np.ones(Re_M_Ro_Const_40.shape)*40
sigma = np.ones(Re_M_Ro_Const_40.shape)*sigma_value

Tu_Ro_Const_40_NN, L_ux_Ro_Const_40_NN = nnModel.evaluate(Re_M_Ro_Const_40, Ro_Ro_Const_40, sigma)
Tu_Ro_Const_40_Poly, L_ux_Ro_Const_40_Poly = polyModel.evaluate(Re_M_Ro_Const_40, Ro_Ro_Const_40, sigma)

# %% Select the experimental data for the constant sigma plot

shaft_speed_product = np.divide(IO_data["Shaft Speed Standard Deviation * M^2 / nu"],IO_data["Grid Re"])

# Boolean masks for selecting data based on conditions
constant_re_30000 = (IO_data["Grid Re"] > 25000) & (IO_data["Grid Re"] < 35000)
constant_re_40000 = (IO_data["Grid Re"] > 35000) & (IO_data["Grid Re"] < 45000)
constant_re_20000 = (IO_data["Grid Re"] > 15000) & (IO_data["Grid Re"] < 25000)
constant_ro_15 = (IO_data["Rossby Number"] == 15)
constant_ro_25 = (IO_data["Rossby Number"] == 25)
constant_ro_40 = (IO_data["Rossby Number"] == 40)
constant_sigma_200 = np.abs(IO_data["Shaft Speed Standard Deviation * M^2 / nu"] - sigma_value) < 1

# For Re_M = 30000
Re_M_Const_30000_Exp = IO_data[constant_re_30000 & constant_sigma_200].copy()
Re_M_Const_30000_Exp["Turbulence Intensity Percent"] = Re_M_Const_30000_Exp["Turbulence Intensity"]*100
Re_M_Const_30000_Exp["Turbulence Intensity Uncertainty Percent"] = Re_M_Const_30000_Exp["Turbulence Intensity Precision Uncertainty"]*100

# For Re_M = 40000
Re_M_Const_40000_Exp = IO_data[constant_re_40000 & constant_sigma_200].copy()
Re_M_Const_40000_Exp["Turbulence Intensity Percent"] = Re_M_Const_40000_Exp["Turbulence Intensity"]*100
Re_M_Const_40000_Exp["Turbulence Intensity Uncertainty Percent"] = Re_M_Const_40000_Exp["Turbulence Intensity Precision Uncertainty"]*100

# For Re_M = 20000
Re_M_Const_20000_Exp = IO_data[constant_re_20000 & constant_sigma_200].copy()
Re_M_Const_20000_Exp["Turbulence Intensity Percent"] = Re_M_Const_20000_Exp["Turbulence Intensity"]*100
Re_M_Const_20000_Exp["Turbulence Intensity Uncertainty Percent"] = Re_M_Const_20000_Exp["Turbulence Intensity Precision Uncertainty"]*100

# For Ro = 25
Ro_Const_25_Exp = IO_data[constant_ro_25 & constant_sigma_200].copy()
Ro_Const_25_Exp["Turbulence Intensity Percent"] = Ro_Const_25_Exp["Turbulence Intensity"]*100
Ro_Const_25_Exp["Turbulence Intensity Uncertainty Percent"] = Ro_Const_25_Exp["Turbulence Intensity Precision Uncertainty"]*100

# For Ro = 15
Ro_Const_15_Exp = IO_data[constant_ro_15 & constant_sigma_200].copy()
Ro_Const_15_Exp["Turbulence Intensity Percent"] = Ro_Const_15_Exp["Turbulence Intensity"]*100
Ro_Const_15_Exp["Turbulence Intensity Uncertainty Percent"] = Ro_Const_15_Exp["Turbulence Intensity Precision Uncertainty"]*100

# For Ro = 40
Ro_Const_40_Exp = IO_data[constant_ro_40 & constant_sigma_200].copy()
Ro_Const_40_Exp["Turbulence Intensity Percent"] = Ro_Const_40_Exp["Turbulence Intensity"]*100
Ro_Const_40_Exp["Turbulence Intensity Uncertainty Percent"] = Ro_Const_40_Exp["Turbulence Intensity Precision Uncertainty"]*100

# %% Plot the data

# Create figure with 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(6.375, 6.375))
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

axs[0].plot(Ro_Re_M_Const_20000, Tu_Re_M_Const_20000_NN*100, c="b", lw = 1)
axs[0].plot(Ro_Re_M_Const_30000, Tu_Re_M_Const_30000_NN*100, c="k", lw = 1)
axs[0].plot(Ro_Re_M_Const_40000, Tu_Re_M_Const_40000_NN*100, c="r", lw = 1)

axs[0].plot(Ro_Re_M_Const_20000, Tu_Re_M_Const_20000_Poly*100, c="b", lw = 1, ls='--')
axs[0].plot(Ro_Re_M_Const_30000, Tu_Re_M_Const_30000_Poly*100, c="k", lw = 1, ls='--')
axs[0].plot(Ro_Re_M_Const_40000, Tu_Re_M_Const_40000_Poly*100, c="r", lw = 1, ls='--')

axs[0].set_xlabel(r"$\textrm{Ro}$")
axs[0].set_ylabel(r'$Tu$ [\%]')
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

axs[1].plot(Re_M_Ro_Const_15,Tu_Ro_Const_15_NN*100, c="b", lw = 1)
axs[1].plot(Re_M_Ro_Const_25,Tu_Ro_Const_25_NN*100, c="k", lw = 1)
axs[1].plot(Re_M_Ro_Const_40,Tu_Ro_Const_40_NN*100, c="r", lw = 1)

axs[1].plot(Re_M_Ro_Const_15,Tu_Ro_Const_15_Poly*100, c="b", lw = 1, ls='--')
axs[1].plot(Re_M_Ro_Const_25,Tu_Ro_Const_25_Poly*100, c="k", lw = 1, ls='--')
axs[1].plot(Re_M_Ro_Const_40,Tu_Ro_Const_40_Poly*100, c="r", lw = 1, ls='--')

axs[1].set_ylabel(r'$Tu$ [\%]')
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

axs[2].plot(Ro_Re_M_Const_20000, L_ux_Re_M_Const_20000_NN, c="b", lw = 1)
axs[2].plot(Ro_Re_M_Const_30000, L_ux_Re_M_Const_30000_NN, c="k", lw = 1)
axs[2].plot(Ro_Re_M_Const_40000, L_ux_Re_M_Const_40000_NN, c="r", lw = 1)

axs[2].plot(Ro_Re_M_Const_20000, L_ux_Re_M_Const_20000_Poly, c="b", lw = 1, ls='--')
axs[2].plot(Ro_Re_M_Const_30000, L_ux_Re_M_Const_30000_Poly, c="k", lw = 1, ls='--')
axs[2].plot(Ro_Re_M_Const_40000, L_ux_Re_M_Const_40000_Poly, c="r", lw = 1, ls='--')

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

axs[3].plot(Re_M_Ro_Const_15, L_ux_Ro_Const_15_NN, c="b", lw = 1)
axs[3].plot(Re_M_Ro_Const_25, L_ux_Ro_Const_25_NN, c="k", lw = 1)
axs[3].plot(Re_M_Ro_Const_40, L_ux_Ro_Const_40_NN, c="r", lw = 1)

axs[3].plot(Re_M_Ro_Const_15, L_ux_Ro_Const_15_Poly, c="b", lw = 1, ls='--')
axs[3].plot(Re_M_Ro_Const_25, L_ux_Ro_Const_25_Poly, c="k", lw = 1, ls='--')
axs[3].plot(Re_M_Ro_Const_40, L_ux_Ro_Const_40_Poly, c="r", lw = 1, ls='--')
axs[3].set_ylabel(r'$L_{ux}/M$')
axs[3].set_xlabel(r"$\textrm{Re}_M$")

# X-Axis Limits
for x in [1,3]:
    axs[x].set_xlim(left=0, right=60000)
    
for x in [0,2]:
    axs[x].set_xlim(left=0, right=80)
    
# Y-Axis Limits
for x in [0,1]:
    axs[x].set_ylim(bottom=9, top=15)
    
for x in [2,3]:
    axs[x].set_ylim(bottom=1, top=5.5)

# Adjust layout
plt.subplots_adjust(top=0.78, hspace=0.45, wspace=0.4)

# Subfigure labels
subfig_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
for i, ax in enumerate(axs):
    ax.text(0.02, 0.95, subfig_labels[i], transform=ax.transAxes, fontsize=10, va='top', ha='left', fontweight='bold')


plt.show()

# Print
modelExperimentComparisonFigure_FileName = "../Figures/modelExperimentComparison.eps"
fig.savefig(modelExperimentComparisonFigure_FileName,format="eps")
