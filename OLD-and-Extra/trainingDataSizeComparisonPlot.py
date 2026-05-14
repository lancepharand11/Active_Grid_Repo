# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:22:06 2025

@author: Connor
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman"
})

PolynomialOrder1Data = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Polynomial_Order1.csv")
PolynomialOrder2Data = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Polynomial_Order2.csv")

NNData_1Layer = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Neural Network 1-Layer.csv")
NNData_2Layer = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Neural Network 2-Layer.csv")

# %% RMS Uncertainty
IO_data_file_path = "../OLD-and-Extra/DataSummaryOutliersRemoved.csv"
IO_data = pd.read_csv(IO_data_file_path)

IO_data = IO_data[["Trial Name",
                   "Grid Re",
                   "Rossby Number",
                   "Shaft Speed Standard Deviation * M^2 / nu",
                   "Turbulence Intensity",
                   "L_ux / M",
                   "Turbulence Intensity Precision Uncertainty",
                   "L_ux Uncertainty"]]

RMS_Uncertainty_Tu = np.sqrt((IO_data["Turbulence Intensity Precision Uncertainty"]**2).mean())
RMS_Uncertainty_L_ux = np.sqrt((IO_data["L_ux Uncertainty"]**2).mean())
 

# %% Plot
fig, ax = plt.subplots(2, 1, figsize=(6.375, 6.375))
# Turbulence Intensity
ax[0].plot(PolynomialOrder1Data["Size of Training Data"],PolynomialOrder1Data["Tu RMS Error"]*100,
           linestyle="--", color="k", 
           label=r"$1^{\textrm{st}}$-Order Polynomial",
           marker="s",
           markersize=3)
ax[0].plot(PolynomialOrder2Data["Size of Training Data"],PolynomialOrder2Data["Tu RMS Error"]*100,
           linestyle=":", color="r",
           label=r"$2^{\textrm{nd}}$-Order Polynomial",
           marker="s",
           markersize=3)
ax[0].plot(NNData_1Layer["Size of Training Data"],NNData_1Layer["Tu RMS Error"]*100,
           linestyle="-", color="g", 
           label="Neural Network - One Hidden Layer",
           marker="s",
           markersize=3)
ax[0].plot(NNData_2Layer["Size of Training Data"],NNData_2Layer["Tu RMS Error"]*100,
           linestyle="-.", color="b",
           label="Neural Network - Two Hidden Layers",
           marker="s",
           markersize=3)
ax[0].axhline(y=RMS_Uncertainty_Tu*100, color="k", linestyle="-")
ax[0].annotate("RMS measurement precision uncertainty", xy=(80, RMS_Uncertainty_Tu*100), xytext=(62, RMS_Uncertainty_Tu*100+0.005))
ax[0].set_title(r"(a) $Tu$")
# ax[0].set_yscale("log")
ax[0].set_ylabel(r"RMSE$\left( Tu \right)$ [\%]")
# ax[0].legend(loc="upper right")

# Set y-axis to show 2 decimal places
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Major ticks every 0.5
ax[0].yaxis.set_major_locator(MultipleLocator(0.05))


# Integral Length Scale
ax[1].plot(PolynomialOrder1Data["Size of Training Data"],PolynomialOrder1Data["L_ux RMS Error"],
           linestyle="--", color="k",
           marker="s",
           markersize=3)
ax[1].plot(PolynomialOrder2Data["Size of Training Data"],PolynomialOrder2Data["L_ux RMS Error"],
           linestyle=":", color="r",
           marker="s",
           markersize=3)
ax[1].plot(NNData_1Layer["Size of Training Data"],NNData_1Layer["L_ux RMS Error"],
           linestyle="-", color="g",
           marker="s",
           markersize=3)
ax[1].plot(NNData_2Layer["Size of Training Data"],NNData_2Layer["L_ux RMS Error"],
           linestyle="-.", color="b",
           marker="s",
           markersize=3)
ax[1].axhline(y=RMS_Uncertainty_L_ux, color="k", linestyle="-")
ax[1].annotate("RMS measurement precision uncertainty", xy=(80, RMS_Uncertainty_L_ux), xytext=(62, RMS_Uncertainty_L_ux+0.005))

ax[1].set_title(r"(b) $L_{ux}/M$")
# ax[1].set_yscale("log")
ax[1].set_xlabel("Number of training data samples")
ax[1].set_ylabel(r"RMSE$\left(L_{ux}\right)/M$")

# Set y-axis to show 2 decimal places
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Major ticks every 0.5
ax[1].yaxis.set_major_locator(MultipleLocator(0.05))

# ax[1].legend(loc="upper right")

# Adjust layout
plt.subplots_adjust(hspace=0.3,top=0.8)

# Legend
legend = fig.legend(loc='outside upper center', ncols=1)

# Print
modelExperimentComparisonFigure_FileName = "../Figures/trainingDataSize.eps"
fig.savefig(modelExperimentComparisonFigure_FileName,format="eps")