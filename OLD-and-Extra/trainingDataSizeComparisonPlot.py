# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:22:06 2025

@author: Connor
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

PolynomialOrder1Data = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Polynomial_Order1.csv")
PolynomialOrder2Data = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Polynomial_Order2.csv")

NNData_1Layer = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Neural Network 1-Layer.csv")
NNData_2Layer = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Neural Network 2-Layer.csv")

# %% Plot
fig, ax = plt.subplots(2, 1, figsize=(6.375, 6.375))
# Turbulence Intensity
ax[0].plot(PolynomialOrder1Data["Size of Training Data"],PolynomialOrder1Data["Tu RMS Error"], linestyle="--", color="k", label=r"$1^{\textrm{st}}$-Order Polynomial")
ax[0].plot(PolynomialOrder2Data["Size of Training Data"],PolynomialOrder2Data["Tu RMS Error"], linestyle=":", color="k", label=r"$2^{\textrm{nd}}$-Order Polynomial")
ax[0].plot(NNData_1Layer["Size of Training Data"],NNData_1Layer["Tu RMS Error"], linestyle="-", color="k", label="Neural Network - 1 Hidden Layer")
ax[0].plot(NNData_2Layer["Size of Training Data"],NNData_2Layer["Tu RMS Error"], linestyle="-.", color="k", label="Neural Network - 2 Hidden Layers")

ax[0].set_title(r"(a) $Tu$")
ax[0].set_yscale("log")
ax[0].set_ylabel("RMS Error")
ax[0].legend(loc="upper right")

# Integral Length Scale
ax[1].plot(PolynomialOrder1Data["Size of Training Data"],PolynomialOrder1Data["L_ux RMS Error"], linestyle="--", color="b", label=r"$1^{\textrm{st}}$-Order Polynomial")
ax[1].plot(PolynomialOrder2Data["Size of Training Data"],PolynomialOrder2Data["L_ux RMS Error"], linestyle=":", color="b", label=r"$2^{\textrm{nd}}$-Order Polynomial")
ax[1].plot(NNData_1Layer["Size of Training Data"],NNData_1Layer["L_ux RMS Error"], linestyle="-", color="b", label="Neural Network - 1 Hidden Layer")
ax[1].plot(NNData_2Layer["Size of Training Data"],NNData_2Layer["L_ux RMS Error"], linestyle="-.", color="b", label="Neural Network - 2 Hidden Layers")

ax[1].set_title(r"(b) $L_{ux}/M$")
ax[1].set_yscale("log")
ax[1].set_xlabel("Size of training data set")
ax[1].set_ylabel("RMS Error")
ax[1].legend(loc="upper right")

# Adjust layout
plt.subplots_adjust(hspace=0.4)



# Print
modelExperimentComparisonFigure_FileName = "../Figures/trainingDataSize.eps"
fig.savefig(modelExperimentComparisonFigure_FileName,format="eps")