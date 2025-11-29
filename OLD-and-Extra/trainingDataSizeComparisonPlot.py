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

NNData = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Neural Network.csv")

# %% Plot
fig, ax = plt.subplots(1, 1, figsize=(6.375, 6.375*2/3))
# Turbulence Intensity
ax.plot(PolynomialOrder1Data["Size of Training Data"],PolynomialOrder1Data["Tu RMS Relative Error"], linestyle="--", color="k", label=r"$Tu$ $1^{\textrm{st}}$-Order Polynomial")
ax.plot(PolynomialOrder2Data["Size of Training Data"],PolynomialOrder2Data["Tu RMS Relative Error"], linestyle=":", color="k", label=r"$Tu$ $2^{\textrm{nd}}$-Order Polynomial")
ax.plot(NNData["Size of Training Data"],NNData["Tu RMS Relative Error"], linestyle="-", color="k", label="$Tu$ Neural Network")
# Integral Length Scale
ax.plot(PolynomialOrder1Data["Size of Training Data"],PolynomialOrder1Data["L_ux RMS Relative Error"], linestyle="--", color="b", label=r"$L_{ux}$ $1^{\textrm{st}}$-Order Polynomial")
ax.plot(PolynomialOrder2Data["Size of Training Data"],PolynomialOrder2Data["L_ux RMS Relative Error"], linestyle=":", color="b", label=r"$L_{ux}$ $2^{\textrm{nd}}$-Order Polynomial")
ax.plot(NNData["Size of Training Data"],NNData["L_ux RMS Relative Error"], linestyle="-", color="b", label="$L_{ux}$ Neural Network")

ax.set_yscale("log")
ax.set_xlabel("Size of training data set")
ax.set_ylabel("RMS Relative Error")
ax.legend()

# Print
modelExperimentComparisonFigure_FileName = "../Figures/trainingDataSize.eps"
fig.savefig(modelExperimentComparisonFigure_FileName,format="eps")