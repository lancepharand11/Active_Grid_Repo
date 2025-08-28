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

PolynomialData = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Polynomial.csv")

NNData = pd.read_csv("../Turb-int-and-Integral-Length-Scale-Model/Training Data Size Analysis Results/Neural Network.csv")

# %% Plot
fig, ax = plt.subplots(1, 1, figsize=(6.375, 6.375*2/3))
ax.plot(PolynomialData["Size of Training Data"],PolynomialData["Tu RMS Relative Error"], linestyle="--", color="k", label=r"$Tu$ $5^{\textrm{th}}$-Order Polynomial")
ax.plot(PolynomialData["Size of Training Data"],PolynomialData["L_ux RMS Relative Error"], linestyle="--", color="b", label=r"$L_{ux}$ $5^{\textrm{th}}$-Order Polynomial")

ax.plot(NNData["Size of Training Data"],NNData["Tu RMS Relative Error"], linestyle="-", color="k", label="$Tu$ Neural Network")
ax.plot(NNData["Size of Training Data"],NNData["L_ux RMS Relative Error"], linestyle="-", color="b", label="$L_{ux}$ Neural Network")
ax.set_yscale("log")
ax.set_xlabel("Size of training data set")
ax.set_ylabel("RMS Relative Error")
ax.legend()

# Print
modelExperimentComparisonFigure_FileName = "../Figures/trainingDataSize.eps"
fig.savefig(modelExperimentComparisonFigure_FileName,format="eps")