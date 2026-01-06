# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:01:04 2025

@author: Connor
"""
import numpy as np
import matplotlib.pyplot as plt
from IntensityLengthModelClass import IntensityLengthModel

modelPath = "./Models_and_Results/best_model_20251224_124907.pth"

# Load the scalers
scaler1Path = "./Models_and_Results/scaler_x_20251224_124907.pkl"
scaler2Path = "./Models_and_Results/scaler_y_20251224_124907.pkl"

NNModel = IntensityLengthModel(modelPath, scaler1Path, scaler2Path)


# Define the range and step size for Re_M and Ro
rem = np.linspace(5000, 300000, num=100)
ro = np.linspace(10, 70, num=100)

# Create the meshgrid
Re_M, Ro = np.meshgrid(rem, ro)

sigma = np.ones(Re_M.shape)*0.014

Tu, L_ux = NNModel.evaluate(Re_M, Ro, sigma)

# %%
# Contour Plots
fig, axs = plt.subplots(2,1)

TuContour = axs[0].contourf(Re_M, Ro, Tu, 10)
axs[0].set_xlabel(r"$\textrm{Re}_M$")
axs[0].set_ylabel(r"$\textrm{Ro}$")
fig.colorbar(TuContour, ax=axs[0], label=r"$Tu$")

L_uxContour = axs[1].contourf(Re_M, Ro, L_ux,10)
axs[1].set_xlabel(r"$\textrm{Re}_M$")
axs[1].set_ylabel(r"$\textrm{Ro}$")
fig.colorbar(L_uxContour, ax=axs[1], label=r"$L_{ux}$")

