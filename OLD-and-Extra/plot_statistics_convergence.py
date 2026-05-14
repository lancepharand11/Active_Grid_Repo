# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:10:12 2026

@author: ctoppings
"""

import matplotlib.pyplot as plt
import pandas as pd
from Turbulence_Parameters_class import Turbulence_Parameters
from pathlib import Path
import scipy.io
import numpy as np

def plot_statistics_convergence(turb_data, dataDir, nu=1.5e-5):
    """
    Plots the convergence of turbulence statistics versus sampling period.
    Args:
        turb_data (pd.DataFrame): DataFrame containing turbulence data.
        dataDir (str): Path to folder containing measurement data
        nu (float): Kinematic viscosity [m^2/s]. Default is 1.5e-5.
    Returns:
        matplotlib.figure.Figure: The resulting figure object.
    """
    mesh_length = 0.06096  # [m] grid mesh length

    
    # Create figure with 2x2 layout
    fig, axs = plt.subplots(2, 1, figsize=(5.8, 5.8 * 1.2))
    axs = axs.flatten()
    
    # Select Case for Plotting
    most_uncertain_Lux_id = turb_data["L_ux / M"].idxmax()
    
    filename = dataDir / ( turb_data.loc[most_uncertain_Lux_id,"Trial Name"] + ".mat")
    file_Ro = float(filename.stem.split("_")[3])
    file_shaftSpeedSTD = float(filename.stem.split("_")[5])
    
    mat_u = scipy.io.loadmat(filename, variable_names=['u'], squeeze_me=True, mat_dtype=True)
    mat_v = scipy.io.loadmat(filename, variable_names=['v'], squeeze_me=True, mat_dtype=True)
    turb_obj = Turbulence_Parameters(filename=filename.stem, u_velo=mat_u['u'][4000000:], v_velo=mat_v['v'][4000000:],
                                          freestream_velo=np.mean(mat_u['u'][4000000:]), Rossby_num=file_Ro,
                                          shaft_speed_std_dev=file_shaftSpeedSTD)
    
    # Sampling range for convergence plots
    sample_range = np.arange(10000,turb_obj.N_samples,10000)
    
    for length in sample_range:
        
        # Integral Length Scale Convergence
        turb_obj.calc_L_ux(length)
        turb_obj.calc_L_ux_Uncertainty(length)
        
        axs[1].errorbar(length / turb_obj.fs * turb_obj.get_freestream_velo() / mesh_length, turb_obj.L_ux_non_dim, yerr=turb_obj.L_ux_uncertainty)
        axs[1].set_ylim(bottom = 0.10, top = 0.16)
        axs[1].set_ylabel(r"$L_{ux}/M$")
        axs[1].set_xlabel(r"$tu_\infty/M$")
    
        # Turbulence Intensity Convergence
        turb_obj.calc_turb_intensity(length)
        turb_obj.calc_turb_int_uncertainty(length)
        
        axs[0].errorbar(length / turb_obj.fs * turb_obj.get_freestream_velo() / mesh_length, turb_obj.turb_int, yerr=turb_obj.turb_int_uncertainty)
        axs[0].set_ylim(bottom = 0, top = 94)
        axs[0].set_ylabel(r"$Tu$")
        axs[0].set_xlabel(r"$tu_\infty/M$")
    
    
    plt.show()
    return fig
    
