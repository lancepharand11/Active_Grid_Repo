import pandas as pd
import numpy as np

def shaftSpeedStdPlot(turb_data : pd.DataFrame):
    
    import matplotlib.pyplot as plt
    import matplotlib.markers as mkr
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import sys
    import os
    sys.path.insert(0, os.path.abspath('../Turb-int-and-Integral-Length-Scale-Model'))
    import IntensityLengthModelClass
    import IntensityLengthPolynomialModelClass
    
    present_marker = mkr.MarkerStyle('^',fillstyle="none")

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10;
    
    # Reynolds Numbers
    re = np.linspace(10000, 50000, 5)
    norm = mcolors.Normalize(vmin=re.min(), vmax=re.max())
    cmap = cm.viridis
    
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
    
    # %% Evaluate the models

    # Choose Shaft Speed Standard Deviation * M^2 / nu
    sigma = np.linspace(20,700,100)
    
    # Choose Rossby Number
    Ro = np.ones(sigma.shape)*30

    # Evaluate the model for Re_M = 10000
    Re_M_Const_10000 = np.ones(sigma.shape)*10000
    Re_M_Const_20000 = np.ones(sigma.shape)*20000
    Re_M_Const_30000 = np.ones(sigma.shape)*30000
    Re_M_Const_40000 = np.ones(sigma.shape)*40000
    Re_M_Const_50000 = np.ones(sigma.shape)*50000

    Tu_Re_M_Const_10000_NN, L_ux_Re_M_Const_10000_NN = nnModel.evaluate(Re_M_Const_10000, Ro, sigma)
    Tu_Re_M_Const_10000_Poly, L_ux_Re_M_Const_10000_Poly = polyModel.evaluate(Re_M_Const_10000, Ro, sigma)
    
    Tu_Re_M_Const_20000_NN, L_ux_Re_M_Const_20000_NN = nnModel.evaluate(Re_M_Const_20000, Ro, sigma)
    Tu_Re_M_Const_20000_Poly, L_ux_Re_M_Const_20000_Poly = polyModel.evaluate(Re_M_Const_20000, Ro, sigma)
    
    Tu_Re_M_Const_30000_NN, L_ux_Re_M_Const_30000_NN = nnModel.evaluate(Re_M_Const_30000, Ro, sigma)
    Tu_Re_M_Const_30000_Poly, L_ux_Re_M_Const_30000_Poly = polyModel.evaluate(Re_M_Const_30000, Ro, sigma)
    
    Tu_Re_M_Const_40000_NN, L_ux_Re_M_Const_40000_NN = nnModel.evaluate(Re_M_Const_40000, Ro, sigma)
    Tu_Re_M_Const_40000_Poly, L_ux_Re_M_Const_40000_Poly = polyModel.evaluate(Re_M_Const_40000, Ro, sigma)
    
    Tu_Re_M_Const_50000_NN, L_ux_Re_M_Const_50000_NN = nnModel.evaluate(Re_M_Const_50000, Ro, sigma)
    Tu_Re_M_Const_50000_Poly, L_ux_Re_M_Const_50000_Poly = polyModel.evaluate(Re_M_Const_50000, Ro, sigma)
        
    # Turbulence intensity as percentage
    turb_data["Turbulence Intensity Percent"] = turb_data["Turbulence Intensity"]*100
    turb_data["Turbulence Intensity Percent Uncertainty"] = turb_data["Turbulence Intensity Precision Uncertainty"]*100
    constant_re_10000 = (turb_data["Grid Re"] > 5000) & (turb_data["Grid Re"] < 15000)
    constant_re_20000 = (turb_data["Grid Re"] > 15000) & (turb_data["Grid Re"] < 25000)
    constant_re_30000 = (turb_data["Grid Re"] > 25000) & (turb_data["Grid Re"] < 35000)
    constant_re_40000 = (turb_data["Grid Re"] > 35000) & (turb_data["Grid Re"] < 45000)
    constant_re_50000 = (turb_data["Grid Re"] > 45000) & (turb_data["Grid Re"] < 55000)

    constant_ro_40 = (turb_data["Rossby Number"] == 30)
    turb_data_re_10000_ro_40 = turb_data[constant_re_10000 & constant_ro_40]
    turb_data_re_20000_ro_40 = turb_data[constant_re_20000 & constant_ro_40]
    turb_data_re_30000_ro_40 = turb_data[constant_re_30000 & constant_ro_40]
    turb_data_re_40000_ro_40 = turb_data[constant_re_40000 & constant_ro_40]
    turb_data_re_50000_ro_40 = turb_data[constant_re_50000 & constant_ro_40]
    
    # Create figure with 2x2 layout
    fig, axs = plt.subplots(2, 1, figsize=(5.8, 4.8))
    axs = axs.flatten()
    
    # Subfigure labels
    subfig_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    for i, ax in enumerate(axs):
        ax.text(0.02, 0.95, subfig_labels[i], transform=ax.transAxes, fontsize=10, va='top', ha='left', fontweight='bold')

    # Tu vs sigma_\Omega
    axs[0].errorbar(turb_data_re_10000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_10000_ro_40["Turbulence Intensity Percent"],
                    label=r'$\textrm{Re}_M=1\times10^4$',
                    yerr=turb_data_re_10000_ro_40["Turbulence Intensity Percent Uncertainty"],
                    marker=present_marker, c=cmap(norm(10000)), linestyle='none',capsize=2)
    axs[0].errorbar(turb_data_re_20000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_20000_ro_40["Turbulence Intensity Percent"],
                    label=r'$\textrm{Re}_M=2\times10^4$',
                    yerr=turb_data_re_20000_ro_40["Turbulence Intensity Percent Uncertainty"],
                    marker=present_marker, c=cmap(norm(20000)), linestyle='none',capsize=2)
    axs[0].errorbar(turb_data_re_30000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_30000_ro_40["Turbulence Intensity Percent"],
                    label=r'$\textrm{Re}_M=3\times10^4$',
                    yerr=turb_data_re_30000_ro_40["Turbulence Intensity Percent Uncertainty"],
                    marker=present_marker, c=cmap(norm(30000)), linestyle='none',capsize=2)
    axs[0].errorbar(turb_data_re_40000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_40000_ro_40["Turbulence Intensity Percent"],
                    label=r'$\textrm{Re}_M=4\times10^4$',
                    yerr=turb_data_re_40000_ro_40["Turbulence Intensity Percent Uncertainty"],
                    marker=present_marker, c=cmap(norm(40000)), linestyle='none',capsize=2)
    axs[0].errorbar(turb_data_re_50000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_50000_ro_40["Turbulence Intensity Percent"],
                    label=r'$\textrm{Re}_M=5\times10^4$',
                    yerr=turb_data_re_50000_ro_40["Turbulence Intensity Percent Uncertainty"],
                    marker=present_marker, c=cmap(norm(50000)), linestyle='none',capsize=2)
    
    # Model
    axs[0].plot(sigma,Tu_Re_M_Const_10000_NN*100, c=cmap(norm(10000)), lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_20000_NN*100, c=cmap(norm(20000)), lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_30000_NN*100, c=cmap(norm(30000)), lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_40000_NN*100, c=cmap(norm(40000)), lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_50000_NN*100, c=cmap(norm(50000)), lw = 1)
    
    axs[0].plot(sigma,Tu_Re_M_Const_10000_Poly*100, c=cmap(norm(10000)), linestyle='--', lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_20000_Poly*100, c=cmap(norm(20000)), linestyle='--', lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_30000_Poly*100, c=cmap(norm(30000)), linestyle='--', lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_40000_Poly*100, c=cmap(norm(40000)), linestyle='--', lw = 1)
    axs[0].plot(sigma,Tu_Re_M_Const_50000_Poly*100, c=cmap(norm(50000)), linestyle='--', lw = 1)
    
    axs[0].set_xlabel(r"")
    axs[0].set_ylabel(r'$Tu$')
    legend = fig.legend(loc='outside upper center', ncols=2)
    legend.get_frame().set_edgecolor('black')
    
    # L_ux vs sigma_\Omega
    axs[1].errorbar(turb_data_re_10000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_10000_ro_40["L_ux / M"],
                    label=r'$\textrm{Re}_M=1\times10^4$',
                    yerr=turb_data_re_10000_ro_40["L_ux Uncertainty"],
                    marker=present_marker, c=cmap(norm(10000)), linestyle='none',capsize=2)
    axs[1].errorbar(turb_data_re_20000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_20000_ro_40["L_ux / M"],
                    label=r'$\textrm{Re}_M=2\times10^4$',
                    yerr=turb_data_re_20000_ro_40["L_ux Uncertainty"],
                    marker=present_marker, c=cmap(norm(20000)), linestyle='none',capsize=2)
    axs[1].errorbar(turb_data_re_30000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_30000_ro_40["L_ux / M"],
                    label=r'$\textrm{Re}_M=3\times10^4$',
                    yerr=turb_data_re_30000_ro_40["L_ux Uncertainty"],
                    marker=present_marker, c=cmap(norm(30000)), linestyle='none',capsize=2)
    axs[1].errorbar(turb_data_re_40000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_40000_ro_40["L_ux / M"],
                    label=r'$\textrm{Re}_M=4\times10^4$',
                    yerr=turb_data_re_40000_ro_40["L_ux Uncertainty"],
                    marker=present_marker, c=cmap(norm(40000)), linestyle='none',capsize=2)
    axs[1].errorbar(turb_data_re_50000_ro_40["Shaft Speed Standard Deviation * M^2 / nu"],turb_data_re_50000_ro_40["L_ux / M"],
                    label=r'$\textrm{Re}_M=5\times10^4$',
                    yerr=turb_data_re_50000_ro_40["L_ux Uncertainty"],
                    marker=present_marker, c=cmap(norm(50000)), linestyle='none',capsize=2)    
    axs[1].set_ylabel(r'$L_{ux}/M$')
    axs[1].set_xlabel(r"$\sigma_\Omega M^2/\nu$")
    
    # Model
    axs[1].plot(sigma, L_ux_Re_M_Const_10000_NN, c=cmap(norm(10000)), lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_20000_NN, c=cmap(norm(20000)), lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_30000_NN, c=cmap(norm(30000)), lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_40000_NN, c=cmap(norm(40000)), lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_50000_NN, c=cmap(norm(50000)), lw = 1)
    
    axs[1].plot(sigma, L_ux_Re_M_Const_10000_Poly, c=cmap(norm(10000)), linestyle='--', lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_20000_Poly, c=cmap(norm(20000)), linestyle='--', lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_30000_Poly, c=cmap(norm(30000)), linestyle='--', lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_40000_Poly, c=cmap(norm(40000)), linestyle='--', lw = 1)
    axs[1].plot(sigma, L_ux_Re_M_Const_50000_Poly, c=cmap(norm(50000)), linestyle='--', lw = 1)
    
    # X-Axis Limits
    # for x in [0,1,2,3]:
    #     axs[x].set_xlim(left=0, right=300000)
    
    # Adjust layout
    plt.subplots_adjust(hspace=0.2,left=0.12,right=0.96,wspace=0.18,bottom=0.1,top=0.83)
    
    # Set Y-Axis Limits
    axs[0].set_ylim([8.5,15.5])
    
    plt.show()
    return fig
