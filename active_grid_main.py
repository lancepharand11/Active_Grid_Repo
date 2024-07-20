# Initial Turbulence Intensity Model for Active Grid
# Author: Lance Pharand, 2024
# NOTEs:
# Download the following packages below if not installed already
# !! IMPORTANT Set the Turbulence Parameters Class variables in the "Initializations" section !!

import scipy.io
from scipy import integrate
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time as time_clock
from Turbulence_Parameters_class import Turbulence_Parameters

###################################################################
## Initializations
###################################################################
dataDir = "/Users/lancepharand/Desktop/URA S24/Experiment_Scripts/Active_Grid_Data_and_Files/Active_Grid_Data/"
counter = 0
turb_objects = []
Turbulence_Parameters.fs = 25600 #Hz
Turbulence_Parameters.N_samples = 6144000
Turbulence_Parameters.overlap = 0.5

###################################################################
### Function Defs
###################################################################
# def window(size):
#     return np.ones(size)/float(size)

def signaltonoise(a, axis=0, ddof=0):
    if type(a) is not pd.DataFrame or type(a) is not np.ndarray or type(a) is not pd.Series:
        a = np.asanyarray(a)

    m = np.mean(a, axis=axis)
    sd = np.std(a, axis=axis, ddof=ddof)
    return (20 * np.log10(abs(np.where(sd == 0, 0, m/sd))))


###################################################################
## Read In and Formating I/O Data
###################################################################

for file in os.listdir(dataDir):
    if counter == 0:
        time_stamps = scipy.io.loadmat((dataDir + file), variable_names=['timeStamps'], squeeze_me=True, mat_dtype=True)
        counter += 1

    if file == ".DS_Store":
        continue
    name_full = os.path.basename(dataDir + file).split("/")[-1]
    name = name_full.split(".mat")[0]
    mat_u = scipy.io.loadmat((dataDir + file), variable_names=['u'], squeeze_me=True, mat_dtype=True)
    mat_v = scipy.io.loadmat((dataDir + file), variable_names=['v'], squeeze_me=True, mat_dtype=True)
    temp_turb_obj = Turbulence_Parameters(filename=name, u_velo=mat_u['u'], v_velo=mat_v['v'],
                                          freestream_velo=float(name.split("_")[1]), Rossby_num=float(name.split("_")[3]),
                                          shaft_speed_std_dev=float(name.split("_")[5]))
    temp_turb_obj.calc_L_ux()
    temp_turb_obj.calc_turb_psd_spectrum()
    temp_turb_obj.calc_turb_intensity()
    turb_objects.append(temp_turb_obj)

# For Debugging only
for i, obj in enumerate(turb_objects):
    print(turb_objects[i])

# Turb Int TEST
print(f"Turbulence Intensity: {turb_objects[9].get_turb_int()}")

# L_ux TEST
print(f"Integral Length scale (L_ux): {turb_objects[9].get_L_ux()} [m]")

# PSD TEST
plt.figure(figsize = (12, 6))
plt.plot(turb_objects[9].get_wavenums(), turb_objects[9].get_E_k(), 'b') # plot 1 sided PSD
plt.xlabel('Wavenumber (m^-1)')
# plt.xlabel('Freq (Hz)')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Power Density (Amp^2/Hz)')
# plt.xlim(0, (30000))
plt.ylim(1e-14, 1)
plt.grid()
plt.show()

###################################################################
## Data Analysis and Trends
###################################################################

# # Plot correlation heat map
# data_corr = IO_data.corr()
# plt.figure(figsize = (10, 8))
# sns.heatmap(data_corr, cmap=sns.diverging_palette(0, 255, as_cmap=True))
# plt.show()
#
# # Look at scatter plots
# plt.figure(figsize = (10, 8))
# sns.pairplot(IO_data)
# plt.show()

#########################################################################
## Create Single Output Lasso regression with polynomial features
#########################################################################

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LassoCV
# from sklearn.pipeline import Pipeline
#
# poly_reg = Pipeline([('poly', PolynomialFeatures(degree = 4)),
#                      ('lasso', LassoCV(alphas=np.linspace(0, 1, 100), max_iter=2000, tol=1e-3, random_state=42))]
#                     )
#
# poly_reg.fit(input_data, turb_intensity)
#
# print('Alphas', poly_reg['lasso'].alphas_)
#
# plt.figure(figsize=(12, 6))
# plt.plot(poly_reg['lasso'].alphas_, poly_reg['lasso'].mse_path_)
# plt.xlabel('Alpha')
# plt.ylabel('Cross Validation Score')
# plt.show()





###################################################################
## EXTRA
###################################################################
## PSD Plot format
# plt.figure(figsize = (12, 6))
# plt.plot(fft_freq, E_k, 'b') # plot 1 sided PSD
# # plt.xlabel('Wavenumber (m^-1)')
# plt.xlabel('Freq (Hz)')
# plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('Power Density (Amp^2/Hz)')
# # plt.xlim(0, (30000))
# plt.ylim(1e-14, 1)
# plt.grid()
# plt.show()

## Autocorrelation Plot format
# plt.figure(figsize = (12, 6))
# plt.plot(range(num_lags), R_ux, 'r')
# plt.plot(range(R_ux_fit.size), R_ux_fit, 'b')
# print(R_ux_fit.size)
# plt.xlabel('Lags')
# plt.ylabel('Correlation Coefficient')
# plt.grid()
# plt.show()

## OLD Scatter plot
# from pandas.plotting import scatter_matrix
# scatter_matrix(IO_data, figsize=(10, 10))

## Filter
# sos = signal.butter(6, 10000, btype='low', analog=False, fs=fs, output='sos')
# u_velo_filtered = signal.sosfilt(sos, u_velo.iloc[:, 0])
# b, a = signal.butter(6, 10000, btype='low', analog=False, fs=fs, output='ba')
# w, h = signal.freqs(b, a)
# plt.semilogx(w, 20 * np.log10(abs(h)))
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [radians / second]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')
# plt.axvline(10000, color='green') # cutoff frequency
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(time, u_velo.iloc[:, 0])
# ax1.set_title('U Velo w/o Filter Example')
# # ax1.axis([0, 1, -2, 2])
# ax2.plot(time, u_velo_filtered)
# ax2.set_title('U Velo w/ Filter Example')
# # ax2.axis([0, 1, -2, 2])
# ax2.set_xlabel('Time [seconds]')
# plt.tight_layout()
# plt.show()


# # Fix left skew of turb intensity data
# from sklearn.preprocessing import FunctionTransformer
#
# print(f"Skewness before: {IO_data['Turbulence Intensity'].skew()}")
#
# sqr_tr = FunctionTransformer(lambda X: X ** 3, inverse_func=lambda X: X ** 1/3)
# # sqr_tr = FunctionTransformer(np.square, inverse_func=np.sqrt)
# tr_turb_int = sqr_tr.transform(IO_data['Turbulence Intensity'])

# print(f"Skewness after: {tr_turb_int.skew()}")

# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# plt.title("Distribution before Transformation", fontsize=15)
# sns.histplot(IO_data['Turbulence Intensity'], kde=True, color="red")
# plt.subplot(1, 2, 2)
#
# plt.title("Distribution after Transformation", fontsize=15)
# sns.histplot(tr_turb_int, bins=20, kde=True, legend=False)
# plt.show()


