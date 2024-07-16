# Initial Turbulence Intensity Model for Active Grid
# Author: Lance Pharand, 2024
# NOTEs: Download the following packages below if not installed already

import scipy.io
import scipy.signal as signal
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns

###################################################################
## Initialization
###################################################################

dataDir = "/Users/lancepharand/Desktop/URA S24/Experiment_Scripts/Turb_Int_Out_Model/Active_Grid_Data/"
u_data_read = []
v_data_read = []
names_files = []
freestream_velo = []
Rossby_nums = []
shaft_speed_std_dev = []
time = []
counter = 0

###################################################################
### Function Defs
###################################################################
def window(size):
    return np.ones(size)/float(size)

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
        temp_time = scipy.io.loadmat((dataDir + file), variable_names=['timeStamps'], squeeze_me=True, mat_dtype=True)
        counter += 1

    if file == ".DS_Store":
        continue

    name_full = os.path.basename(dataDir + file).split("/")[-1]
    name = name_full.split(".mat")[0]
    freestream_velo.append(float(name.split("_")[1]))
    Rossby_nums.append(float(name.split("_")[3]))
    shaft_speed_std_dev.append(float(name.split("_")[5]))
    mat_u = scipy.io.loadmat((dataDir + file), variable_names=['u'], squeeze_me=True, mat_dtype=True)
    mat_v = scipy.io.loadmat((dataDir + file), variable_names=['v'], squeeze_me=True, mat_dtype=True)
    names_files.append("velo_from_" + name)
    u_data_read.append(mat_u['u'])
    v_data_read.append(mat_v['v'])

input_data = pd.DataFrame({"Freestream Velocity [m/s]" : freestream_velo,
                        "Rossby Number" : Rossby_nums,
                        "Shaft Speed Standard Deviation [rev/s]" : shaft_speed_std_dev})

time_data = np.array(temp_time['timeStamps']).T
time = pd.Series(time_data, name='timeStamps')

u_temp_data = np.array(u_data_read).T
u_velo = pd.DataFrame(u_temp_data)

v_temp_data = np.array(v_data_read).T
v_velo = pd.DataFrame(v_temp_data)

q_var = np.var(u_temp_data, axis=0) + (2 * np.var(v_temp_data, axis=0))

fs_temp = input_data.iloc[:, 0]
turb_intensity = pd.Series((np.sqrt(q_var) / (fs_temp * math.sqrt(3))), name="Turbulence Intensity")

snr_u_velo = pd.Series(signaltonoise(u_velo, axis=0, ddof=0), name="SNR of u velo [dB]")

total_data = pd.concat([input_data, turb_intensity, snr_u_velo], axis=1)
total_data.index = names_files

IO_data = pd.concat([input_data, turb_intensity], axis=1)
IO_data.index = names_files


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
#                      ('lasso', LassoCV(alphas=np.linspace(0, 1, 100), max_iter=2000, tol=1e-4, random_state=42))]
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



print(1)


## Generate turbulence PSD spectrum

fs = 25600 #Hz
s = u_velo.iloc[:, 0] - np.mean(u_velo.iloc[:, 0])
N = u_velo.iloc[:, 0].shape[0]

# # For debugging ONLY:
# N = 60000
# T = 1/fs
# x = np.linspace(0.0, N*T, N)
# s = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

y_fft = np.fft.fft(s)
fftfreq = np.linspace(0, fs/2, (N//2)+1)
E_1D = 0.5 * signal.fftconvolve(in1=y_fft, in2=y_fft)
y_psd = (np.abs(E_1D) ** 2) / (fs * N) # 2 sided
y_psd = y_psd[:((N//2)+1)] # 1 sided
y_psd[1:-1] = 2 * y_psd[1:-1] #double values except for dc gain and nyquist freq

# y_psd = (np.abs(y_fft) ** 2) / (fs * N) # 2 sided
# y_psd = y_psd[:((N//2)+1)] # 1 sided
# y_psd[1:-1] = 2 * y_psd[1:-1] #double values except for dc gain and nyquist freq

plt.figure(figsize = (12, 6))
ax = plt.plot(fftfreq, y_psd, 'b') # plot 1 sided PSD
plt.xlabel('Freq (Hz)')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Power Density (Amp^2/Hz)')
# plt.xlim(0, (30000))
# plt.ylim(0, 1e6)
plt.grid()
plt.show()


######## EXTRA
## PSD
# P = abs(y_fft)*2/len(y_fft)
# plt.psd(u_velo.iloc[:, 0], Fs=fs, window="window_" , sides=)
# plt.xlabel('Frequency')
# plt.ylabel('PSD(db)')
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

