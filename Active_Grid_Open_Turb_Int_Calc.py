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

dataDir = "/Users/lancepharand/Desktop/URA S24/Experiment_Scripts/Active_Grid_Data_and_Files/Active_Grid_Open_Data/"
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
    mat_u = scipy.io.loadmat((dataDir + file), variable_names=['u'], squeeze_me=True, mat_dtype=True)
    mat_v = scipy.io.loadmat((dataDir + file), variable_names=['v'], squeeze_me=True, mat_dtype=True)
    names_files.append("velo_from_" + name)
    u_data_read.append(mat_u['u'])
    v_data_read.append(mat_v['v'])

input_data = pd.Series(freestream_velo, name="Freestream Velocity [m/s]")

time_data = np.array(temp_time['timeStamps']).T
time = pd.Series(time_data, name='timeStamps')

u_temp_data = np.array(u_data_read).T
u_velo = pd.DataFrame(u_temp_data)

v_temp_data = np.array(v_data_read).T
v_velo = pd.DataFrame(v_temp_data)

q_var = np.var(u_temp_data, axis=0) + (2 * np.var(v_temp_data, axis=0))

fs_temp = input_data
turb_intensity = pd.Series((np.sqrt(q_var) / (fs_temp * math.sqrt(3))), name="Turbulence Intensity")

snr_u_velo = pd.Series(signaltonoise(u_velo, axis=0, ddof=0), name="SNR of u velo [dB]")

total_data = pd.concat([input_data, turb_intensity, snr_u_velo], axis=1)
total_data.index = names_files

IO_data = pd.concat([input_data, turb_intensity], axis=1)
IO_data.index = names_files

print(1)