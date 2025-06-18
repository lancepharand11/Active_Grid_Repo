import scipy.io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from Turbulence_Parameters_class import Turbulence_Parameters


###################################################################
## Initialization
###################################################################

dataDir = Path('F:/Lance/Active_Grid_Model_Data')
u_data_read = []
v_data_read = []
names_files = []
freestream_velo = []
Rossby_nums = []
shaft_speed_std_dev = []
time = []
counter = 0

Turbulence_Parameters.fs = 25600 #Hz

###################################################################
### Setup Figure
###################################################################
plt.ion()
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xlabel(r"$t$ [s]")
ax1.set_ylabel(r"$u'$ [m/s]")
ax1.set_xlim(left=0,right=0.1)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r"$fM/u_\infty$")
ax2.set_ylabel(r"$E_{u'u'}/\left(Mu_\infty\right)$")

unfiltered_velo_fluct_plot, = ax1.plot([],[],
        linewidth=0.2)
filtered_velo_fluct_plot, = ax1.plot([],[],
        linewidth=0.2)

unfiltered_psd_plot, = ax2.plot([],[],
        linewidth=0.2)
filtered_psd_plot, = ax2.plot([],[],
        linewidth=0.2)

###################################################################
## Read In and Filter Data
###################################################################

for file in list(dataDir.glob('*.mat')):
    if counter == 0:
        temp_time = scipy.io.loadmat(file, variable_names=['timeStamps'], squeeze_me=True, mat_dtype=True)
        time_data = np.array(temp_time['timeStamps']).T
        counter += 1

    file_uinfty = (float(file.stem.split("_")[1]))
    Ro_string = file.stem.split("_")[3]
    if Ro_string == '-':
        continue
    
    file_Ro = float(Ro_string)
    file_shaftSpeedSTD = float(file.stem.split("_")[5])
    mat_u = scipy.io.loadmat(file, variable_names=['u'], squeeze_me=True, mat_dtype=True)
    mat_v = scipy.io.loadmat(file, variable_names=['v'], squeeze_me=True, mat_dtype=True)
    names_files.append("velo_from_" + file.stem)
    u_velocity = np.array(mat_u['u'])
    v_velocity = np.array(mat_v['v'])
    
    temp_turb_obj = Turbulence_Parameters(filename=file.stem, u_velo=mat_u['u'][4000000:], v_velo=mat_v['v'][4000000:],
                                          freestream_velo=np.mean(u_velocity), Rossby_num=file_Ro,
                                          shaft_speed_std_dev=file_shaftSpeedSTD)
    
    
    # Plot Unfiltered Velocity
    unfiltered_velo_fluct_plot.set_xdata(temp_turb_obj.get_timestamps())
    unfiltered_velo_fluct_plot.set_ydata(temp_turb_obj.get_u_velo_fluct())
    
    temp_turb_obj.calc_turb_psd_spectrum()
    
    # Plot Unfiltered Spectrum
    unfiltered_psd_plot.set_xdata(temp_turb_obj.freq_non_dim)
    unfiltered_psd_plot.set_ydata(temp_turb_obj.E_u)
    
    
    temp_turb_obj.filter_velo()
    
    # Plot Filtered Velocity
    filtered_velo_fluct_plot.set_xdata(temp_turb_obj.get_timestamps())
    filtered_velo_fluct_plot.set_ydata(temp_turb_obj.get_u_velo_fluct())
    
    temp_turb_obj.calc_turb_psd_spectrum()
    
    # Plot Filtered Spectrum
    filtered_psd_plot.set_xdata(temp_turb_obj.freq_non_dim)
    filtered_psd_plot.set_ydata(temp_turb_obj.E_u)
    
    # Adjust axis limits
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw()
    plt.pause(0.01)
    
plt.show()
