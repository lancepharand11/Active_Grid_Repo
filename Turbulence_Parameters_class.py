# Turbulence Parameters Class
# Author: Lance Pharand, 2024
# NOTE(s):
# IMPORTANT Set class variables
# Integral Length calculation based on Taylor Frozen turbulence hypothesis. Used auto correlation w/ zero crossing
# Turbulence PSD obtained using welch method
# For Turbulence Intensity, assumed w^2 = v^2

import numpy as np
import math
from scipy import integrate
from scipy import optimize
from statsmodels.tsa.stattools import acf
import scipy.signal as signal

class Turbulence_Parameters:
    # MUST BE SET by user
    fs = 0
    N_samples = 0

    # DEFAULT: Don't need to be set
    kinematicVisc_Air = 1.516e-5 # m^2/s, Used air at 20 C
    overlap = 0.5
    mesh_length = 0.06096 # [m] grid mesh length

    def __init__(self, filename, u_velo, v_velo, freestream_velo, Rossby_num, shaft_speed_std_dev):
        # Read in attributes
        self.trial_name = filename
        self.u_velo = u_velo
        self.v_velo = v_velo
        self.freestream_velo = freestream_velo
        self.Rossby_num = Rossby_num
        self.shaft_speed_std_dev = shaft_speed_std_dev * self.mesh_length / self.freestream_velo # non-dim

        # Calculated attributes
        self.u_velo_fluct = self.u_velo - np.mean(self.u_velo)
        self.v_velo_fluct = self.v_velo - np.mean(self.v_velo)
        self.grid_Re = self.freestream_velo * self.mesh_length / self.kinematicVisc_Air
        self.turb_int = 0
        self.L_ux_non_dim = 0
        self.E_u = []
        # self.wavenums = []
        self.freq_non_dim = []
        # self.freq_vals_psd = []

    #########################################################################
    ## Get methods
    #########################################################################
    def get_trial_name(self):
        return self.trial_name

    def get_u_velo(self):
        return self.u_velo

    def get_v_velo(self):
        return self.v_velo

    def get_freestream_velo(self):
        return self.freestream_velo

    def get_Rossby_num(self):
        return self.Rossby_num

    def get_shaft_speed_std_dev(self):
        return self.shaft_speed_std_dev

    def get_u_velo_fluct(self):
        return self.u_velo_fluct

    def get_v_velo_fluct(self):
        return self.v_velo_fluct

    def get_grid_Re(self):
        return self.grid_Re

    def get_turb_int(self):
        return self.turb_int

    def get_L_ux_non_dim(self):
        return self.L_ux_non_dim

    def get_E_u(self):
        return self.E_u

    # def get_wavenums(self):
    #     return self.wavenums

    def get_freq_non_dim(self):
        return self.freq_non_dim

    # def get_freq_psd(self):
    #     return self.freq_vals_psd

    #########################################################################
    ## Private methods for the class
    #########################################################################
    def __auto_corr_cutoff(self, data, overlap, mode):

        if np.asarray(data).ndim != 1:
            raise ValueError("Data must be 1-dimensional array")

        M_pperseg = -1  # Number of points in each segment or batch size

        auto_corr_vals = acf(data, nlags=(len(data) - 1), fft=True)
        zero_crossings_index = np.where(np.diff(np.sign(auto_corr_vals)))[0] + 1

        if zero_crossings_index.size != 0:
            M_pperseg = zero_crossings_index[0]  # use first zero crossing index
        else:
            raise ValueError("No zero crossing detected in auto-correlation coefficients")

        S_pinshift = overlap * M_pperseg  # S = Number of points to shift between segments
        # K_numsegs = len(data) / M_pperseg # K = Number of segments or batches

        if mode == 'segments':
            return M_pperseg, S_pinshift
        elif mode == 'normal':
            return M_pperseg, auto_corr_vals[:M_pperseg]
        else:
            raise ValueError("mode must be either 'segments' or 'normal'")

    def __exp_fit_auto_corr(self, x, alpha):
        return np.exp(-alpha * x)

    #########################################################################
    ## Methods
    #########################################################################
    def calc_L_ux(self):
        num_lags, R_ux = self.__auto_corr_cutoff(data=self.u_velo_fluct, overlap=self.overlap, mode='normal')
        params, pcov = optimize.curve_fit(f=self.__exp_fit_auto_corr, xdata=range(num_lags), ydata=R_ux, p0=(0.5),
                                         check_finite=True)
        alpha_opt = params[0]
        R_ux_fit = self.__exp_fit_auto_corr(range(num_lags), alpha=alpha_opt)

        time_lags = np.linspace(0, num_lags, num_lags) * (1 / self.fs)
        L_ux_fit = np.mean(self.u_velo) * integrate.trapezoid(y=R_ux_fit, x=time_lags)

        self.L_ux_non_dim = (L_ux_fit / self.mesh_length)
        # print(f"Integral length scale based on correlation coeff: {L_ux_fit} [m]")

    def calc_turb_psd_spectrum(self):
        if self.u_velo_fluct.size != self.v_velo_fluct.size:
            raise ValueError("Velocity fluctuations must have the same length")

        # For saving arithmetic operations and providing smoothing to PSD
        u_M_pperseg, u_S_pinshift = self.__auto_corr_cutoff(data=self.u_velo, overlap=self.overlap, mode='segments')
        v_M_pperseg, v_S_pinshift = self.__auto_corr_cutoff(data=self.v_velo, overlap=self.overlap, mode='segments')

        if u_M_pperseg > v_M_pperseg:
            if u_M_pperseg > len(self.v_velo_fluct):
                raise ValueError("Segment length is longer than one of the arrays")
            M = u_M_pperseg
            S = u_S_pinshift
        else:
            if v_M_pperseg > len(self.u_velo_fluct):
                raise ValueError("Segment length is longer than one of the arrays")
            M = v_M_pperseg
            S = v_S_pinshift

        freq_psd, E_u_psd = signal.welch(self.u_velo_fluct, window='hamming', fs=self.fs, nperseg=M, noverlap=S/M, scaling='density')
        freq_psd, E_v_psd = signal.welch(self.v_velo_fluct, window='hamming', fs=self.fs, nperseg=M, noverlap=S/M, scaling='density')

        # self.freq_vals_psd = freq_psd
        wavenums = 2 * math.pi * freq_psd / self.freestream_velo # m^-1
        self.freq_non_dim = wavenums * self.mesh_length

        # 1D longitudinal energy spectrum (non-dim):
        self.E_u = E_u_psd / (self.freestream_velo * self.mesh_length)
        # Pre-multiplied by wavenumber:
        # self.E_u = wavenums * E_u_psd / np.var(self.u_velo)

        # # NEED TO work on this further:
        # self.E_k = 2 * np.pi * (wavenums ** 2) * (abs(E_u_psd) ** 2)
        # Not premultiplied:
        # self.E_k_non_dim = 0.5 * (E_u_psd + (2 * E_v_psd)) # assuming var(v_velo) = var(w_velo)
        # Pre-multiplied by wavenumber:
        # self.E_k = (0.5 * (E_u_psd + (2 * E_v_psd))) * wavenums  # assuming var(v_velo) = var(w_velo)

    def calc_turb_intensity(self):
        u_temp_data = np.array(self.u_velo).T
        v_temp_data = np.array(self.v_velo).T
        q_var = np.var(u_temp_data, axis=0) + (2 * np.var(v_temp_data, axis=0)) # assuming v^2 = w^2

        self.turb_int = np.sqrt(q_var) / (self.freestream_velo * math.sqrt(3))



###################################################################
## OLD/Extra
###################################################################

## Old Turbulence PSD method. Changed to Welch method since shorter runtime
# #Depending on if num samples is even or odd, do the following to capture Nyquist freq
        # if N_samples % 2 == 0:
        #     fft_freq = np.linspace(0, fs / 2, (N_samples // 2) + 1) # plus 1 to capture Nyquist freq
        #     E_u = np.fft.fft(u_fluct)
        #     E_u_psd = (np.abs(E_u) ** 2) / (fs * N_samples)  # 2 sided PSD (divided Power spectrum by ENBW)
        #     E_u_psd = E_u_psd[:((N_samples // 2) + 1)] # 1 sided, plus 1 to capture Nyquist freq
        #     E_u_psd[1:-1] = 2 * E_u_psd[1:-1]  # double values except for dc gain and nyquist freq
        #
        #     E_v = np.fft.fft(v_fluct)
        #     E_v_psd = (np.abs(E_v) ** 2) / (fs * N_samples)  # 2 sided PSD (divided Power spectrum by ENBW)
        #     E_v_psd = E_v_psd[:((N_samples // 2) + 1)]  # 1 sided, plus 1 accounts for Nyquist freq
        #     E_v_psd[1:-1] = 2 * E_v_psd[1:-1]  # double values except for dc gain and Nyquist freq
        # else:
        #     fft_freq = np.linspace(0, fs / 2, math.ceil(N_samples / 2)) # rounded up to capture Nyquist Freq
        #     E_u = np.fft.fft(u_fluct)
        #     E_u_psd = (np.abs(E_u) ** 2) / (fs * N_samples) # 2 sided PSD (divided Power spectrum by ENBW)
        #     E_u_psd = E_u_psd[:math.ceil(N_samples / 2)]  # 1 sided, rounded up to capture Nyquist Freq
        #     E_u_psd[1:-1] = 2 * E_u_psd[1:-1]  # double values except for dc gain and nyquist freq
        #
        #     E_v = np.fft.fft(v_fluct)
        #     E_v_psd = (np.abs(E_v) ** 2) / (fs * N_samples)  # 2 sided PSD (divided Power spectrum by ENBW)
        #     E_v_psd = E_v_psd[:math.ceil(N_samples / 2)]  # 1 sided, rounded up to capture Nyquist Freq
        #     E_v_psd[1:-1] = 2 * E_v_psd[1:-1]  # double values except for dc gain and nyquist freq