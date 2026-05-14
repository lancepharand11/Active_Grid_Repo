# Turbulence Parameters Class
# Author: Lance Pharand, 2024
# NOTE(s):
# IMPORTANT Set class variables
# Integral Length calculation based on Taylor Frozen turbulence hypothesis. Used auto correlation w/ zero crossing
# Turbulence PSD obtained using welch method

import numpy as np
import math
from scipy import integrate
from scipy import optimize
from scipy import signal
from scipy import stats
from statsmodels.tsa.stattools import acf
import scipy.signal as signal


class Turbulence_Parameters:
    # MUST BE SET by user
    fs = 25600 # Sampling frequency
    num_sections = 5  # Default number of sections for psd integration
    mesh_length = 0.06096  # [m] grid mesh length
    alpha = 0.00332509 # Assumed calibration uncertainty parameter alpha (See Yavuzkurt 1985)
    beta = 0.00360652  # Assumed calibration uncertainty parameter beta (See Yavuzkurt 1985)
    k = 1 # Assumed pitch coefficient of hot-wire anemometer

    # This is for turb spectrum gen model
    # DEFAULT: Tuned based on the collected dataset.
    # MAY NEED TO BE TUNED BY USER for different dataset
    psd_inertial_slope_threshold = 1.25  # NOTE: This is for a log-log graph
    psd_dissip_slope_threshold = 3.0  # NOTE: This is for a log-log graph
    kernel_size = 25  # This is for the median filter which further smoothens PSD. Should be an odd number

    # DEFAULT: Don't need to be set
    kinematicVisc_Air = 1.516e-5  # m^2/s, Used air at 20 C
    overlap = 0.5

    def __init__(self, filename, u_velo, v_velo, freestream_velo, Rossby_num, shaft_speed_std_dev):
        # Read in attributes
        self._trial_name = filename
        self._u_velo = u_velo
        self._v_velo = v_velo
        self._freestream_velo = freestream_velo
        self._Rossby_num = Rossby_num
        self._shaft_speed_std_dev = shaft_speed_std_dev * self.mesh_length / self._freestream_velo # non-dim

        # Calculated attributes
        self._u_velo_fluct = self._u_velo - np.mean(self._u_velo)
        self._v_velo_fluct = self._v_velo - np.mean(self._v_velo)
        self._grid_Re = self._freestream_velo * self.mesh_length / self.kinematicVisc_Air
        self.Re_lambda = 0
        self.freq_non_dim = []
        self.E_u = []
        self.__N_samples()
        # self.wavenums = []
        # self.freq_vals_psd = []

        # PSD attributes for NN
        self.log_E_u = []
        self.log_freq_non_dim = []
        self.zero_freq = -1
        self.e_zero_freq = 0
        self.breakaway_freq_inertial = -1
        self.breakaway_freq_dissip = -1
        self.e_slope = 0
        self.dE_u_dfreq = []
        self.index_breakaway_inertial = 0
        self.index_breakaway_dissip = 0
        self.integral_sections = []

    #########################################################################
    ## Get methods
    #########################################################################
    def get_trial_name(self):
        return self._trial_name
    
    def get_anisotropy(self):
        return self.anisotropy

    def get_u_velo(self):
        return self._u_velo

    def get_v_velo(self):
        return self._v_velo

    def get_freestream_velo(self):
        return self._freestream_velo

    def get_Rossby_num(self):
        return self._Rossby_num

    def get_shaft_speed_std_dev(self):
        return self._shaft_speed_std_dev

    def get_u_velo_fluct(self):
        return self._u_velo_fluct

    def get_v_velo_fluct(self):
        return self._v_velo_fluct

    def get_grid_Re(self):
        return self._grid_Re
    
    def get_Re_lambda(self):
        return self.Re_lambda
    
    def get_epsilon(self):
        return self.epsilon

    def get_turb_int(self):
        return self.turb_int
    
    def get_turb_int_uncertainty(self):
        return self.turb_int_uncertainty
    
    def get_turb_int_precis_uncert(self):
        return self.precis_uncert_Tu

    def get_L_ux_non_dim(self):
        return self.L_ux_non_dim
    
    def get_L_ux_uncertainty(self):
        return self.L_ux_uncertainty

    def get_freq_non_dim(self):
        return self.freq_non_dim

    def get_E_u(self):
        return self.E_u

    def get_log_E_u(self):
        return self.log_E_u

    def get_log_freq_non_dim(self):
        return self.log_freq_non_dim

    def get_zero_freq(self):
        return self.zero_freq

    def get_e_zero_freq(self):
        return self.e_zero_freq

    def get_breakaway_freq_inertial(self):
        return self.breakaway_freq_inertial

    def get_breakaway_freq_dissip(self):
        return self.breakaway_freq_dissip

    def get_e_slope(self):
        return self.e_slope

    def get_dE_u_dfreq(self):
        return self.dE_u_dfreq

    def get_integral_sections(self):
        return self.integral_sections
    
    def get_timestamps(self):
        timeStamps = np.linspace(0,(self.N_samples - 1)/self.fs,self.N_samples)
        return timeStamps

    # def get_wavenums(self):
    #     return self.wavenums

    # def get_freq_psd(self):
    #     return self.freq_vals_psd

    #########################################################################
    ## Private methods for the class
    #########################################################################
    def __N_samples(self):
        self.N_samples = self._u_velo.shape[0]
    
    def __auto_corr_cutoff(self, data, overlap, mode):
        if np.asarray(data).ndim != 1:
            raise ValueError("Data must be 1-dimensional array")

        M_pperseg = -1  # Number of points in each segment or batch size

        auto_corr_vals = acf(data, nlags=(len(data) - 1), fft=True)
        zero_crossings_index = np.where(np.diff(np.sign(auto_corr_vals)))[0] + 1 # Get first zero crossing

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
    
    def __Lux(self,x,length):
        corr = signal.correlate(x, x, mode='full') / (np.std(x) * np.std(x) * length)
        lags = np.arange(-length + 1, length)
        lags = lags / self.fs * self._freestream_velo / self.mesh_length
        positiveLags = lags>=0
        maxlag = 40
        fitlags = np.logical_and(positiveLags,lags <= maxlag)
        try:
            popt, _ = optimize.curve_fit(self.__exp_fit_auto_corr, lags[fitlags], corr[fitlags], bounds=([0], [np.inf]), p0=[1])
            a_fit = popt[0]
        except Exception:
            a_fit = np.nan

        Lux = (1/a_fit if a_fit != 0 else np.nan)
        
        return Lux

    #########################################################################
    ## Mutator methods
    #########################################################################
    def set_u_velo(self, new_u_velo):
        self._u_velo = new_u_velo
        self.__N_samples()
        
    def set_v_velo(self, new_v_velo):
        if self._u_velo_fluct.size != self._v_velo_fluct.size:
            raise ValueError("Velocity fluctuations must have the same length")
        else:
            self._v_velo = new_v_velo

    def calc_L_ux(self, length = None):
        # length argument can be used to calculate the L_ux on a shorter
        # dataset. This is used for the convergence study
        if length is None:
            length = self.N_samples
            
        self.L_ux_non_dim = self.__Lux(self._u_velo_fluct[:length], length)  
        
    def calc_N_eff(self, length = None):
        # length argument can be used to calculate the L_ux on a shorter
        # dataset. This is used for the convergence study
        if length is None:
            length = self.N_samples
            
        self.N_eff = length * self._freestream_velo / (self.L_ux_non_dim * self.mesh_length *self.fs)
        
    def calc_L_ux_Uncertainty(self, length = None):
        # Bootstrap(-like) method on 5 shorter segments
        # length argument can be used to calculate the L_ux on a shorter
        # dataset. This is used for the convergence study
        if length is None:
            length = self.N_samples
            
        NSegments = 5
        SegmentLength = length // NSegments
        
        LuxExpFit = np.zeros(NSegments)

        for segment in range(NSegments):
            startID = segment * SegmentLength
            endID = startID + SegmentLength
            seg_u = self._u_velo_fluct[startID:endID]

            LuxExpFit[segment] = self.__Lux(seg_u, SegmentLength)
        
        self.L_ux_uncertainty = stats.t.ppf(0.975, NSegments-1) * np.nanstd(LuxExpFit) / np.sqrt(NSegments)

    def calc_turb_psd_spectrum(self):
        if self._u_velo_fluct.size != self._v_velo_fluct.size:
            raise ValueError("Velocity fluctuations must have the same length")

        # For saving arithmetic operations and providing smoothing to PSD
        u_M_pperseg, u_S_pinshift = self.__auto_corr_cutoff(data=self._u_velo, overlap=self.overlap, mode='segments')
        v_M_pperseg, v_S_pinshift = self.__auto_corr_cutoff(data=self._v_velo, overlap=self.overlap, mode='segments')

        if u_M_pperseg > v_M_pperseg:
            if u_M_pperseg > len(self._v_velo_fluct):
                raise ValueError("Segment length is longer than one of the arrays")
            M = u_M_pperseg
            S = u_S_pinshift
        else:
            if v_M_pperseg > len(self._u_velo_fluct):
                raise ValueError("Segment length is longer than one of the arrays")
            M = v_M_pperseg
            S = v_S_pinshift

        u_freq_psd, E_u_psd = signal.welch(self._u_velo_fluct, window='hamming', fs=self.fs, nperseg=M, noverlap=S / M, scaling='density')
        # v_freq_psd, E_v_psd = signal.welch(self._v_velo_fluct, window='hamming', fs=self.fs, nperseg=M, noverlap=S / M, scaling='density')

        # self.freq_vals_psd = freq_psd
        # wavenums = 2 * math.pi * freq_psd / self._freestream_velo # m^-1
        # self.freq_non_dim = wavenums * self.mesh_length
        self.freq_non_dim = (u_freq_psd * self.mesh_length) / self._freestream_velo

        # 1D longitudinal energy spectrum (non-dim):
        self.E_u = E_u_psd / (self.mesh_length * self._freestream_velo)

        # PSD attributes for NN
        self.log_freq_non_dim = np.log10(self.freq_non_dim)
        self.log_E_u = np.log10(self.E_u)

        # Check for and remove infinite values
        finite_mask = np.isfinite(self.log_freq_non_dim)
        self.log_E_u = self.log_E_u[finite_mask]
        self.log_freq_non_dim = self.log_freq_non_dim[finite_mask]

        # More smoothing for PSD so np.gradient doesn't generate large outlier values
        # self.log_E_u = signal.medfilt(self.log_E_u, kernel_size=self.kernel_size)

        # More PSD attributes for NN
        self.dE_u_dfreq = np.gradient(self.log_E_u, self.log_freq_non_dim)
        # Median filter on the slope
        self.dE_u_dfreq = signal.medfilt(self.dE_u_dfreq, kernel_size=11)

        self.zero_freq = self.log_freq_non_dim[0]
        self.e_zero_freq = self.log_E_u[2]
        
    def __calc_q_var(self, length = None):
        # length argument can be used to calculate the L_ux on a shorter
        # dataset. This is used for the convergence study
        if length is None:
            length = self.N_samples
            
        u_temp_data = np.array(self._u_velo[:length]).T
        v_temp_data = np.array(self._v_velo[:length]).T
        q_var = np.var(u_temp_data, axis=0) + (2 * np.var(v_temp_data, axis=0))
        return q_var

    def calc_turb_intensity(self, length = None):
        # length argument can be used to calculate the L_ux on a shorter
        # dataset. This is used for the convergence study
        if length is None:
            length = self.N_samples

        self.turb_int = np.sqrt(self.__calc_q_var(length)) / (self._freestream_velo * math.sqrt(3))

    def calc_turb_int_uncertainty(self, length = None):
        # length argument can be used to calculate the L_ux on a shorter
        # dataset. This is used for the convergence study
        if length is None:
            length = self.N_samples
            
        # Calculate mean and RMS of U
        u_rms = np.std(self._u_velo[:length], ddof=0).item()  # Use ddof=0 for population standard deviation
        v_rms = np.std(self._v_velo[:length], ddof=0).item()  # Use ddof=0 for population standard deviation

        # Relative approximation errors
        rel_error_u_mean = self.k**2/2*(v_rms / self._freestream_velo)**2 # Only account for w component because we are using an x-wire
        rel_error_u_rms = self.k**2/4*(v_rms**2 / (u_rms*self._freestream_velo)) # Only account for w component because we are using an x-wire

        # Relative uncertainties (Δu_rms/u_rms and Δu_mean/u_mean)
        rel_uncert_u_rms = np.sqrt(self.alpha**2 + self.beta**2) + rel_error_u_mean  # Yavuzkurt: add instrument uncertainty and approximation error
        rel_uncert_u_mean = np.sqrt(self.alpha**2 + self.beta**2) + rel_error_u_rms

        # Propagate uncertainty to Tu
        rel_uncert_Tu = np.sqrt(rel_uncert_u_rms**2 + rel_uncert_u_mean**2).item()
        self.turb_int_uncertainty = 2 * self.turb_int * rel_uncert_Tu  # 95% confidence
        
        # Precision uncertainty
        self.calc_N_eff(length)
        precis_uncert_u = stats.t.ppf(0.975,self.N_eff-1) * u_rms / np.sqrt(self.N_eff)
        # precis_uncert_v = stats.t.ppf(0.975,self.N_eff-1) * v_rms / np.sqrt(self.N_eff)
        
        precis_uncert_u_var = stats.t.ppf(0.975,self.N_eff-1) * u_rms**2 * np.sqrt(2/(self.N_eff-1))
        precis_uncert_v_var = stats.t.ppf(0.975,self.N_eff-1) * v_rms**2 * np.sqrt(2/(self.N_eff-1))
        
        precis_uncert_q_var = np.sqrt(precis_uncert_u_var**2 + (2*precis_uncert_v_var)**2)
        self.precis_uncert_Tu = np.sqrt((1/(2*np.sqrt(3)*self._freestream_velo)*precis_uncert_q_var)**2
                                   + (np.sqrt(self.__calc_q_var()/3)/(self._freestream_velo**2)*precis_uncert_u)**2)
        
    def calc_anisotropy(self):
        u_temp_data = np.array(self._u_velo).T
        v_temp_data = np.array(self._v_velo).T
        self.anisotropy = np.std(u_temp_data)/np.std(v_temp_data)

    def psd_breakaway_freq_inertial(self):
        # NOTE: Slope threshold is lower bound here
        self.index_breakaway_inertial = np.argmax(np.abs(self.dE_u_dfreq) > self.psd_inertial_slope_threshold)
        # print(index_breakaway_inertial)
        self.breakaway_freq_inertial = self.log_freq_non_dim[self.index_breakaway_inertial]

    def psd_breakaway_freq_dissip(self):
        # NOTE: Slope threshold is lower bound here
        self.index_breakaway_dissip = np.argmax(np.abs(self.dE_u_dfreq) > self.psd_dissip_slope_threshold)
        # print(index_breakaway_dissip)
        self.breakaway_freq_dissip = self.log_freq_non_dim[self.index_breakaway_dissip]

    def psd_inertial_range_slope(self):
        # print(f"Inertial breakaway frequency range: {self.index_breakaway_inertial}")
        # print(f"Dissip breakaway frequency range: {self.index_breakaway_dissip}")
        self.e_slope = np.mean(self.dE_u_dfreq[self.index_breakaway_inertial:self.index_breakaway_dissip])

    def psd_integral_sectioning(self):
        section_length = len(self.log_freq_non_dim) // self.num_sections
        areas = []

        # Loop through each section and calculate area under the curve
        for i in range(self.num_sections):
            start_idx = i * section_length

            if i < self.num_sections - 1:
                end_idx = (i + 1) * section_length
            else:
                # Since there could be a remainder of data points in the last section
                end_idx = len(self.log_freq_non_dim)

            # Integrate the PSD over the current section
            area = np.trapz(y=self.log_E_u[start_idx:end_idx],
                            x=self.log_freq_non_dim[start_idx:end_idx])
            areas.append(area)

        self.integral_sections = areas
        
    def calc_dissipation_rate(self):
        timeStamps = self.get_timestamps()
        dudt = np.gradient(self.get_u_velo_fluct(), timeStamps)
        dvdt = np.gradient(self.get_v_velo_fluct(), timeStamps)
        
        uinfty = np.mean(self.get_u_velo())
        
        # Apply Taylor Hypothesis
        dudx = dudt/uinfty
        dvdx = dvdt/uinfty
        
        self.epsilon = 3*self.kinematicVisc_Air*(np.mean(dudx**2)+2*np.mean(dvdx**2))
        
    def calc_kolmogorov_length(self):
        self.calc_dissipation_rate()
        self.eta = self.kinematicVisc_Air**(3/4)/self.epsilon**(1/4)
        
    def filter_velo(self):
        
        # Store the unfiltered velocity
        u_velo_unfiltered =  self.get_u_velo_fluct() 
        v_velo_unfiltered =  self.get_v_velo_fluct() 
        
        # Initialise the cutoff frequency
        cutoff_frequency = [0,1]
        
        # Iterate the filtering procedure until convergence of the cutoff frequency
        while np.absolute(cutoff_frequency[-1]-cutoff_frequency[-2]) > 10**(-3):
        
            # Get Kolmogorov length and freestream velocity for calculating cutoff frequency
            self.calc_kolmogorov_length()
            uinfty = np.mean(self.get_u_velo())
            
            # Filter
            # Design the Butterworth filter
            order = 5
            cutoff_frequency.append(uinfty/(2*np.pi*self.eta))  # Hz
            
            #print(f"Cutoff Frequency:{cutoff_frequency[-1]}")
            
            if cutoff_frequency[-1] < self.fs/2:
            
                b, a = signal.butter(order, cutoff_frequency[-1], btype='lowpass', analog=False, fs=self.fs)
            
                # Filter the velocity fluctuations
                u_velo_filtered = signal.filtfilt(b, a, u_velo_unfiltered)
                v_velo_filtered = signal.filtfilt(b, a, v_velo_unfiltered)
            
                # Assign filtered velocity fluctuations
                self._u_velo_fluct = u_velo_filtered
                self._v_velo_fluct = v_velo_filtered
                
            else:
                break
            
        # Highpass
        # sos = signal.butter(N=4, Wn=0.75, btype='high',fs=self.fs, output='sos')
        
        # self._u_velo_fluct = signal.sosfilt(sos, self._u_velo_fluct)
        # self._v_velo_fluct = signal.sosfilt(sos, self._v_velo_fluct)
            
    def calc_taylor_length(self):
        self.calc_dissipation_rate()
        self.taylor_length = np.sqrt(5*self.kinematicVisc_Air*self.__calc_q_var()/self.epsilon)
        
    def calc_Re_lambda(self):
        self.calc_taylor_length()
        self.Re_lambda = np.sqrt(self.__calc_q_var())*self.taylor_length/(3**(1/2)*self.kinematicVisc_Air)
    
        
        
            
