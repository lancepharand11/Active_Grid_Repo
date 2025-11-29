import numpy as np
import scipy.io
from pathlib import Path

# Assumed Uncertainties at 95% Confidence Level
Up = 0.7  # [Pa] Uncertainty in pressure measurement
Urho = 0.009  # [kg/m^3] Uncertainty in density
UT = 2  # [C] Uncertainty in temperature


c = 0.1  # [m] Airfoil chord
R = 287  # [J/kgK] Gas Constant

fs = 20000  # [Hz] Hotwire sampling frequency
k = 1 # Assumed hotwire pitch sensitivity coefficient

# Load calibration data (adjust path as needed)
calibrationData = scipy.io.loadmat(r'202407101234_Connor.mat')

# Hotwire data directory
dataDir = Path('/Users/ctoppings/Nextcloud/Experimental Data/Active_Grid_Data_Lance')

TurbulenceCases = list(dataDir.glob('*.mat'))

Tu = np.zeros(len(TurbulenceCases))
UTu = np.zeros(len(TurbulenceCases))

for TuID, dataFile in enumerate(TurbulenceCases):
    # Load measurement data
    measurementData = scipy.io.loadmat(dataFile, variable_names=['u','v'], squeeze_me=True, mat_dtype=True)
    u = measurementData['u'].flatten()
    v = measurementData['v'].flatten()

    # Calculate calibration values
    T = calibrationData['T'].flatten()
    pBar = calibrationData['pBar'].flatten()
    referenceVelocity = calibrationData['referenceVelocity'].flatten()
    HW1Voltage = calibrationData['HW1Voltage'].flatten()
    HW2Voltage = calibrationData['HW2Voltage'].flatten()
    # Fit a 4th order polynomial
    HW1Poly_Coeffs = np.polyfit(HW1Voltage, referenceVelocity, 4)
    HW2Poly_Coeffs = np.polyfit(HW2Voltage, referenceVelocity, 4)

    rho = pBar * 1333.2 / (R * (T + 273.15))
    referenceP = 0.5 * rho * np.mean(u) ** 2

    # Uncertainty calculations (Yavuzkurt methodology)
    alpha = 0.5 * np.sqrt((UT/2/(T+273.15))**2 + (Urho/2/rho)**2 + (Up/2/referenceP)**2) # Divide uncertainties by 2 to work at 1-std confidence level
    beta = np.sqrt(np.sum(((referenceVelocity - np.polyval(HW1Poly_Coeffs, HW1Voltage)) / np.polyval(HW1Poly_Coeffs, HW1Voltage))**2) / len(HW1Voltage))
    
    # Calculate mean and RMS of U
    u_mean = np.mean(u).item()
    u_rms = np.std(u, ddof=0).item()  # Use ddof=0 for population standard deviation
    v_rms = np.std(v, ddof=0).item()  # Use ddof=0 for population standard deviation
    q_var = np.var(u, ddof=0).item() + 2 * np.var(v, ddof=0).item()   

    # Relative approximation errors
    rel_error_u_mean = k**2/2*(v_rms / u_mean)**2 # Only account for w component because we are using an x-wire
    rel_error_u_rms = k**2/4*(v_rms**2 / (u_rms*u_mean)) # Only account for w component because we are using an x-wire

    # Relative uncertainties (Δu_rms/u_rms and Δu_mean/u_mean)
    rel_uncert_u_rms = np.sqrt(alpha**2 + beta**2) + rel_error_u_rms  # Yavuzkurt: add instrument uncertainty and approximation error
    rel_uncert_u_mean = np.sqrt(alpha**2 + beta**2) + rel_error_u_mean

    # Because the uncertainties in u_rms and v_rms are the same, we can skip the calculation of uncertainty for q

    # Compute turbulence intensity
    Tu[TuID] = np.sqrt(q_var) / (u_mean * np.sqrt(3))

    # Yavuzkurt: propagate uncertainty in Tu
    rel_uncert_Tu = np.sqrt(rel_uncert_u_rms**2 + rel_uncert_u_mean**2).item()
    UTu[TuID] = 2 * Tu[TuID] * rel_uncert_Tu  # 95% confidence

# Print results
for TuID, dataFile in enumerate(TurbulenceCases):
    print(f"Case: {dataFile.stem}")
    print(f"Turbulence Intensity (Tu): {Tu[TuID]}")
    print(f"Uncertainty (UTu, 95% conf.): {UTu[TuID]}")
