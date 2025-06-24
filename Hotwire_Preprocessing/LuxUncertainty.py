import numpy as np
import scipy.io
import scipy.signal
from scipy.stats import t
from scipy.optimize import curve_fit
from pathlib import Path

# Helper function for cross-correlation coefficient
def cross_correlation_coefficient(x, y, mean_x, mean_y):
    x = x - mean_x
    y = y - mean_y
    corr = scipy.signal.correlate(x, y, mode='full') / (np.std(x) * np.std(y) * len(x))
    lags = np.arange(-len(x) + 1, len(x))
    return corr, lags

# Constants
fs = 20000
c = 0.1

# Hotwire data directory
dataDir = Path('/Users/Connor/Nextcloud/Experimental Data/Active_Grid_Data_Lance')

TurbulenceCases = list(dataDir.glob('*.mat'))

for TuID, dataFile in enumerate(TurbulenceCases):

    # Load measurement data
    measurementData = scipy.io.loadmat(dataFile)
    u = measurementData['u'].flatten()
    v = measurementData['v'].flatten()

    # Segmentation
    lengthData = len(u)
    NSegments = 5
    SegmentLength = lengthData // NSegments
    
    LuxExpFit = np.zeros(NSegments)
    Tu = np.zeros(NSegments)

    for segment in range(NSegments):
        startID = segment * SegmentLength
        endID = startID + SegmentLength
        seg_u = u[startID:endID]
        seg_v = v[startID:endID]
        mean_seg = np.mean(u)
        rho, lags = cross_correlation_coefficient(seg_u, seg_u, mean_seg, mean_seg)
        lags = lags / fs * np.mean(u) / c

        # Exponential fit
        def exp_func(x, a):
            return np.exp(a * x)
        try:
            popt, _ = curve_fit(exp_func, lags[lags>=0], rho[lags>=0], bounds=([-np.inf], [0]), p0=[-1])
            a_fit = popt[0]
        except Exception:
            a_fit = np.nan

        LuxExpFit[segment] = (-1/a_fit if a_fit != 0 else np.nan)
        q_var = np.var(seg_u) + 2 * np.var(seg_v)
        Tu[segment] = np.sqrt(q_var) / (mean_seg * np.sqrt(3))

    print(f"\nConditions: {dataFile.stem}")
    print(f"Turbulence Intensity = {np.nanmean(Tu)}c")
    uncertaintyTu = t.ppf(0.975, NSegments-1) * np.nanstd(Tu) / np.sqrt(NSegments)
    print(f"Uncertainty = {uncertaintyTu}c")
    print(f"Integral length scale from exponential fit = {np.nanmean(LuxExpFit)}c")
    uncertaintyLux = t.ppf(0.975, NSegments-1) * np.nanstd(LuxExpFit) / np.sqrt(NSegments)
    print(f"Uncertainty = {uncertaintyLux}c")
