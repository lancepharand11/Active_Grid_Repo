# Initial Turbulence Intensity Model for Active Grid
# Author: Lance Pharand, 2024
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code)
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
Turbulence_Parameters.mesh_length = 0.06096

###################################################################
### Functions and Classes Defs
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


# # To view Turbulence Characteristics of a trial
# # Turb Int TEST
# print(f"Turbulence Intensity: {turb_objects[1].get_turb_int()}")
#
# # L_ux TEST
# print(f"Integral Length scale (L_ux): {turb_objects[1].get_L_ux_non_dim()} M (times mesh length)")
#
# # PSD TEST
# plt.figure(figsize=(10, 8))
# plt.plot(turb_objects[1].get_freq_non_dim(), turb_objects[1].get_E_k_non_dim(), 'b')
# plt.xlabel('k*M')
# # plt.xlabel('Freq (Hz)')
# # plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('k*E_k / var(u)')
# # plt.xlim(0, (30000))
# # plt.ylim(1e-14, 1)
# plt.grid()
# plt.show()


# Load Data into data frame for access
IO_data = pd.DataFrame({"Trial Name": (turb_obj.get_trial_name() for turb_obj in turb_objects),
                        "Grid Re": (turb_obj.get_grid_Re() for turb_obj in turb_objects),
                        "Rossby Number": (turb_obj.get_Rossby_num() for turb_obj in turb_objects),
                        "Shaft Speed Standard Deviation [rev/s]": (turb_obj.get_shaft_speed_std_dev() for turb_obj in turb_objects),
                        "Turbulence Intensity": (turb_obj.get_turb_int() for turb_obj in turb_objects),
                        "L_ux * M": (turb_obj.get_L_ux_non_dim() for turb_obj in turb_objects),
                        "k*E_11 / var(u) [Non-Dim PSD]": (turb_obj.get_E_u().tolist() for turb_obj in turb_objects),
                        "k*M [Non-Dim Freq]": (turb_obj.get_freq_non_dim().tolist() for turb_obj in turb_objects)
                        })


###################################################################
## Data Analysis and Trends
###################################################################

# Plot correlation heat map
data_corr = IO_data.iloc[:, 1:6].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(data_corr, cmap=sns.diverging_palette(0, 255, as_cmap=True), annot=True)
plt.show()

# Look at scatter plots
plt.figure(figsize=(10, 8))
sns.pairplot(IO_data.iloc[:, 1:6], kind="reg", diag_kind="hist", plot_kws={'line_kws':{'color':'red'}})
plt.show()


#########################################################################
## Preprocessing, Import functions, and Define the data
#########################################################################

from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, FunctionTransformer, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import root_mean_squared_error

# Split the data. NOTE: Using only turb intensity for the y
X = IO_data.iloc[:, 1:4]
Y = IO_data.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    stratify=X["Grid Re"], random_state=42)

# NOT NEEDED: Transform y data to reduce - skew
# output_trans = TransformedTargetRegressor(func=np.log, inverse_func=np.exp)
# y_train_trans = output_trans.transform(y_train)

#########################################################################
## Create Single Output Ridge/ElasticNet regression with polynomial features
#########################################################################
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.inspection import permutation_importance

# Create preprocessor
preprocessor = ColumnTransformer(transformers=[
                                ('poly', PolynomialFeatures(degree=3, include_bias=False), ["Rossby Number"])
                                ],
                                remainder='passthrough')

# OLD: Scaler and Normalizer tested. No impact so removed
# ('scaler', StandardScaler(), ["Freestream Velocity [m/s]", "Rossby Number", "Shaft Speed Standard Deviation [rev/s]"]),
# ('scaler', MinMaxScaler(), ["Freestream Velocity [m/s]", "Rossby Number", "Shaft Speed Standard Deviation [rev/s]"]),

# Create pipeline w/ RidgeCV and preprocessor
poly_reg_ridge = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', RidgeCV(alphas=np.linspace(0.1, 1, 100), fit_intercept=True,
                                  scoring='neg_root_mean_squared_error', cv=5))
                                ])

# # Check learning_curves for over or underfitting of model
# train_sizes, train_scores, valid_scores = learning_curve(poly_reg_ridge, X, Y, train_sizes=np.linspace(0.1, 1.0, 50),
#                                                          cv=5, scoring='neg_root_mean_squared_error', random_state=42)
# train_errors = -train_scores.mean(axis=1)
# valid_errors = -valid_scores.mean(axis=1)
# plt.plot(train_sizes, train_errors, 'r-+', label='Training error')
# plt.plot(train_sizes, valid_errors, 'b-', label='Validation error')
# plt.legend(loc='best')
# plt.xlabel('Set Size')
# plt.ylabel('RMSE')
# plt.show()

# # Eval if model is an improvement using normalized rmse as metric (normalized by diff between max and min observed in test set)
# cv_scores_full_model = cross_val_score(poly_reg_elastic, X, Y, scoring='neg_root_mean_squared_error', cv=5)
# nrmse_full_model = np.abs(cv_scores_full_model) / (max(y_test) - min(y_test))

# Fit model and obtain error
poly_reg_ridge.fit(x_train, y_train)
y_pred_full = poly_reg_ridge.predict(x_test)
rmse_full_model = root_mean_squared_error(y_true=y_test, y_pred=y_pred_full)
nrmse_full_model = np.abs(rmse_full_model) / (max(y_test) - min(y_test))
print(f'Model - Score: {poly_reg_ridge.score(x_test, y_test)}')
list_feat = poly_reg_ridge['preprocessor'].get_feature_names_out()

# Plot residuals
plt.figure(figsize=(10, 8))
residuals = y_test - y_pred_full
sns.regplot(x=y_pred_full, y=residuals, ci=None, color='b')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# # Print coefficients
# feat_coeffs = poly_reg_ridge['model'].coef_
# for i, coeff in enumerate(feat_coeffs):
#     print(f'Feature {i} coefficient: {coeff}')

# # Plot/print feature importances
# feat_imp = permutation_importance(poly_reg_ridge, x_train, y_train, scoring='neg_root_mean_squared_error', random_state=42)
# mean_imp = feat_imp.importances_mean
# for i, val in enumerate(mean_imp):
#     print(f'Input Feature ({i}) Importance: {val}')



###################################################################
## EXTRA
###################################################################
## Results
# Full Ridge Model w/o scaling or normalizing - Normalized Root Mean Squared Error: 0.10809219101201785
# Full Ridge Model w/o scaling or normalizing - Score: 0.8643151448427969
# Full Ridge Model w/ StandardScaling: Normalized Root Mean Squared Error= 0.11031650189119191 (NO impact, so removed)
# Full Ridge Model w/ MinMaxScaling: Normalized Root Mean Squared Error= 0.10893274961590171 (NO impact, so removed)


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

#########################################################################
## ORIGINAL: Create Single Output Linear regression with polynomial features
#########################################################################

# # Create preprocessor
# preprocessor = ColumnTransformer(transformers=[
#                                 ('poly', PolynomialFeatures(degree=3, include_bias=False), ["Rossby Number"])
#                                 ],
#                                 remainder='passthrough')
#
# # Create pipeline
# poly_reg = Pipeline([('preprocessor', preprocessor),
#                      ('model', LinearRegression(fit_intercept=True))
#                      ])
#
# # # Create Target transformer (HAD little impact, so removed)
# # poly_reg_trans = TransformedTargetRegressor(regressor=poly_reg,
# #                                             transformer=PowerTransformer(method='yeo-johnson', standardize=False))
#
# # Check learning_curves for over or underfitting of model
# train_sizes, train_scores, valid_scores = learning_curve(poly_reg, X, Y, train_sizes=np.linspace(0.1, 1.0, 50),
#                                                          cv=5, scoring='neg_root_mean_squared_error', random_state=42)
# train_errors = -train_scores.mean(axis=1)
# valid_errors = -valid_scores.mean(axis=1)
# plt.plot(train_sizes, train_errors, 'r-+', label='Training error')
# plt.plot(train_sizes, valid_errors, 'b-', label='Validation error')
# plt.legend(loc='best')
# plt.xlabel('Set Size')
# plt.ylabel('RMSE')
# plt.show()
#
# # # Eval if model is sufficient using normalized rmse as metric (normalized by diff between max and min observed in test set)
# # cv_scores = cross_val_score(poly_reg, X, Y, scoring='neg_root_mean_squared_error', cv=5)
# # nrmse = np.abs(cv_scores) / (max(y_test) - min(y_test))
# # print(f'Normalized Root Mean Squared Error: {np.mean(nrmse)}')
#
# # Results of cv and learning curves are good so fit model
# poly_reg.fit(x_train, y_train)
#
# # Obtain predicitons
# y_pred = poly_reg.predict(x_test)
# rmse_full_model = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
# nrmse_full_model = np.abs(rmse_full_model) / (max(y_test) - min(y_test))
# print(f'Model - Score: {poly_reg.score(x_test, y_test)}')
# list_feat = poly_reg['preprocessor'].get_feature_names_out()
#
# # Plot residuals
# plt.figure(figsize=(10, 8))
# residuals = y_test - y_pred
# sns.regplot(x=y_pred, y=residuals, ci=None, color='b')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.show()


# # Plot predictions vs y test (Turb intensity in this case)
# plt.figure(figsize=(10, 8))
# sns.regplot(x=y_test, y=y_pred, ci=None, color='b')
# plt.xlabel('Actual Turbulence Intensities')
# plt.ylabel('Predicted Turbulence Intensities')
# plt.legend(loc='best')
# plt.show()

# # Feature importances
# feat_imp = poly_reg['model'].coef_
# for i, coeff in enumerate(feat_imp):
#     print(f'Feature {i}: {coeff}')