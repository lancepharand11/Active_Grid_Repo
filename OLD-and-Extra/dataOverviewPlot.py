
import pandas as pd
from plot_statistics_comparision import plot_statistics_comparision
from plot_statistics_surface import plot_statistics_surface
import matplotlib.pyplot as plt

def dataOverviewPlot(turb_data : pd.DataFrame):
    
    import mat73
    import matplotlib.pyplot as plt
    import matplotlib.markers as mkr
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    # Turbulence intensity as percentage
    turb_data["Turbulence Intensity Percent"] = turb_data["Turbulence Intensity"]*100


    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # Turbulence intensity as percentage
    turb_data["Turbulence Intensity Percent"] = turb_data["Turbulence Intensity"] * 100

    # Kinematic viscosity
    nu = 1.5e-5  # [m^2/s]

    fig_statistics_comparison = plot_statistics_comparision(turb_data, nu=nu)
    
    
    fig_statistics_surface = plot_statistics_surface(turb_data, nu=nu)
    
    return fig_statistics_comparison, fig_statistics_surface
