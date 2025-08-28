import mat73
import matplotlib.pyplot as plt
import matplotlib.markers as mkr
import numpy as np
import matplotlib as mpl

def plot_statistics_surface(turb_data, nu=1.5e-5):
    """
    Plots an overview of turbulence data and comparison with previous studies.
    Args:
        turb_data (pd.DataFrame): DataFrame containing turbulence data.
        nu (float): Kinematic viscosity [m^2/s]. Default is 1.5e-5.
    Returns:
        matplotlib.figure.Figure: The resulting figure object.
    """

    # Create figure with 2x2 layout
    fig, axs = plt.subplots(4, 1, figsize=(5.8, 5.8 * 1.2))
    axs = axs.flatten()
    
    # Colourmap
    viridis = mpl.colormaps['viridis'].resampled(8)

    # Subfigure labels
    subfig_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axs):
        ax.text(0.02, 0.95, subfig_labels[i], transform=ax.transAxes, fontsize=10, va='top', ha='left', fontweight='bold')

    # Shaft Speed Standard Deviation * M^2/nu
    turb_data["Shaft Speed Standard Deviation * M^2/nu"] = np.multiply(turb_data["Shaft Speed Standard Deviation * M / u_inf"], turb_data["Grid Re"])

    # Select Shaft Speed Standard Deviation
    turb_data_for_contour = turb_data[turb_data["Shaft Speed Standard Deviation * M^2/nu"] > 0]
    
    # Tu vs Ro and Re_M
    scatter_Tu = axs[0].scatter(turb_data_for_contour["Grid Re"].to_numpy(),
                   turb_data_for_contour["Rossby Number"].to_numpy(),
                   c=turb_data_for_contour["Turbulence Intensity Percent"].to_numpy(),
                   cmap=viridis,
                   vmin=10,
                   vmax=14)
    axs[0].set_ylabel(r"$\textrm{Ro}$")
    fig.colorbar(scatter_Tu, ax=axs[0], location="right", label=r"$Tu$ [\%]")

    # L_ux vs Ro and Re_M
    scatter_Lux = axs[1].scatter(turb_data_for_contour["Grid Re"].to_numpy(),
                   turb_data_for_contour["Rossby Number"].to_numpy(),
                   c=turb_data_for_contour["L_ux / M"].to_numpy(),
                   cmap=viridis,
                   vmin=2.5,
                   vmax=4.5)
    axs[1].set_ylabel(r"$\textrm{Ro}$")
    fig.colorbar(scatter_Lux, ax=axs[1], location="right", label=r"$L_{ux}/M$")

    # u'/v' vs Ro and Re_M
    scatter_u_v = axs[2].scatter(turb_data_for_contour["Grid Re"].to_numpy(),
                   turb_data_for_contour["Rossby Number"].to_numpy(),
                   c=turb_data_for_contour["Anisotropy"].to_numpy(),
                   cmap=viridis,
                   vmin=1,
                   vmax=1.4)
    axs[2].set_ylabel(r"$\textrm{Ro}$")
    fig.colorbar(scatter_u_v, ax=axs[2], location="right", label=r"$\sqrt{\overline{u'^2}/\overline{v'^2}}$")
    
    # Re_lambda vs Ro and Re_M
    scatter_Re_lambda = axs[3].scatter(turb_data_for_contour["Grid Re"].to_numpy(),
                   turb_data_for_contour["Rossby Number"].to_numpy(),
                   c=turb_data_for_contour["Re_lambda"].to_numpy(),
                   cmap=viridis,
                   vmin=200,
                   vmax=800)
    axs[3].set_xlabel(r"$\textrm{Re}_M$")
    axs[3].set_ylabel(r"$\textrm{Ro}$")
    fig.colorbar(scatter_Re_lambda, ax=axs[3], location="right", label=r"$\textrm{Re}_\lambda$")
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.2, left=0.12, right=0.96, wspace=0.18, bottom=0.06, top=0.94)



    plt.show()
    return fig
