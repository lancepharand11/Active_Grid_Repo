import mat73
import matplotlib.pyplot as plt
import matplotlib.markers as mkr

def plot_statistics_comparision(turb_data, nu=1.5e-5):
    """
    Plots an overview of turbulence data and comparison with previous studies.
    Args:
        turb_data (pd.DataFrame): DataFrame containing turbulence data.
        nu (float): Kinematic viscosity [m^2/s]. Default is 1.5e-5.
    Returns:
        matplotlib.figure.Figure: The resulting figure object.
    """
    # Load .mat files of previous studies
    hearst_ro = mat73.loadmat("Hearst2015_Ro.mat")
    hearst_u = mat73.loadmat("Hearst2015_U.mat")
    larssen = mat73.loadmat("Larssen2011.mat")
    makita = mat73.loadmat("Makita1991.mat")

    hearst_marker = mkr.MarkerStyle('o', fillstyle="none")
    larssen_marker = mkr.MarkerStyle('*', fillstyle="none")
    makita_marker = mkr.MarkerStyle('s', fillstyle="none")
    present_marker = mkr.MarkerStyle('^', fillstyle="none")

    # Create figure with 2x2 layout
    fig, axs = plt.subplots(4, 2, figsize=(5.8, 5.8 * 1.2))
    axs = axs.flatten()

    # Subfigure labels
    subfig_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    for i, ax in enumerate(axs):
        ax.text(0.02, 0.95, subfig_labels[i], transform=ax.transAxes, fontsize=10, va='top', ha='left', fontweight='bold')

    # Tu vs Ro
    turb_data.plot(kind="scatter", x="Rossby Number", y="Turbulence Intensity Percent", ax=axs[0], label='Present Study', marker=present_marker, c="k", linewidths=0.5)
    axs[0].scatter(hearst_ro['Ro'], hearst_ro['Tu'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[0].scatter(hearst_u['Ro'], hearst_u['Tu'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[0].scatter(larssen['Ro'][4:], larssen['Tu'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[0].scatter(makita['Ro'], makita['Tu'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[0].set_xlabel(r"")
    axs[0].set_ylabel(r'$Tu$')
    axs[0].get_legend().remove()
    legend = fig.legend(loc='outside upper center', ncols=2)
    legend.get_frame().set_edgecolor('black')

    # Tu vs Re_M
    turb_data.plot(kind="scatter", x="Grid Re", y="Turbulence Intensity Percent", ax=axs[1], label='Present Study', legend=False, marker=present_marker, c="k", linewidths=0.5)
    axs[1].scatter(hearst_ro['Re_M'], hearst_ro['Tu'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[1].scatter(hearst_u['Re_M'], hearst_u['Tu'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[1].scatter(larssen['uinfty'][4:] * 0.21 / nu, larssen['Tu'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[1].scatter(makita['Re_M'], makita['Tu'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[1].set_ylabel(r'')
    axs[1].set_xlabel(r"")

    # L_ux vs Ro
    turb_data.plot(kind="scatter", x="Rossby Number", y="L_ux / M", ax=axs[2], label='Present Study', legend=False, marker=present_marker, c="k", linewidths=0.5)
    axs[2].scatter(hearst_ro['Ro'], hearst_ro['L_ux'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[2].scatter(hearst_u['Ro'], hearst_u['L_ux'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[2].scatter(larssen['Ro'][4:], larssen['L_ux'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[2].scatter(makita['Ro'], makita['L_ux'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[2].set_ylabel(r'$L_{ux}/M$')
    axs[2].set_xlabel(r"")

    # L_ux vs Re_M
    turb_data.plot(kind="scatter", x="Grid Re", y="L_ux / M", ax=axs[3], label='Present Study', legend=False, marker=present_marker, c="k", linewidths=0.5)
    axs[3].scatter(hearst_ro['Re_M'], hearst_ro['L_ux'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[3].scatter(hearst_u['Re_M'], hearst_u['L_ux'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[3].scatter(larssen['uinfty'][4:] * 0.21 / nu, larssen['L_ux'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[3].scatter(makita['Re_M'], makita['L_ux'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[3].set_ylabel(r'')
    axs[3].set_xlabel(r"")

    # u'/v' vs Ro
    turb_data.plot(kind="scatter", x="Rossby Number", y="Anisotropy", ax=axs[4], label='Present Study', legend=False, marker=present_marker, c="k", linewidths=0.5)
    axs[4].scatter(hearst_ro['Ro'], hearst_ro['u_v'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[4].scatter(hearst_u['Ro'], hearst_u['u_v'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[4].scatter(larssen['Ro'][4:], larssen['u_v'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[4].scatter(makita['Ro'], makita['u_v'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[4].set_ylabel(r"$\sqrt{\overline{u'^2}/\overline{v'^2}}$")
    axs[4].set_xlabel(r"")

    # u'/v' vs Re_M
    turb_data.plot(kind="scatter", x="Grid Re", y="Anisotropy", ax=axs[5], label='Present Study', legend=False, marker=present_marker, c="k", linewidths=0.5)
    axs[5].scatter(hearst_ro['Re_M'], hearst_ro['u_v'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[5].scatter(hearst_u['Re_M'], hearst_u['u_v'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[5].scatter(larssen['uinfty'][4:] * 0.21 / nu, larssen['u_v'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[5].scatter(makita['Re_M'], makita['u_v'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[5].set_ylabel(r"")
    axs[5].set_xlabel(r"")

    # Re_lambda vs Ro
    turb_data.plot(kind="scatter", x="Rossby Number", y="Re_lambda", ax=axs[6], label='Present Study', legend=False, marker=present_marker, c="k", linewidths=0.5)
    axs[6].scatter(hearst_ro['Ro'], hearst_ro["Re_lambda"], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[6].scatter(hearst_u['Ro'], hearst_u["Re_lambda"], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[6].scatter(larssen['Ro'][4:], larssen["Re_lambda"][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[6].scatter(makita['Ro'], makita['Re_lambda'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[6].set_ylabel(r"$\textrm{Re}_\lambda$")
    axs[6].set_xlabel(r"$\textrm{Ro}$")

    # Re_lambda vs Re_M
    turb_data.plot(kind="scatter", x="Grid Re", y="Re_lambda", ax=axs[7], label='Present Study', legend=False, marker=present_marker, c="k", linewidths=0.5)
    axs[7].scatter(hearst_ro['Re_M'], hearst_ro["Re_lambda"], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    axs[7].scatter(hearst_u['Re_M'], hearst_u["Re_lambda"], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    axs[7].scatter(larssen['uinfty'][4:] * 0.21 / nu, larssen["Re_lambda"][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    axs[7].scatter(makita['Re_M'], makita['Re_lambda'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[7].set_xlabel(r"$\textrm{Re}_M$")
    axs[7].set_ylabel(r"")

    # X-Axis Limits
    for x in [1, 3, 5, 7]:
        axs[x].set_xlim(left=0, right=300000)

    for x in [0, 2, 4, 6]:
        axs[x].set_xlim(left=0, right=250)

    # Adjust layout
    plt.subplots_adjust(hspace=0.2, left=0.12, right=0.96, wspace=0.18, bottom=0.06, top=0.91)

    # Set Y-Axis Limits
    axs[0].set_ylim(bottom=4, top=20)
    axs[1].set_ylim(bottom=4, top=20)

    plt.show()
    return fig
