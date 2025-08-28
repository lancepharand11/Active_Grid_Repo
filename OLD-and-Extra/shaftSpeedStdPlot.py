import pandas as pd
import numpy as np

def shaftSpeedStdPlot(turb_data : pd.DataFrame):
    
    import mat73
    import matplotlib.pyplot as plt
    import matplotlib.markers as mkr
    
    # Load .mat files of previous studies
    # hearst_ro = mat73.loadmat("Hearst2015_Ro.mat")
    # hearst_u = mat73.loadmat("Hearst2015_U.mat")
    # larssen = mat73.loadmat("Larssen2011.mat")
    # makita = mat73.loadmat("Makita1991.mat")
    
    # Mesh Size
    # hearst_M = 0.08 # [m]
    # larssen_M = 0.21 # [m]
    # makita_M = 0.046 # [m]
    M = 0.061 # [m]
    
    # Kinematic viscosity
    nu = 1.5e-5  # [m^2/s]
    
    # hearst_marker = mkr.MarkerStyle('o',fillstyle="none")
    # larssen_marker = mkr.MarkerStyle('*',fillstyle="none")
    # makita_marker = mkr.MarkerStyle('s',fillstyle="none")
    present_marker = mkr.MarkerStyle('^',fillstyle="none")

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10;
    
    # Turbulence intensity as percentage
    turb_data["Turbulence Intensity Percent"] = turb_data["Turbulence Intensity"]*100
    turb_data["Shaft Speed Standard Deviation * M^2 / nu"] = np.multiply(turb_data["Shaft Speed Standard Deviation * M / u_inf"], turb_data["Grid Re"])
    constant_re_10000 = (turb_data["Grid Re"] > 5000) & (turb_data["Grid Re"] < 15000)
    constant_re_20000 = (turb_data["Grid Re"] > 15000) & (turb_data["Grid Re"] < 25000)
    constant_re_30000 = (turb_data["Grid Re"] > 25000) & (turb_data["Grid Re"] < 35000)
    constant_re_40000 = (turb_data["Grid Re"] > 35000) & (turb_data["Grid Re"] < 45000)
    constant_re_50000 = (turb_data["Grid Re"] > 45000) & (turb_data["Grid Re"] < 55000)

    constant_ro_25 = (turb_data["Rossby Number"] == 25)
    turb_data_re_10000_ro_25 = turb_data[constant_re_10000 & constant_ro_25]
    turb_data_re_20000_ro_25 = turb_data[constant_re_20000 & constant_ro_25]
    turb_data_re_30000_ro_25 = turb_data[constant_re_30000 & constant_ro_25]
    turb_data_re_40000_ro_25 = turb_data[constant_re_40000 & constant_ro_25]
    turb_data_re_50000_ro_25 = turb_data[constant_re_50000 & constant_ro_25]
    
    # Create figure with 2x2 layout
    fig, axs = plt.subplots(4, 1, figsize=(5.8, 5.8*1.2))
    axs = axs.flatten()
    
    # Subfigure labels
    subfig_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    for i, ax in enumerate(axs):
        ax.text(0.02, 0.95, subfig_labels[i], transform=ax.transAxes, fontsize=10, va='top', ha='left', fontweight='bold')

    # Tu vs sigma_\Omega
    turb_data_re_10000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Turbulence Intensity Percent", ax=axs[0], label=r'$\textrm{Re}_M=1\times10^4$', marker=present_marker, c="g", linewidths=0.5)
    turb_data_re_20000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Turbulence Intensity Percent", ax=axs[0], label=r'$\textrm{Re}_M=2\times10^4$', marker=present_marker, c="b", linewidths=0.5)
    turb_data_re_30000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Turbulence Intensity Percent", ax=axs[0], label=r'$\textrm{Re}_M=3\times10^4$', marker=present_marker, c="k", linewidths=0.5)
    turb_data_re_40000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Turbulence Intensity Percent", ax=axs[0], label=r'$\textrm{Re}_M=4\times10^4$', marker=present_marker, c="r", linewidths=0.5)
    turb_data_re_50000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Turbulence Intensity Percent", ax=axs[0], label=r'$\textrm{Re}_M=5\times10^4$', marker=present_marker, c="y", linewidths=0.5)
    # axs[0].scatter(np.divide(hearst_ro['omega'], hearst_ro['Re_M'])*hearst_M**2/nu , hearst_ro['Tu'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    # axs[0].scatter(np.divide(hearst_u['omega'], hearst_u['Re_M'])*hearst_M**2/nu, hearst_u['Tu'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    # axs[0].scatter(np.divide(larssen['omega'][4:], larssen['Re_M'][4:])*larssen_M**2/nu, larssen['Tu'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    # axs[0].scatter(np.divide(makita['omega'], makita['Re_M'])*makita_M**2/nu, makita['Tu'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[0].set_xlabel(r"")
    axs[0].set_ylabel(r'$Tu$')
    axs[0].get_legend().remove();
    legend = fig.legend(loc='outside upper center', ncols=2)
    legend.get_frame().set_edgecolor('black')
    
    # L_ux vs sigma_\Omega
    turb_data_re_10000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="L_ux / M", ax=axs[1], label=r'_nolegend_', marker=present_marker, c="g", linewidths=0.5)
    turb_data_re_20000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="L_ux / M", ax=axs[1], label=r'_nolegend_', marker=present_marker, c="b", linewidths=0.5)
    turb_data_re_30000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="L_ux / M", ax=axs[1], label=r'_nolegend_', marker=present_marker, c="k", linewidths=0.5)
    turb_data_re_40000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="L_ux / M", ax=axs[1], label=r'_nolegend_', marker=present_marker, c="r", linewidths=0.5)
    turb_data_re_50000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="L_ux / M", ax=axs[1], label=r'_nolegend_', marker=present_marker, c="y", linewidths=0.5)
    # axs[1].scatter(np.divide(hearst_ro['omega'], hearst_ro['Re_M'])*hearst_M**2/nu , hearst_ro['L_ux'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    # axs[1].scatter(np.divide(hearst_u['omega'], hearst_u['Re_M'])*hearst_M**2/nu, hearst_u['L_ux'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    # axs[1].scatter(np.divide(larssen['omega'][4:], larssen['Re_M'][4:])*larssen_M**2/nu, larssen['L_ux'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    # axs[1].scatter(np.divide(makita['omega'], makita['Re_M'])*makita_M**2/nu, makita['L_ux'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[1].set_ylabel(r'$L_{ux}/M$')
    axs[1].set_xlabel(r"")
    
    # u'/v' vs sigma_\Omega
    turb_data_re_10000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Anisotropy", ax=axs[2], label=r'_nolegend_', marker=present_marker, c="g", linewidths=0.5)
    turb_data_re_20000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Anisotropy", ax=axs[2], label=r'_nolegend_', marker=present_marker, c="b", linewidths=0.5)
    turb_data_re_30000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Anisotropy", ax=axs[2], label=r'_nolegend_', marker=present_marker, c="k", linewidths=0.5)
    turb_data_re_40000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Anisotropy", ax=axs[2], label=r'_nolegend_', marker=present_marker, c="r", linewidths=0.5)
    turb_data_re_50000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Anisotropy", ax=axs[2], label=r'_nolegend_', marker=present_marker, c="y", linewidths=0.5)
    # axs[2].scatter(np.divide(hearst_ro['omega'], hearst_ro['Re_M'])*hearst_M**2/nu , hearst_ro['u_v'], c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    # axs[2].scatter(np.divide(hearst_u['omega'], hearst_u['Re_M'])*hearst_M**2/nu, hearst_u['u_v'], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    # axs[2].scatter(np.divide(larssen['omega'][4:], larssen['Re_M'][4:])*larssen_M**2/nu, larssen['u_v'][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    # axs[2].scatter(np.divide(makita['omega'], makita['Re_M'])*makita_M**2/nu, makita['u_v'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[2].set_ylabel(r"$\sqrt{\overline{u'^2}/\overline{v'^2}}$")
    axs[2].set_xlabel(r"")
    
    # Re_lambda vs Ro
    turb_data_re_10000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Re_lambda", ax=axs[3], label=r'_nolegend_', marker=present_marker, c="g", linewidths=0.5)
    turb_data_re_20000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Re_lambda", ax=axs[3], label=r'_nolegend_', marker=present_marker, c="b", linewidths=0.5)
    turb_data_re_30000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Re_lambda", ax=axs[3], label=r'_nolegend_', marker=present_marker, c="k", linewidths=0.5)
    turb_data_re_40000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Re_lambda", ax=axs[3], label=r'_nolegend_', marker=present_marker, c="r", linewidths=0.5)
    turb_data_re_50000_ro_25.plot(kind="scatter", x="Shaft Speed Standard Deviation * M^2 / nu", y="Re_lambda", ax=axs[3], label=r'_nolegend_', marker=present_marker, c="y", linewidths=0.5)
    # axs[3].scatter(np.divide(hearst_ro['omega'], hearst_ro['Re_M'])*hearst_M**2/nu, hearst_ro["Re_lambda"] , c='r', marker=hearst_marker, label=r'Hearst \& Lavoie (2015)', linewidths=0.5)
    # axs[3].scatter(np.divide(hearst_u['omega'], hearst_u['Re_M'])*hearst_M**2/nu, hearst_u["Re_lambda"], c='r', marker=hearst_marker, label='_nolegend_', linewidths=0.5)
    # axs[3].scatter(np.divide(larssen['omega'][4:], larssen['Re_M'][4:])*larssen_M**2/nu, larssen["Re_lambda"][4:], c='b', marker=larssen_marker, label=r'Larssen \& Devenport. (2011)', linewidths=0.5)
    # axs[3].scatter(np.divide(makita['omega'], makita['Re_M'])*makita_M**2/nu, makita['Re_lambda'], c='g', marker=makita_marker, label='Makita (1991)', linewidths=1)
    axs[3].set_ylabel(r"$\textrm{Re}_\lambda$")
    axs[3].set_xlabel(r"$\sigma_\Omega M^2/\nu$")
    
    # X-Axis Limits
    # for x in [0,1,2,3]:
    #     axs[x].set_xlim(left=0, right=300000)
    
    # Adjust layout
    plt.subplots_adjust(hspace=0.2,left=0.12,right=0.96,wspace=0.18,bottom=0.06,top=0.91)
    
    # Set Y-Axis Limits
    axs[0].set_ylim(bottom=4, top=20)
    
    plt.show()
    return fig
