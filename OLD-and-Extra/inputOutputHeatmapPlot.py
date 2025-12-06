# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 13:12:49 2025

@author: ctoppings
"""

import seaborn as sns
import matplotlib.pyplot as plt

def inputOutputHeatmapPlot(IO_data):
    
    # Select core variables
    core_cols = [
        "Grid Re",
        "Rossby Number",
        "Shaft Speed Standard Deviation * M^2 / nu",
        "Turbulence Intensity",
        "L_ux / M",
       # "Anisotropy",
       # "Re_lambda"
    ]

    # Combine and compute statistics
    df_all = IO_data[core_cols]
    stats = df_all.agg(['mean', 'std', 'min', 'median', 'max']).T
    stats.index.name = 'Variable'
    stats.rename_axis(columns='Statistic', inplace=True)
    print(stats.to_markdown())

    plt.rcParams['text.usetex'] = True
    heatmapLabels = [r"$\textrm{Re}_M$",
                     r"$\textrm{Ro}$",
                     r"$\sigma_\Omega M^2/\nu$",
                     r"$Tu$",
                     r"$L_{ux}/M$"
                     #r"$\sqrt{\overline{u'^2}/\overline{v'^2}}$",
                     #r"$\textrm{Re}_{\lambda}$"
                     ]
    
    
    fig, axs = plt.subplots(1, 1, figsize=(5.8, 5.8))
    
    sns.heatmap(df_all.corr(),
                annot=True,
                xticklabels=heatmapLabels,
                yticklabels=heatmapLabels,
                cbar_kws=dict(label="Correlation Coefficient"),
                ax=axs,
                square=True)
    
    plt.show()
    return fig