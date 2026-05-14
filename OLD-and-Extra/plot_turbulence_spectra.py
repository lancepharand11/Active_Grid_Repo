# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:07:20 2026

@author: ctoppings
"""

import matplotlib.pyplot as plt
import pandas as pd
from Turbulence_Parameters_class import Turbulence_Parameters
from pathlib import Path
import scipy.io
import numpy as np

def plot_turbulence_spectra(turb_data, dataDir, nu=1.5e-5):
    return