# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 07:59:33 2025

@author: Connor
"""

import numpy as np
import torch
import joblib

class IntensityLengthModel:
    
    def __init__(self, modelPath : str, scaler1Path : str, scaler2Path : str):

        ###################################################################
        ## Load Trained Model and Scalers
        ###################################################################
        
        self.model = torch.load(modelPath)
        
        print("Model successfully loaded and ready for inference.")
        
        # Load the scalers
        self.scaler1 = joblib.load(scaler1Path)
        self.scaler2 = joblib.load(scaler2Path)
        
        print("Scalers loaded successfully.")

    # %% Methods
    
    def evaluate(self, Re_M : np.ndarray, Ro : np.ndarray, sigma : np.ndarray):
        
        if Re_M.shape != Ro.shape != sigma.shape:
            raise("Grid motion parameters must have the same shape")
            
        inputShape = Re_M.shape

        Re_M = Re_M.reshape(-1,1)
        Ro = Ro.reshape(-1,1)
        sigma = sigma.reshape(-1,1)

        input_parameters = np.column_stack((Re_M, Ro, sigma))
        scaled_input_parameters = self.scaler1.transform(input_parameters)
        input_tensor = torch.tensor(scaled_input_parameters, dtype=torch.float32, requires_grad=False)
        
        with torch.no_grad():
            predicted_output = self.model.predict(input_tensor)
            predicted_output_unscaled = self.scaler2.inverse_transform(predicted_output)
            
        Intensity = predicted_output_unscaled[:,0]
        Length = predicted_output_unscaled[:,1]
        
        Intensity = Intensity.reshape(inputShape)
        Length = Length.reshape(inputShape)
        
        return Intensity, Length
    