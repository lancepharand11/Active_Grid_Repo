# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 07:59:33 2025

@author: Connor
"""

import numpy as np
import torch
from torch import nn
import joblib

class IntensityLengthModel:
    
    # Define the model architecture
    # IMPORTANT: Must match the saved model's architecture
    input_size = 3
    hidden_size = 3
    output_size = 2
    n_hidden_layers = 2
    
    def __init__(self, modelPath : str, scaler1Path : str, scaler2Path : str, n_hidden_layers : int):

        ###################################################################
        ## Load Trained Model and Scalers
        ###################################################################

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # IMPORTANT: make sure the model architecture matches what was used in training
        self.model = nn.Sequential(
                             nn.Linear(self.input_size, self.hidden_size),
                             nn.Tanh()).to(self.device)
        
        # Add the hidden layers
        hiddenLayer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
        for layer in range(self.n_hidden_layers-1):
            self.model.extend(hiddenLayer)
        
        # Add the output layer
        self.model.append(nn.Linear(self.hidden_size, self.output_size))
        
        # Load the saved model weights
        self.model.load_state_dict(torch.load(modelPath, map_location=self.device, weights_only=False))
        self.model.eval()  # Set the model to evaluation mode
        
        print("Model successfully loaded and ready for inference.")
        
        # Load the scalers
        self.scaler1 = joblib.load(scaler1Path)
        self.scaler2 = joblib.load(scaler2Path)
        
        print("Scalers loaded successfully.")

    # %% Methods
    
    def evaluate(self, Re_M : np.ndarray, Ro : np.ndarray, sigma : np.ndarray):
        
        if Re_M.shape != Ro.shape :
            raise("Grid motion parameters must have the same shape")
            
        inputShape = Re_M.shape

        Re_M = Re_M.reshape(-1,1)
        Ro = Ro.reshape(-1,1)
        sigma = sigma.reshape(-1,1)

        input_parameters = np.column_stack((Re_M, Ro, sigma))
        scaled_input_parameters = self.scaler1.transform(input_parameters)
        input_tensor = torch.tensor(scaled_input_parameters, dtype=torch.float32, requires_grad=False).to(self.device)
        
        with torch.no_grad():
            predicted_output = self.model(input_tensor)
            predicted_output_unscaled = self.scaler2.inverse_transform(predicted_output.cpu().numpy())
            
        Intensity = predicted_output_unscaled[:,0]
        Length = predicted_output_unscaled[:,1]
        
        Intensity = Intensity.reshape(inputShape)
        Length = Length.reshape(inputShape)
        
        return Intensity, Length
    