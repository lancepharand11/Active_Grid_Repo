# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 09:00:35 2026

@author: ctoppings
"""

from torch import nn

def get_model(input_size, hidden_size, output_size, n_hidden_layers, device):
        
    # IMPORTANT: make sure the model architecture matches what was used in training
    model = nn.Sequential(
                         nn.Linear(input_size, hidden_size),
                         nn.Tanh()).to(device)
    
    # Add the hidden layers
    hiddenLayer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
    for layer in range(n_hidden_layers-1):
        model.extend(hiddenLayer)
    
    # Add the output layer
    model.append(nn.Linear(hidden_size, output_size))
    
    return model