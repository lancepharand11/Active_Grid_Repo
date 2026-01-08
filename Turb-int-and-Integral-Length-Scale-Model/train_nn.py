import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import copy


def train_nn(model, X_train, Y_train, 
             hidden_size=3, n_hidden_layers=1, 
             max_epochs=1000, min_epochs=100, 
             learning_rate=1e-3, batch_size=16, 
             convergence_threshold=1e-3, device=None, plot=False):

    # Scale and prepare training data
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    scaler_x = scaler_x.fit(X_train)
    scaler_y = scaler_y.fit(Y_train)
    
    x_train = scaler_x.transform(X_train)
    y_train = scaler_y.transform(Y_train)

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    # Training criterion and optimizer
    training_crit = nn.MSELoss()        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for m in model.modules():
        1
        
    # Train model
    rmse = np.inf
    weights = None

    train_mse_curve = []

    for epoch in range(0,max_epochs-1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = training_crit(y_pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_train_pred = model(x_train)
            
            y_train_pred_unscaled = torch.tensor(scaler_y.inverse_transform(y_train_pred.cpu().numpy()))

            y_train_unscaled = torch.tensor(Y_train.numpy())

            train_mse = training_crit(y_train_pred_unscaled, y_train_unscaled).item()

            train_mse_curve.append(train_mse)
  
                
            if epoch >= min_epochs - 1:
                convergence_metric = np.max(np.abs(np.subtract(train_mse,train_mse_curve[epoch-min_epochs+1:epoch+1])))
                
                if convergence_metric < convergence_threshold:

                    break
                
    orig_model = getattr(model, "_orig_mod", None)
    weights = copy.deepcopy(orig_model.state_dict())               
                

    print(f"Number of Epochs: {epoch+1}, RMSE (unscaled): {train_mse:.4f}")

    #
    # Learning curve
    #
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(train_mse_curve, label="Train MSE (unscaled)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return (weights, scaler_x, scaler_y, training_crit)