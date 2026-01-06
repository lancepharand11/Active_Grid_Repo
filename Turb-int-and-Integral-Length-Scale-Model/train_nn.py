import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


import copy


def train_nn(X_train, Y_train, hidden_size=3, n_hidden_layers=1, max_epochs=1000, min_epochs=100, learning_rate=1e-3, batch_size=16, convergence_threshold=1e-3, device=None, plot=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    input_size, output_size = X_train.shape[1], Y_train.shape[1]

    def get_model():
        
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
        
        return nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.Tanh(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.Tanh(),
                             nn.Linear(hidden_size, output_size),
                             ).to(device)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # Train model
    model = get_model()
    training_crit = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    rmse = np.inf
    weights = None

    train_rmse_curve = []

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

            train_rmse = torch.sqrt(training_crit(y_train_pred_unscaled, y_train_unscaled)).item()

            train_rmse_curve.append(train_rmse)
  
                
            if epoch >= min_epochs - 1:
                convergence_metric = np.max(np.abs(np.subtract(train_rmse,train_rmse_curve[epoch-min_epochs+1:epoch+1])))
                
                if convergence_metric < convergence_threshold:

                    break
                
    rmse = train_rmse
    weights = copy.deepcopy(model.state_dict())               
                

    print(f"Number of Epochs: {epoch}, RMSE (unscaled): {rmse:.4f}")

    #
    # Learning curve
    #
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(train_rmse_curve, label="Train RMSE (unscaled)")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return (weights, scaler_x, scaler_y,
            rmse, training_crit, model)