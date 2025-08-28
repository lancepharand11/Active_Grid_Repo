import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


import copy


def train_nn_kfold(X, Y, k_folds=5, hidden_size=64, num_epochs=1000, learning_rate=1e-3, batch_size=16, device=None, plot=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    kf = KFold(n_splits=k_folds, shuffle=True)  # NOTE: no seed used

    input_size, output_size = X.shape[1], Y.shape[1]

    def get_model():
        return nn.Sequential(nn.Linear(input_size, hidden_size),
                             nn.BatchNorm1d(hidden_size),
                             nn.LeakyReLU(),
                             nn.Linear(hidden_size, hidden_size // 2),
                             nn.BatchNorm1d(hidden_size // 2),
                             nn.LeakyReLU(),
                             nn.Linear(hidden_size // 2, output_size)
                             ).to(device)

    fold_results = []
    best_overall_rmse = np.inf
    best_overall_weights = None
    best_norm_rmse_turb_int = None
    best_norm_rmse_L_ux = None
    best_scaler_x = None
    best_scaler_y = None
    best_overall_train_idx = None
    best_overall_val_idx = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}")
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        x_train = scaler_x.fit_transform(X[train_idx])
        y_train = scaler_y.fit_transform(Y[train_idx])
        x_val = scaler_x.transform(X[val_idx])
        y_val = scaler_y.transform(Y[val_idx])

        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        
        # Avoid error caused by batches with one sample
        if train_idx.size % batch_size == 1:
            batch_size = batch_size - 1

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = get_model()
        training_crit = nn.SmoothL1Loss()
        mse_crit = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        best_rmse, best_epoch = np.inf, -1
        best_weights = None

        train_rmse_curve = []
        val_rmse_curve = []

        for epoch in range(num_epochs):
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
                y_val_pred = model(x_val)

                y_train_pred_unscaled = torch.tensor(scaler_y.inverse_transform(y_train_pred.cpu().numpy()))
                y_val_pred_unscaled = torch.tensor(scaler_y.inverse_transform(y_val_pred.cpu().numpy()))

                y_train_unscaled = torch.tensor(Y[train_idx].numpy())
                y_val_unscaled = torch.tensor(Y[val_idx].numpy())

                train_rmse = torch.sqrt(mse_crit(y_train_pred_unscaled, y_train_unscaled)).item()
                val_rmse = torch.sqrt(mse_crit(y_val_pred_unscaled, y_val_unscaled)).item()

                train_rmse_curve.append(train_rmse)
                val_rmse_curve.append(val_rmse)

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_epoch = epoch
                    best_weights = copy.deepcopy(model.state_dict())

        print(f"Best Epoch: {best_epoch}, Best RMSE (unscaled): {best_rmse:.4f}")
        fold_results.append(best_rmse)

        #
        # Learning curve
        #
        if plot:
            plt.figure(figsize=(10, 8))
            plt.plot(train_rmse_curve, label="Train RMSE (unscaled)")
            plt.plot(val_rmse_curve, label="Val RMSE (unscaled)")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.title(f"Learning Curve - Fold {fold+1}")
            plt.legend()
            plt.grid(True)
            plt.show()

        #
        # Residual Plot
        #
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            y_val_pred_unscaled_np = scaler_y.inverse_transform(y_val_pred.cpu().numpy())
            y_val_unscaled_np = Y[val_idx].numpy()
            residuals = y_val_unscaled_np - y_val_pred_unscaled_np

            # Compute per output RMSE
            rmse_turb_int = torch.sqrt(mse_crit(torch.tensor(y_val_pred_unscaled_np[:, 0]),
                                                              torch.tensor(y_val_unscaled_np[:, 0])).clone().detach()
                                       ).item()
            rmse_L_ux = torch.sqrt(mse_crit(torch.tensor(y_val_pred_unscaled_np[:, 1]),
                                                          torch.tensor(y_val_unscaled_np[:, 1])).clone().detach()
                                   ).item()

            # Normalize based on range
            range_turb_int = Y[0].max() - Y[0].min()
            range_L_ux = Y[1].max() - Y[1].min()
            norm_rmse_turb_int = rmse_turb_int / range_turb_int
            norm_rmse_L_ux = rmse_L_ux / range_L_ux

            print(f"Fold {fold + 1} Normalized RMSEs:")
            print(f"    Turbulence Intensity: {norm_rmse_turb_int:.4f}")
            print(f"    L_ux / M: {norm_rmse_L_ux:.4f}")

            temp_inputs = X[val_idx].detach().numpy()

            if plot:
                for i, target_name in enumerate(["Turbulence Intensity", "L_ux / M"]):
                    fig1 = plt.figure(figsize=(10, 8))
                    ax1 = fig1.add_subplot(111, projection='3d')
                    p1 = ax1.scatter(temp_inputs[:, 0], temp_inputs[:, 1], temp_inputs[:, 2],
                                     c=residuals[:, i], cmap='magma',
                                     marker='o', s=50, alpha=0.8
                                     )
                    cbar1 = fig1.colorbar(p1, ax=ax1, shrink=0.5, pad=0.1)
                    cbar1.set_label('Residuals - ' + target_name)
                    ax1.set_xlabel('Grid Re', labelpad=7)
                    ax1.set_ylabel('Rossby Number')
                    ax1.set_zlabel('Shaft Speed Std Dev * M / u_inf', labelpad=8, rotation=0)
                    ax1.set_title('3D Scatter: ' + target_name)
                    plt.show()

        # Track best model across all folds
        if best_rmse < best_overall_rmse:
            best_overall_rmse = best_rmse
            best_overall_train_idx, best_overall_val_idx = train_idx, val_idx
            best_norm_rmse_turb_int = norm_rmse_turb_int
            best_norm_rmse_L_ux = norm_rmse_L_ux
            best_overall_weights = copy.deepcopy(best_weights)
            best_scaler_x = copy.deepcopy(scaler_x)
            best_scaler_y = copy.deepcopy(scaler_y)

    return (best_overall_weights, best_scaler_x, best_scaler_y,
            best_overall_train_idx, best_overall_val_idx,
            best_overall_rmse, best_norm_rmse_turb_int, best_norm_rmse_L_ux, fold_results, model)