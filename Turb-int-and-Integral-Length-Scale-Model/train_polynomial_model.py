from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import torch


def train_polynomial_model(X, Y, order, device):
    def get_model():
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=order)),
            ('linear', RidgeCV(fit_intercept=True))
        ])
        return model

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    x_train = scaler_x.fit_transform(X)
    y_train = scaler_y.fit_transform(Y)
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    model = get_model()
    mse_crit = torch.nn.MSELoss()

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_train_pred_unscaled = torch.tensor(scaler_y.inverse_transform(y_train_pred))
    y_train_unscaled = torch.tensor(Y.numpy())
    train_rmse = torch.sqrt(mse_crit(y_train_pred_unscaled, y_train_unscaled)).item()

    return model, scaler_x, scaler_y, train_rmse