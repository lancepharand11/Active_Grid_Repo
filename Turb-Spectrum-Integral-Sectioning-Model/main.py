import torch
from torch import nn
import numpy as np
import joblib


device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device)

#
# USER Parameters - Load Trained Model and Scalers
#
# IMPORTANT: Must match the saved model's architecture
input_size = 4  # IMPORTANT: Make sure this matches Turbulence_Parameters.num_sections in training file
hidden_size = 128
output_size = 1  # Grid Re only

model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_size, hidden_size // 2),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_size // 2, output_size)
                      ).to(device)
criterion = nn.MSELoss()

# Load the saved model weights
model_path = "./turb_spectrum_integral_section_model.pth"
model.load_state_dict(torch.load(model_path,map_location=device))
model.eval()  # Set the model to evaluation mode

print("Model successfully loaded and ready for inference.")

# Load the scalers
scaler1 = joblib.load("./x_scaler.pkl")
scaler2 = joblib.load("./y_scaler.pkl")

print("Scalers loaded successfully.")

#
# USER Parameter - Define Model Input: (Integral Section 0, Integral Section 1, ...)
#

model_input = np.array([[-9.0, -2.0, -1.5, 1.0]])

#
# Obtain Prediction: Grid Re
#
model_input_tens = torch.tensor(scaler1.transform(model_input), dtype=torch.float32).to(device)
model_output = model(model_input_tens)
print(f"Model output (Grid Re): {scaler2.inverse_transform(model_output.detach().cpu().numpy())}")
