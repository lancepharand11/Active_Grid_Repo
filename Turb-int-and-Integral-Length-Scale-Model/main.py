# Model for Active Grid Input Generation (predictions for turb int and integral length scale ONLY)
# Author: Lance Pharand, 2024
# NOTEs:
# See ReadMe and License file (Please reference me if you use this code in an academic application)
# Download the following packages below if not installed already

import torch
from torch import nn
import numpy as np
import joblib
import pygad
import pandas as pd

###################################################################
## Define Key Genetic Algo Parameters
###################################################################
results = []
num_rerun = 5  # Number of times to rerun each trial

# param_combinations = list(turb_int, L_ux))
# IMP: These are outputs that we want to generate active grid inputs for
param_combinations = [[0.143, 2.08],
                      [0.135, 2.42],
                      [0.143, 2.89],
                      [0.101, 4.37],
                      [0.0832, 1.575]]

gene_space = [
    {'low': -1.1, 'high': 1.1},  # Range for SCALED Grid Re
    {'low': -1.1, 'high': 1.1},  # Range for SCALED Rossby Number
    {'low': -1, 'high': 1}  # Range for SCALED Shaft Speed Standard Deviation * M / u_inf
]

###################################################################
## Loaded Trained Model and Scalers
###################################################################
# Define the model architecture
# IMPORTANT: Must match the saved model's architecture
input_size = 3
hidden_size = 64
output_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT: make sure the model architecture matches what was used in training
model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.BatchNorm1d(hidden_size),
                      nn.LeakyReLU(),
                      nn.Linear(hidden_size, hidden_size // 2),
                      nn.BatchNorm1d(hidden_size // 2),
                      nn.LeakyReLU(),
                      nn.Linear(hidden_size // 2, output_size)
                      ).to(device)
criterion = nn.MSELoss()

# Load the saved model weights
model_path = "./Models_and_Results/best_model_20250518_113100.pth"
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

print("Model successfully loaded and ready for inference.")

# Load the scalers
scaler1 = joblib.load("./Models_and_Results/scaler_x_20250518_113100.pkl")
scaler2 = joblib.load("./Models_and_Results/scaler_y_20250518_113100.pkl")

print("Scalers loaded successfully.")

###################################################################
## Generate Active Grid Inputs for Experiments
###################################################################
# Parameters for PyGAD
population_size = 150
num_generations = 300


# Define fitness function
def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function to minimize the RMSE between the model's output and the target output.

    Returns: Negative RMSE (since PyGAD maximizes the fitness and we want to minimize RMSE)
    """
    # Convert solution to tensor and pass it through the model
    input_tensor = torch.tensor(solution, dtype=torch.float32, requires_grad=False).to(device)
    input_tensor = input_tensor.view(1, -1)
    predicted_output = model(input_tensor)

    # Calculate UNSCALED RMSE
    predicted_output_unscaled = torch.tensor(
        scaler2.inverse_transform(predicted_output.detach().cpu().numpy().reshape(1, -1))
    ).to(device)

    loss_unscaled = torch.sqrt(criterion(predicted_output_unscaled, target_output_tens))

    return -loss_unscaled.item()


# Run the GA for each parameter combination multiple times and save results
for turb_int, L_ux in param_combinations:
    print(f"Running GA for Turbulence Intensity: {turb_int} and L_ux / M: {L_ux}")

    target_output = np.array([[turb_int, L_ux]])
    target_output_tens = torch.tensor(target_output, dtype=torch.float32,
                                      requires_grad=False).to(device)

    for run in range(num_rerun):
        print(f"GA Run {run + 1} for current parameter combination.")

        # Initialize the GA
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=10,
                               fitness_func=fitness_func,
                               sol_per_pop=population_size,
                               num_genes=input_size,  # Number of input features
                               gene_space=gene_space,
                               mutation_percent_genes=20,
                               mutation_type="random",
                               crossover_type="single_point",
                               # init_range_low=-1.5, # NOTE: Possible for inputs to be smaller than -1 after scaling
                               # init_range_high=1.5, # NOTE: Possible for inputs to be greater than 2 after scaling
                               parent_selection_type="sss",  # Stochastic universal sampling selection
                               crossover_probability=0.7,
                               mutation_probability=0.2)

        # Get the best input and eval the model with it
        ga_instance.run()
        best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
        optimized_input = scaler1.inverse_transform(np.array(best_solution).reshape(1, -1))
        optimized_input_tensor = torch.tensor(best_solution, dtype=torch.float32, requires_grad=False).to(device)
        optimized_input_tensor = optimized_input_tensor.view(1, -1)

        model.eval()
        with torch.no_grad():
            predicted_output = model(optimized_input_tensor)
            predicted_output_unscaled = scaler2.inverse_transform(predicted_output.cpu().numpy().reshape(1, -1))

        # Append results for this GA run
        results.append({
            "Turbulence Intensity Target": turb_int,
            "L_ux / M Target": L_ux,
            "GA Run": run + 1,
            "Active Grid Input from Model": optimized_input.flatten().tolist(),
            "Predicted Output Turbulence Intensity": predicted_output_unscaled[0][0],
            "Predicted Output L_ux / M": predicted_output_unscaled[0][1],
            "Best Fitness (negative RMSE)": best_solution_fitness
        })

# Save results to an Excel file
results_df = pd.DataFrame(results)
output_path = "model_experimental_tests.xlsx"
results_df.to_excel(output_path, index=False)

print(f"Results saved to {output_path}")
