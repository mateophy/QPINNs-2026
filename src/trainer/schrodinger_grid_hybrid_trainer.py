import torch
import os
import sys
import numpy as np

# Global path
global_path = os.getcwd()

# Linea adicional para ubicación de path en los scripts
sys.path.append(global_path)

from src.utils.logger                   import Logging
from data.synthetic.schrodinger_dataset import exact_eigenstate
from src.nn.DVPDESolver                 import DVPDESolver

import src.trainer.schrodinger_train as wave_train

# Auxiliar function definition -> magnitude mse loss 
def magnitude_mean(y_pred_r, y_true_r, y_pred_i, y_true_i):

    # Squared differences for each value
    mse_r = (y_pred_r.cpu().detach().numpy() - y_true_r.cpu().detach().numpy())**2
    mse_i = (y_pred_i.cpu().detach().numpy() - y_true_i.cpu().detach().numpy())**2

    # Magnitude mse + norm
    mse = np.sqrt(mse_r + mse_i).mean()

    return mse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input parameters: Iterable
L = 5.0        # dominio espacial [0, L]
T = 0.1        # tiempo final
hbar = 1.0
mass = 1.0
n_level = 1    # nivel (usaremos n=1)
omega  = 1.0
example = 'ho'
potential_fn = 0

if example.lower() != 'box':
    potential_fn = f"lambda t, x : 0.5 * {mass} * ({omega}**2) * (x**2)"

# Equation parameters
eq_params = {
    'example' : example,             # Ejemplo a correr: "BOX", "HO" 
    'L': L,                          # dominio espacial [0, L]
    'T' : T,                         # tiempo final
    'hbar' : hbar,                   # Atomic coordinates = 1
    'mass' : mass,                   # Atomic coordinates = 1
    'n_level' : n_level,             # nivel del pozo (usaremos n=1)
    'omega'   : omega,               # Natural frequency
    'potential_fn' : potential_fn,   # potential function
}

# - General parameters declarations
mode = "hybrid"
num_qubits = 5
output_dim = 2
input_dim = 2
cutoff_dim = 20

# - Definition of grids to train
hidden_dims        = [4**i for i in range(1, 4 +1)]
num_quantum_layers = [layer for layer in range(1, 5 +1)]

# - Typing and arguments definition 
DTYPE = torch.float32 
torch.set_default_dtype(DTYPE)
torch.manual_seed(69420)

args = {
    "batch_size": 64,
    "epochs": 100, 
    "lr": 1E-3,
    "seed": 42,
    "print_every": 10,
    "log_path": "./results/benchmarks/schrodinger/parameter_grid",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    # "hidden_dim": hidden_dim,                     # Depth grid
    # "num_quantum_layers": num_quantum_layers,     # length grid
    # "classic_network": classic_network,
    "q_ansatz": "sim_circ_15",  # options: "alternating_layer_tdcnot", "abbas" , farhi , sim_circ_13_half, sim_circ_13 , sim_circ_14_half, sim_circ_14 , sim_circ_15 ,sim_circ_19
    "mode": mode,
    "activation": "null",  # options: "null", "partial_measurement_half" , partial_measurement_x, tanh (Classical)
    "shots": None,  # Analytical gradients enabled
    "problem": "schrodinger",
    "solver": "DV",  # options : "CV", "Classical", "DV"
    "device": DEVICE,
    "method": "None",
    "cutoff_dim": cutoff_dim,  # num_qubits >= cutoff_dim
    "class": "CVNeuralNetwork2",  # options CVNeuralNetwork1, CVNeuralNetwork2, CVNeuralNetwork3
    "encoding": "None",  # options : "ampiltude" , "angle" for DV , none for others
    "eq_params" : eq_params,
}

# Iteration for depth
for index_nql, num_quantum_layer in enumerate(num_quantum_layers):

    # Restart of score
    best_score = 1;

    # Iteration in number of variables
    for index_hd, hidden_dim in enumerate(hidden_dims):

        # Iteration feedback 
        print(f"Layers = {num_quantum_layer} ({index_nql +1} / {len(num_quantum_layers)}), dim = {hidden_dim} ({index_hd +1} / {len(hidden_dims)})")
        
        # Feedback on number of tasks
        print(f"Number of tasks estimated: ~{hidden_dim * num_quantum_layer * 25 * 20}")

        # Arguments filling
        classic_network = [input_dim, hidden_dim, output_dim];

        args["classic_network"   ] = classic_network
        args["hidden_dim"        ] = hidden_dim 
        args["num_quantum_layers"] = num_quantum_layer

        # Logger definition
        log_path = args["log_path"]
        logger = Logging(log_path)

        # Solver definition
        model = DVPDESolver(args, logger, DEVICE)
        model.logger.print("Using DV Solver")

        # Print total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        model.logger.print(f"Total number of parameters: {total_params}")

        # Training of model
        wave_train.train(model, N_0=5, N_b=5, N_f=20)
        model.save_state()

        model.logger.print(f"Training completed successfuly!, shots made: {model.quantum_layer.shots_done}")

        # Testing of values 
        number_of_points = 10
    
        with torch.no_grad():
            # mesh of t - x evaluation
            # - Create mesh grid with float32
            if example.lower() == 'ho':
                t = np.linspace(0, T, number_of_points, dtype=np.float32)[:, None]
                x = np.linspace(-L, L, number_of_points, dtype=np.float32)[:, None]
            else:
                t = np.linspace(0, T, number_of_points, dtype=np.float32)[:, None]
                x = torch.linspace(0.0, L, number_of_points, device=DEVICE, dtype=DTYPE).unsqueeze(1)
        
            t, x = np.meshgrid(t, x);
         
            t = t.flatten()[:, None]
            x = x.flatten()[:, None]
        
            # - Generation of torch elements
            t_eval = torch.from_numpy(t); x_eval = torch.from_numpy(x)
        
            psi_pred = model(torch.cat((t_eval, x_eval), dim=1))
            psi_r_pred = psi_pred[:, 0:1]
            psi_i_pred = psi_pred[:, 1:2]
        
            psi_r_true, psi_i_true, En = exact_eigenstate(n_level, t_eval, x_eval, L=L, mass=mass, hbar=hbar, omega=omega, example=example)
            
            # MSE values 
            score = magnitude_mean(psi_r_pred, psi_r_true, psi_i_pred, psi_i_true)
            
            # Saving best models       
            if score < best_score:

                # Update score
                best_score = score; best_model = model

    # saving of best model statistics
    total_params = sum(p.numel() for p in best_model.parameters())
    print(f"Best model of parameters: {total_params}")

    best_model.logger.print(f"Model score: {best_score}, under: {best_model.log_path}")

    best_model.save_state()


# End of program sequence 
print("End of grid")
