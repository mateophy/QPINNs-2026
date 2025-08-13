import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Global path
global_path = os.getcwd()

# Linea adicional para ubicación de path en los scripts
sys.path.append(global_path)

from src.utils.logger                   import Logging
from src.nn.pde                         import schrodinger_operator
from data.synthetic.schrodinger_dataset import exact_eigenstate
from src.nn.DVPDESolver                 import DVPDESolver
from src.nn.CVPDESolver                 import CVPDESolver
from src.nn.ClassicalSolver2            import ClassicalSolver2

import src.trainer.schrodinger_train as wave_train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "hybrid"
num_qubits = 5
output_dim = 2
input_dim = 2
hidden_dim = 50
num_quantum_layers = 1
cutoff_dim = 20
classic_network = [input_dim, hidden_dim, output_dim]

args = {
    "batch_size": 64,
    "epochs": 200,
    "lr": 0.001,
    "seed": 42,
    "print_every": 100,
    "log_path": "./results/models/checkpoints/wave",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    "hidden_dim": hidden_dim,
    "num_quantum_layers": num_quantum_layers,
    "classic_network": classic_network,
    "q_ansatz": "sim_circ_5",  # options: "alternating_layer_tdcnot", "abbas" , farhi , sim_circ_13_half, sim_circ_13 , sim_circ_14_half, sim_circ_14 , sim_circ_15 ,sim_circ_19
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
}

log_path = args["log_path"]
logger = Logging(log_path)

if args["solver"] == "CV":
    model = CVPDESolver(args, logger, DEVICE)
    model.logger.print("Using CV Solver")
elif args["solver"] == "Classical":
    model = ClassicalSolver2(args, logger, DEVICE)
    model.logger.print("Using Classical Solver")
else:
    model = DVPDESolver(args, logger, DEVICE)
    model.logger.print("Using DV Solver")

model.logger.print(f"The settings used:")
for key, value in args.items():
    model.logger.print(f"{key} : {value}")

# Definición de tipado para entrenamiento
DTYPE = torch.float32 
torch.set_default_dtype(DTYPE)
torch.manual_seed(0)

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
model.logger.print(f"Total number of parameters: {total_params}")

wave_train.train(model)

model.save_state()

model.logger.print("Training completed successfuly!")

# Testing

L = 1.0        # dominio espacial [0, L]
T = 0.2        # tiempo final
hbar = 1.0
mass = 1.0
n_level = 1    # nivel del pozo (usaremos n=1)

with torch.no_grad():
    t_eval = torch.full((800, 1), T, device=DEVICE, dtype=DTYPE)
    x_eval = torch.linspace(0.0, L, 800, device=DEVICE, dtype=DTYPE).unsqueeze(1)

    psi_pred = model(torch.cat((t_eval, x_eval), dim=1))
    psi_r_pred = psi_pred[:, 0:1]
    psi_i_pred = psi_pred[:, 1:2]
    mod2_pred = (psi_r_pred**2 + psi_i_pred**2).squeeze(1).cpu().numpy()

    psi_r_true, psi_i_true, En = exact_eigenstate(n_level, t_eval, x_eval, L=L, mass=mass, hbar=hbar)
    mod2_true = (psi_r_true**2 + psi_i_true**2).squeeze(1).cpu().numpy()

# Gráfico 1: |psi|^2 en t=T (PINN vs exacto)
plt.figure()
plt.plot(x_eval.squeeze(1).cpu().numpy(), mod2_true, label="|ψ|^2 exacto")
plt.plot(x_eval.squeeze(1).cpu().numpy(), mod2_pred, "--", label="|ψ|^2 PINN")
plt.title("|ψ(x,T)|^2 en pozo infinito (n=1)")
plt.xlabel("x")
plt.ylabel("|ψ|^2")
plt.legend()
plt.show()

# Gráfico 2: error absoluto en |psi|^2
plt.figure()
abs_err = np.abs(mod2_pred - mod2_true)
plt.plot(x_eval.squeeze(1).cpu().numpy(), abs_err, label="Error absoluto")
plt.title("Error absoluto en |ψ(x,T)|^2")
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()
plt.show()
