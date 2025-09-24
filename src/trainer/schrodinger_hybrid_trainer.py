import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Set variable for number of threads 
os.environ['OMP_NUM_THREADS']    = '64'
os.environ['QULACS_NUM_THREADS'] = '64'

from src.utils.plot_loss                import smooth_loss
from src.utils.logger                   import Logging
from src.utils.plot_prediction          import plt_prediction
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
hidden_dim = 30
num_quantum_layers = 1
cutoff_dim = 20
classic_network = [input_dim, hidden_dim, output_dim]

# Input parameters: Iterable
L = 5.0        # dominio espacial [0, L]
T = 0.1        # tiempo final
hbar = 1.0
mass = 1.0
n_level = 0    # nivel (usaremos n=1)
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

args = {
    "batch_size": 64,
    "epochs": 500, 
    "lr": 1E-3,
    "seed": 42,
    "print_every": 1,
    "log_path": "./results/models/checkpoints/schrodinger/qho",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    "hidden_dim": hidden_dim,
    "num_quantum_layers": num_quantum_layers,
    "classic_network": classic_network,
    "q_ansatz": "sim_circ_19",      # options: "alternating_layer_tdcnot", "abbas" , farhi , sim_circ_13_half, sim_circ_13 , sim_circ_14_half, sim_circ_14 , sim_circ_15 ,sim_circ_19
    "mode": mode,
    "activation": "null",           # options: "null", "partial_measurement_half" , partial_measurement_x, tanh (Classical)
    "shots": 20,                    # Analytical gradients enabled
    "problem": "schrodinger",
    "solver": "DV",                 # options : "CV", "Classical", "DV"
    "device": DEVICE,
    "method": "None",
    "cutoff_dim": cutoff_dim,  # num_qubits >= cutoff_dim
    "class": "CVNeuralNetwork2",  # options CVNeuralNetwork1, CVNeuralNetwork2, CVNeuralNetwork3
    "encoding": "angle",  # options : "ampiltude" , "angle" for DV , none for others
    "eq_params" : eq_params,
    "noise" : True,
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

if args['solver'] != "Classical":
    model.logger.print(f"Training completed successfuly!, shots made: {model.quantum_layer.shots_done}")

# Loss history plot 
plt.semilogy(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

file_path = os.path.join(model.log_path, "loss_history.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.show()

plt.close(
    "all",
)

# Loss history plot 
plt.semilogy(range(len(model.loss_history)), smooth_loss(model.loss_history))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

file_path = os.path.join(model.log_path, "loss_history_smooth.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.show()

plt.close(
    "all",
)

# settings
number_of_points = 50 

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
    mod2_pred = (psi_r_pred**2 + psi_i_pred**2).squeeze(1).cpu().numpy()

    psi_r_true, psi_i_true, En = exact_eigenstate(n_level, t_eval, x_eval, L=L, mass=mass, hbar=hbar, omega=omega, example=example)
    mod2_true = (psi_r_true**2 + psi_i_true**2).squeeze(1).cpu().numpy()

# - Generation of input argument
X = (
    torch.hstack(
        (torch.from_numpy(t.flatten()[:, None]), torch.from_numpy(x.flatten()[:, None]))
    )
    .to(DEVICE)
    .to(torch.float32)
).cpu().detach().numpy()

# Gráfico 1: |psi|^2 en t=T (PINN vs exacto)
plt.figure()
plt.plot(x_eval.squeeze(1).cpu().numpy(), mod2_true, label="|ψ|^2 exacto")
plt.plot(x_eval.squeeze(1).cpu().numpy(), mod2_pred, "--", label="|ψ|^2 PINN")
plt.title(f"|ψ(x,T)|^2 en pozo infinito (n={n_level})")
plt.xlabel("x"); plt.ylabel("|ψ|^2")
plt.legend(); plt.grid()

# Saving of solution comparison plot 
file_path = os.path.join(model.log_path, "solution_plot.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.show(); plt.close("all")

# Gráfico 2: error absoluto en |psi|^2
plt.figure()
abs_err = np.abs(mod2_pred - mod2_true)
plt.plot(x_eval.squeeze(1).cpu().numpy(), abs_err, label="Error absoluto")
plt.title("Error absoluto en |ψ(x,T)|^2")
plt.xlabel("x"); plt.ylabel("Error")
plt.legend(); plt.grid()

# Saving of solution comparison plot 
file_path = os.path.join(model.log_path, "error_plot.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.show(); plt.close("all")

# Routine plots:
plt_prediction(
    logger,
    X,
    psi_r_true.cpu().detach().numpy(),
    psi_r_pred.cpu().detach().numpy(),
    psi_i_true.cpu().detach().numpy(),
    psi_i_pred.cpu().detach().numpy(),
)
