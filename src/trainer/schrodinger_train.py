import time
import torch

import sys; import os

from torch.optim import Adam
from torch.nn    import MSELoss

# ALR: Lineas adicionales para compatibilidad de path
# Global path
global_path = os.getcwd()

# Composición del path
global_path = global_path.split('/')

# Generación de path global al directorio padre 
relative_path = '/'.join(global_path[:-2])

# Linea adicional para ubicación de path en los scripts
sys.path.append(relative_path)

from data.synthetic.schrodinger_dataset import sample_collocation, exact_eigenstate
from src.nn.pde import schrodinger_operator

# Parametros físicos del dominio
L = 1.0        # dominio espacial [0, L]
T = 0.2        # tiempo final
hbar = 1.0
mass = 1.0
n_level = 2    # nivel del pozo (usaremos n=1)

# Parametros de Muestreo
N_f = 100     # collocation (interior)
N_b = 100      # borde (x=0 y x=L)
N_0 = 100      # inicial (t=0)

LR = 1e-3
PRINT_EVERY = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # mejor precisión para EDP de 2º orden
torch.set_default_dtype(DTYPE)
torch.manual_seed(0);

def train(model):
    opt = model.optimizer if model.optimizer is not None else Adam(model.parameters(), lr=1E-3)
    mse = model.loss_fn if model.loss_fn is not None else MSELoss()

    (t_f, x_f), (t_b, x_b), (t_0, x_0) = sample_collocation(N_f, N_b, N_0, L=L, T=T, device=DEVICE, dtype=DTYPE)
    psi0_r, psi0_i, _ = exact_eigenstate(n_level, t_0, x_0, L=L, mass=mass, hbar=hbar)

    t0 = time.time()
    for epoch in range(1, model.epochs + 1):
        opt.zero_grad()

        # PDE (interior) con V=0 (pozo interior)
        [_, _, rR, rI] = schrodinger_operator(model, t_f, x_f, potential_fn=0, mass=mass, hbar=hbar)
        loss_pde = mse(rR[:, 0:1], torch.zeros_like(rR)) + mse(rI[:, 0:1], torch.zeros_like(rI))

        # BC Dirichlet: ψ=0 en x=0 y x=L
        psi_b = model(torch.cat((t_b, x_b), dim=1))
        loss_bc = mse(psi_b[:, 0:1], torch.zeros_like(psi_b[:, 0:1])) + \
                  mse(psi_b[:, 1:2], torch.zeros_like(psi_b[:, 1:2]))

        # IC: ψ(t=0,x) = sqrt(2/L) sin(pi x/L) (parte imag=0 al inicio)
        psi0 = model(torch.cat((t_0, x_0), dim=1))
        loss_ic = mse(psi0[:, 0:1], psi0_r[:, 0:1]) + mse(psi0[:, 1:2], psi0_i[:, 0:1])

        # Ponderación básica (ajústala si alguna pérdida domina)
        loss = 1.0 * loss_pde + 1.0 * loss_bc + 2.0 * loss_ic

        loss.backward()
        opt.step()

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:5d} | loss={loss.item():.3e} "
                  f"(pde={loss_pde.item():.3e}, bc={loss_bc.item():.3e}, ic={loss_ic.item():.3e}) | {elapsed:.1f}s")
