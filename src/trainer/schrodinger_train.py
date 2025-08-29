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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # mejor precisión para EDP de 2º orden

def train(model, N_f = 7, N_b = 5, N_0 = 5):

    # Parametros de decisión 
    example = model.args['eq_params']['example']    # Ejemplo a replicar

    # Parametros físicos del dominio
    L       = model.args['eq_params']['L']          # dominio espacial [0, L]
    T       = model.args['eq_params']['T']          # tiempo final
    hbar    = model.args['eq_params']['hbar'] 
    mass    = model.args['eq_params']['mass']
    n_level = model.args['eq_params']['n_level']    # nivel del pozo (usaremos n=1)

    # Definición por defecto de parametros
    potential_fn = 0; omega = 0

    # Cargado según ejemplos de uso
    if example.lower() == 'ho':                                        # Oscilador armonico

        potential_fn = eval(model.args['eq_params']['potential_fn'])   # Función de potencial
        omega        = model.args['eq_params']['omega']                # Frecuencia natural


    # Constants parameters
    LR = model.args['lr']; PRINT_EVERY = model.args['print_every']

    # Parameters definition by model
    opt = model.optimizer if model.optimizer is not None else Adam(model.parameters(), lr=LR)
    mse = model.loss_fn if model.loss_fn is not None else MSELoss() 
    
    DEVICE = model.args['device']; DTYPE = torch.float32 

    # Dtype definition 
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(42);

    t0 = time.time()
 
    for epoch in range(1, model.epochs + 1): 

        # Points definition
        (t_f, x_f), (t_b, x_b), (t_0, x_0) = sample_collocation(N_f, N_b, N_0, L=L, T=T, device=DEVICE, dtype=DTYPE, example=example)
        psi0_r, psi0_i, _ = exact_eigenstate(n_level, t_0, x_0, L=L, mass=mass, hbar=hbar, omega=omega, example=example)

        # Preparation per epoch 
        opt.zero_grad()

        # PDE (interior) con V=0 (pozo interior)
        [_, _, rR, rI] = schrodinger_operator(model, t_f, x_f, potential_fn= potential_fn, mass=mass, hbar=hbar)
        loss_pde = mse(rR[:, 0:1], torch.zeros_like(rR)) + mse(rI[:, 0:1], torch.zeros_like(rI))

        # BC Dirichlet: ψ=0 en x=0 y x=L
        psi_b = model(torch.cat((t_b, x_b), dim=1))
        loss_bc = mse(psi_b[:, 0:1], torch.zeros_like(psi_b[:, 0:1])) + \
                  mse(psi_b[:, 1:2], torch.zeros_like(psi_b[:, 1:2]))

        # IC: ψ(t=0,x) = sqrt(2/L) sin(pi x/L) (parte imag=0 al inicio)
        psi0 = model(torch.cat((t_0, x_0), dim=1))
        loss_ic = mse(psi0[:, 0:1], psi0_r[:, 0:1]) + mse(psi0[:, 1:2], psi0_i[:, 0:1])

        # Ponderación básica 
        loss = 2.0 * loss_pde + 1.0 * loss_bc + 2.0 * loss_ic

        loss.backward()
        opt.step()

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            elapsed = time.time() - t0 
            model.logger.print(
                    "It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e, lr: %.3e, Time: %.2e"
                    % (
                        epoch,
                        loss.item(),
                        loss_pde.item(),
                        loss_bc.item(),
                        loss_ic.item(),
                        model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0,
                        elapsed,
                )
            )

            # Compute and Print adaptive weights during training
            # Compute the adaptive constant
            model.save_state()

        # Save of loss for each epoch
        model.loss_history.append(loss.item())

