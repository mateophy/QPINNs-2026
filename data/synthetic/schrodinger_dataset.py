import os
import sys
import torch

import numpy as np

# Sample collocation of points
def sample_collocation(Nf, Nb, N0, L=1.0, T=0.2, device="cpu", dtype=torch.float64):
    # Interior (PDE)
    t_f = torch.rand(Nf, 1, device=device, dtype=dtype) * T
    x_f = torch.rand(Nf, 1, device=device, dtype=dtype) * L
    # Bordes (x=0 y x=L)
    t_b = torch.rand(Nb, 1, device=device, dtype=dtype) * T
    xb0 = torch.zeros(Nb//2, 1, device=device, dtype=dtype)
    xbL = torch.full((Nb - Nb//2, 1), L, device=device, dtype=dtype)
    x_b = torch.cat([xb0, xbL], dim=0)
    t_b = torch.cat([t_b[:Nb//2], t_b[Nb//2:]], dim=0)
    # Inicial (t=0)
    t_0 = torch.zeros(N0, 1, device=device, dtype=dtype)
    x_0 = torch.rand(N0, 1, device=device, dtype=dtype) * L
    return (t_f, x_f), (t_b, x_b), (t_0, x_0)

# Definition of exact eigen state of analysis
def exact_eigenstate(n, t, x, L=1.0, mass=1.0, hbar=1.0):
    pi = torch.tensor(np.pi, device=t.device, dtype=t.dtype)
    k = n * pi / L
    En = (n**2) * (pi**2) * (hbar**2) / (2.0 * mass * (L**2))
    phase = - (En / hbar) * t
    spatial = torch.sqrt(2.0 / torch.tensor(L, device=t.device, dtype=t.dtype)) * torch.sin(k * x)
    psi_r = spatial * torch.cos(phase)
    psi_i = spatial * torch.sin(phase)
    return psi_r, psi_i, En