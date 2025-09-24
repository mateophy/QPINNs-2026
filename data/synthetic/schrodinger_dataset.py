import os
import sys
import torch

from math import factorial

# Sample collocation of points
def sample_collocation(Nf, Nb, N0, L=1.0, T=0.2, device="cpu", dtype=torch.float64, example= 'box'):
    
    # Harmonic portential points
    if example.lower() == 'ho':
        # Interior (PDE)
        t_f = torch.rand(Nf,1, device=device, dtype=dtype)*T
        x_f = (torch.rand(Nf,1, device=device, dtype=dtype)-0.5)*2*L
        # Bordes (x=0 y x=L)
        t_b = torch.rand(Nb, 1, device=device, dtype=dtype) * T
        xb0 = torch.full(     (Nb//2, 1), -L, device=device, dtype=dtype)
        xbL = torch.full((Nb - Nb//2, 1),  L, device=device, dtype=dtype)
        x_b = torch.cat([xb0, xbL], dim=0)
        t_b = torch.cat([t_b[:Nb//2], t_b[Nb//2:]], dim=0)
        # Inicial (t=0)
        t_0 = torch.zeros(N0, 1, device=device, dtype=dtype)
        x_0 = (torch.rand(N0,1, device=device)-0.5)*2*L

    # Box potential points
    else:
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

# Hermite polynomials definition
def hermite_physicists(n, z):
    if n == 0: return torch.ones_like(z)
    if n == 1: return 2.0*z
    Hnm2 = torch.ones_like(z); Hnm1 = 2.0*z
    for k in range(1, n):
        Hn = 2.0*z*Hnm1 - 2.0*k*Hnm2
        Hnm2, Hnm1 = Hnm1, Hn
    return Hnm1

# Definition of exact eigen state of analysis
def exact_eigenstate(n, t, x, L=1.0, mass=1.0, hbar=1.0, omega= 1.0, example= 'box'):

    # Harmonic Oscillator solution
    if example.lower() == 'ho':

        xi   = (mass*omega/hbar)**0.5 * x
        fact = factorial(n)
        norm = (mass*omega/(torch.pi*hbar))**0.25 * (1.0/(2.0**n)*fact)**0.5
        phi_x = norm * hermite_physicists(n, xi) * torch.exp(-0.5*(xi**2))
        E_n   = hbar*omega*(n + 0.5)
        phase = -(E_n/hbar) * t
        psi_r = phi_x * torch.cos(phase)
        psi_i = phi_x * torch.sin(phase)

    # Box potential solution
    else:
        pi = torch.tensor(torch.pi, device=t.device, dtype=t.dtype)
        k = n * pi / L
        E_n = (n**2) * (pi**2) * (hbar**2) / (2.0 * mass * (L**2))
        phase = - (E_n / hbar) * t
        spatial = torch.sqrt(2.0 / torch.tensor(L, device=t.device, dtype=t.dtype)) * torch.sin(k * x)
        psi_r = spatial * torch.cos(phase)
        psi_i = spatial * torch.sin(phase)

    return psi_r, psi_i, E_n