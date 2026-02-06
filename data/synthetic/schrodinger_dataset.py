import os
import sys
import torch
import numpy as np

from math import factorial

# Cache global para eigenestados numéricos de barrera (evita diagonalizar cada epoch)
_BARRIER_EIGEN_CACHE = {}


# Sample collocation of points
def sample_collocation(Nf, Nb, N0, L=1.0, T=0.2, device="cpu", dtype=torch.float64, example='box'):

    # Harmonic portential points
    if example.lower() == 'ho':
        # Interior (PDE)
        t_f = torch.rand(Nf, 1, device=device, dtype=dtype) * T
        x_f = (torch.rand(Nf, 1, device=device, dtype=dtype) - 0.5) * 2 * L
        # Bordes (x=-L y x=+L)
        t_b = torch.rand(Nb, 1, device=device, dtype=dtype) * T
        xb0 = torch.full((Nb // 2, 1), -L, device=device, dtype=dtype)
        xbL = torch.full((Nb - Nb // 2, 1),  L, device=device, dtype=dtype)
        x_b = torch.cat([xb0, xbL], dim=0)
        t_b = torch.cat([t_b[:Nb // 2], t_b[Nb // 2:]], dim=0)
        # Inicial (t=0)
        t_0 = torch.zeros(N0, 1, device=device, dtype=dtype)
        x_0 = (torch.rand(N0, 1, device=device, dtype=dtype) - 0.5) * 2 * L

    # Box points
    elif example.lower() in ['box', 'well', 'infinite_well', 'barrier']:
        t_f = torch.rand(Nf, 1, device=device, dtype=dtype) * T
        x_f = torch.rand(Nf, 1, device=device, dtype=dtype) * L

        t_b = torch.rand(Nb, 1, device=device, dtype=dtype) * T
        xb0 = torch.zeros(Nb // 2, 1, device=device, dtype=dtype)
        xbL = torch.full((Nb - Nb // 2, 1), L, device=device, dtype=dtype)
        x_b = torch.cat([xb0, xbL], dim=0)
        t_b = torch.cat([t_b[:Nb // 2], t_b[Nb // 2:]], dim=0)

        t_0 = torch.zeros(N0, 1, device=device, dtype=dtype)
        x_0 = torch.rand(N0, 1, device=device, dtype=dtype) * L

    else:
        # default: box
        t_f = torch.rand(Nf, 1, device=device, dtype=dtype) * T
        x_f = torch.rand(Nf, 1, device=device, dtype=dtype) * L
        t_b = torch.rand(Nb, 1, device=device, dtype=dtype) * T
        xb0 = torch.zeros(Nb // 2, 1, device=device, dtype=dtype)
        xbL = torch.full((Nb - Nb // 2, 1), L, device=device, dtype=dtype)
        x_b = torch.cat([xb0, xbL], dim=0)
        t_b = torch.cat([t_b[:Nb // 2], t_b[Nb // 2:]], dim=0)
        t_0 = torch.zeros(N0, 1, device=device, dtype=dtype)
        x_0 = torch.rand(N0, 1, device=device, dtype=dtype) * L

    return (t_f, x_f), (t_b, x_b), (t_0, x_0)


# ---- HO helpers ----
def _hermite_phys(n, z):
    """Hermite (física) H_n(z) por recurrencia."""
    if n == 0:
        return torch.ones_like(z)
    if n == 1:
        return 2.0 * z

    Hnm2 = torch.ones_like(z)
    Hnm1 = 2.0 * z
    for k in range(1, n):
        Hn = 2.0 * z * Hnm1 - 2.0 * k * Hnm2
        Hnm2, Hnm1 = Hnm1, Hn
    return Hnm1


def _numeric_barrier_eigenstate_1d(
    n,
    L=1.0,
    mass=1.0,
    hbar=1.0,
    V0=50.0,
    barrier_width=None,
    x_center=None,
    num_grid=400,
    device="cpu",
    dtype=torch.float64,
):
    """
    Eigenestado numérico para un pozo infinito en [0,L] con barrera rectangular.
    Resuelve H φ = E φ con diferencias finitas (Dirichlet en bordes).
    """
    if barrier_width is None:
        barrier_width = L / 5.0
    if x_center is None:
        x_center = 0.5 * L

    x_grid = torch.linspace(0.0, L, num_grid, device=device, dtype=dtype)
    dx = x_grid[1] - x_grid[0]

    x_int = x_grid[1:-1]
    N = x_int.shape[0]

    x1 = x_center - 0.5 * barrier_width
    x2 = x_center + 0.5 * barrier_width
    inside = (x_int >= x1) & (x_int <= x2)
    V_int = torch.where(inside, V0 * torch.ones_like(x_int), torch.zeros_like(x_int))

    diag = 2.0 * torch.ones(N, device=device, dtype=dtype)
    off = -1.0 * torch.ones(N - 1, device=device, dtype=dtype)
    lap = torch.diag(diag) + torch.diag(off, 1) + torch.diag(off, -1)

    coef = (hbar**2) / (2.0 * mass * dx**2)
    H = coef * lap + torch.diag(V_int)

    H_np = H.detach().cpu().numpy()
    E_vals, U = np.linalg.eigh(H_np)
    idx = np.argsort(E_vals)
    E_vals = E_vals[idx]
    U = U[:, idx]

    n_idx = max(0, min(int(n), len(E_vals) - 1))
    E_n = E_vals[n_idx]
    phi_int_np = U[:, n_idx]

    phi = torch.zeros(num_grid, device=device, dtype=dtype)
    phi[1:-1] = torch.from_numpy(phi_int_np).to(device=device, dtype=dtype)

    prob = phi**2
    norm = torch.sqrt(torch.trapz(prob, x_grid))
    phi = phi / (norm + 1e-12)

    return x_grid, phi, torch.tensor(E_n, device=device, dtype=dtype)


def _numeric_barrier_eigenstate_1d_cached(
    n,
    L=1.0,
    mass=1.0,
    hbar=1.0,
    V0=50.0,
    barrier_width=None,
    x_center=None,
    num_grid=400,
    device="cpu",
    dtype=torch.float64,
):
    """
    Wrapper con cache para _numeric_barrier_eigenstate_1d.
    Clave incluye parámetros físicos + (device,dtype) para evitar mezclas.
    """
    if barrier_width is None:
        barrier_width = L / 5.0
    if x_center is None:
        x_center = 0.5 * L

    key = (
        int(n),
        float(L), float(mass), float(hbar),
        float(V0), float(barrier_width), float(x_center),
        int(num_grid),
        str(torch.device(device)),
        str(dtype),
    )

    if key in _BARRIER_EIGEN_CACHE:
        return _BARRIER_EIGEN_CACHE[key]

    out = _numeric_barrier_eigenstate_1d(
        n=n, L=L, mass=mass, hbar=hbar, V0=V0,
        barrier_width=barrier_width, x_center=x_center,
        num_grid=num_grid, device=device, dtype=dtype
    )
    _BARRIER_EIGEN_CACHE[key] = out
    return out


def exact_eigenstate(
    n,
    t,
    x,
    L=1.0,
    mass=1.0,
    hbar=1.0,
    omega=1.0,
    example="box",
    **kwargs,
):
    """
    Devuelve (psi_r, psi_i, E_n) para:
      - example="ho"      → oscilador armónico
      - example in {"box","well","infinite_well"} → pozo infinito
      - example="barrier" → pozo con barrera rectangular (solución numérica)
    """
    if example.lower() == "ho":
        device = x.device
        dtype = x.dtype

        n_int = int(n)
        alpha = mass * omega / hbar
        z = torch.sqrt(torch.tensor(alpha, device=device, dtype=dtype)) * x

        Hn = _hermite_phys(n_int, z)
        norm = 1.0 / torch.sqrt(
            (2.0**n_int) * torch.tensor(float(factorial(n_int)), device=device, dtype=dtype)
        )
        pref = (alpha / torch.pi) ** 0.25
        phi = pref * norm * torch.exp(-0.5 * z**2) * Hn

        E_n = hbar * omega * (n_int + 0.5)
        phase = -(E_n / hbar) * t

        psi_r = phi * torch.cos(phase)
        psi_i = phi * torch.sin(phase)

    elif example.lower() == "barrier":
        V0 = kwargs.get("V0", 50.0)
        barrier_width = kwargs.get("barrier_width", L / 5.0)
        x_center = kwargs.get("x_center", 0.5 * L)
        num_grid = kwargs.get("num_grid", 400)

        device = x.device
        dtype = x.dtype

        x_grid, phi_grid, E_n = _numeric_barrier_eigenstate_1d_cached(
            n=n,
            L=L,
            mass=mass,
            hbar=hbar,
            V0=V0,
            barrier_width=barrier_width,
            x_center=x_center,
            num_grid=num_grid,
            device=device,
            dtype=dtype,
        )

        x_flat = x.reshape(-1)
        xg_np = x_grid.detach().cpu().numpy()
        phi_np = phi_grid.detach().cpu().numpy()
        x_np = x_flat.detach().cpu().numpy()

        phi_x_np = np.interp(x_np, xg_np, phi_np)
        phi_x = torch.from_numpy(phi_x_np).to(device=device, dtype=dtype).reshape_as(x)

        phase = -(E_n / hbar) * t
        psi_r = phi_x * torch.cos(phase)
        psi_i = phi_x * torch.sin(phase)

    else:
        pi = torch.tensor(torch.pi, device=t.device, dtype=t.dtype)
        n_int = int(n)
        k = n_int * pi / L
        E_n = (n_int**2) * (pi**2) * (hbar**2) / (2.0 * mass * (L**2))
        phase = -(E_n / hbar) * t
        spatial = torch.sqrt(2.0 / torch.tensor(L, device=t.device, dtype=t.dtype)) * torch.sin(k * x)

        psi_r = spatial * torch.cos(phase)
        psi_i = spatial * torch.sin(phase)

    return psi_r, psi_i, E_n

