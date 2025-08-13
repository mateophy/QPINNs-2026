import numpy as np
import torch


class Sampler:
    def __init__(self, dim, coords, func, device="cpu"):
        self.dim = dim
        self.coords = torch.tensor(
            coords, dtype=torch.cfloat, device=device
        )  # Convert coords to float32 tensor
        self.func = func
        self.device = device

    def sample(self, N):
        rand_vals = torch.rand(N, self.dim, dtype=torch.cfloat, device=self.device)
        x = (
            self.coords[0:1, :]
            + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        )
        y = self.func(x)
        return x, y


def u(x, n):
    """
    :param x: x = (t, x)
    """
    t = x[:, 0:1]
    x = x[:, 1:2]

    # Frecuency definition
    pi    = torch.tensor(np.pi, dtype=torch.cfloat, device=x.device)
    omega = torch.tensor(n * n * pi * pi, dtype=torch.cfloat, device=x.device)
    
    return -(0 + 1.j) / 2 * torch.exp(-(0+1.j)* omega * t) * (
        torch.exp((0 + 1.j) * n * pi * x) - torch.exp(-(0 + 1.j) * n * pi * x))

def u_t(x, n): 
    t = x[:, 0:1]
    x = x[:, 1:2]

    # Frecuency definition
    pi    = torch.tensor(np.pi, dtype=torch.cfloat, device=x.device) 
    omega = torch.tensor(n * n * pi * pi, dtype=torch.cfloat, device=x.device)

    u_t = - omega / 2 * torch.exp(-(0+1.j)* omega * t) * (
        torch.exp((0 + 1.j) * n * pi * x) - torch.exp(-(0 + 1.j) * n * pi * x))

    return u_t


def u_xx(x, n):
    t = x[:, 0:1]
    x = x[:, 1:2]

    # Frecuency definition
    pi    = torch.tensor(np.pi, dtype=torch.cfloat, device=x.device)
    omega = torch.tensor(n * n * pi * pi, dtype=torch.cfloat, device=x.device)

    u_xx = (0 + 1.j) * omega / 2 * torch.exp(-(0+1.j)* omega * t) * (
        torch.exp((0 + 1.j) * n * pi * x) - torch.exp(-(0 + 1.j) * n * pi * x))

    return u_xx


def r(x, n):
    return u_t(x, n) - (0 + 1.j) * u_xx(x, n)


def generate_training_dataset(device):
    n = torch.tensor(1., dtype=torch.cfloat, device=device)

    ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.cfloat)
    bc1_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.cfloat)
    bc2_coords = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.cfloat)
    dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.cfloat)

    ics_sampler = Sampler(2, ics_coords, lambda x: u(x, n), device=device)

    bc1 = Sampler(2, bc1_coords, lambda x: u(x, n), device=device)
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, n), device=device)
    bcs_sampler = [bc1, bc2]

    res_sampler = Sampler(2, dom_coords, lambda x: r(x, n), device=device)
    coll_sampler = Sampler(2, dom_coords, lambda x: u(x, n), device=device)

    return [ics_sampler, bcs_sampler, res_sampler]
