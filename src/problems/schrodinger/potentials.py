from __future__ import annotations

from typing import Callable, Dict, Any
import torch


def zero_potential(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def harmonic_oscillator_potential(mass: float, omega: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    mass = float(mass)
    omega = float(omega)

    def V(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * mass * (omega ** 2) * (x ** 2)

    return V


def rectangular_barrier_potential(V0: float, x1: float, x2: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    V0 = float(V0)
    x1 = float(x1)
    x2 = float(x2)

    def V(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inside = (x >= x1) & (x <= x2)
        return torch.where(inside, torch.full_like(x, V0), torch.zeros_like(x))

    return V


def build_potential_fn(eq_params: Dict[str, Any]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Devuelve potential_fn(t,x) consistente con eq_params."""
    ex = str(eq_params.get("example", "box")).lower()

    if ex == "ho":
        return harmonic_oscillator_potential(
            mass=eq_params.get("mass", 1.0),
            omega=eq_params.get("omega", 1.0),
        )

    if ex in {"box", "well", "infinite_well"}:
        return zero_potential

    if ex == "barrier":
        L = float(eq_params.get("L", 1.0))
        V0 = float(eq_params.get("V0", 50.0))
        bw = float(eq_params.get("barrier_width", L / 5.0))
        xc = float(eq_params.get("x_center", 0.5 * L))
        x1 = xc - 0.5 * bw
        x2 = xc + 0.5 * bw
        return rectangular_barrier_potential(V0=V0, x1=x1, x2=x2)

    return zero_potential

