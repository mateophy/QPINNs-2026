from __future__ import annotations

from typing import Any, Dict
import torch

from src.problems.schrodinger.spec import validate_eq_params, normalize_example
from src.problems.schrodinger.potentials import build_potential_fn
from src.problems.schrodinger.sampling import sample_points
from src.problems.schrodinger.exact import build_exact_solution


def build_problem(eq_params: Dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """
    Bundle coherente:
      - potential_fn(t,x)
      - sample(Nf,Nb,N0)
      - exact(n,t,x)
      - dominio y parámetros físicos
    """
    eq = dict(eq_params)  # copia
    eq["example"] = normalize_example(eq.get("example", "box"))
    validate_eq_params(eq)

    x_min = float(eq.get("x_min", 0.0))
    x_max = float(eq.get("x_max", eq.get("L", 1.0)))
    T = float(eq.get("T", 0.2))
    example = str(eq.get("example", "box")).lower()

    potential_fn = build_potential_fn(eq)
    exact = build_exact_solution(eq)

    barrier_params = None
    if example == "barrier":
        barrier_params = dict(
            L=float(eq.get("L", 1.0)),
            barrier_width=float(eq.get("barrier_width", float(eq.get("L", 1.0)) / 5.0)),
            x_center=float(eq.get("x_center", 0.5 * float(eq.get("L", 1.0)))),
            focus_frac=float(eq.get("focus_frac", 0.35)),
            focus_window=float(eq.get("focus_window", 0.05 * (x_max - x_min))),
        )

    def sampler(Nf: int, Nb: int, N0: int):
        return sample_points(
            Nf, Nb, N0,
            x_min=x_min, x_max=x_max, T=T,
            device=device, dtype=dtype,
            example=example,
            barrier_params=barrier_params
        )

    return dict(
        eq_params=eq,
        example=example,
        x_min=x_min,
        x_max=x_max,
        T=T,
        L=float(eq.get("L", 1.0)),
        mass=float(eq.get("mass", 1.0)),
        hbar=float(eq.get("hbar", 1.0)),
        n_level=int(eq.get("n_level", 1)),
        potential_fn=potential_fn,
        sample=sampler,
        exact=exact,
        device=device,
        dtype=dtype,
    )
