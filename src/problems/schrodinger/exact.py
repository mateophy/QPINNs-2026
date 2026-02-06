from __future__ import annotations

from typing import Any, Callable, Dict, Tuple
import torch

from data.synthetic.schrodinger_dataset import exact_eigenstate


def _exact_kwargs_from_eq_params(eq_params: Dict[str, Any]) -> Dict[str, Any]:
    ex = str(eq_params.get("example", "box")).lower()
    kw = dict(
        L=float(eq_params.get("L", 1.0)),
        mass=float(eq_params.get("mass", 1.0)),
        hbar=float(eq_params.get("hbar", 1.0)),
        omega=float(eq_params.get("omega", 1.0)),
        example=ex,
    )
    if ex == "barrier":
        kw.update(
            V0=float(eq_params.get("V0", 50.0)),
            barrier_width=float(eq_params.get("barrier_width", kw["L"] / 5.0)),
            x_center=float(eq_params.get("x_center", 0.5 * kw["L"])),
            num_grid=int(eq_params.get("num_grid", 400)),
        )
    return kw


def build_exact_solution(
    eq_params: Dict[str, Any],
) -> Callable[[int, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    kw = _exact_kwargs_from_eq_params(eq_params)

    def exact(n: int, t: torch.Tensor, x: torch.Tensor):
        return exact_eigenstate(n, t, x, **kw)

    return exact
