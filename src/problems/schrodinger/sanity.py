from __future__ import annotations

from typing import Any, Dict
import torch

from src.problems.schrodinger.problem import build_problem


def sanity_check(eq_params: Dict[str, Any], *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> Dict[str, Any]:
    """
    Chequeo rápido:
      - construye bundle
      - evalúa V(x)
      - evalúa exacta en t=0 y t=T
    """
    dev = torch.device(device)
    pb = build_problem(eq_params, device=dev, dtype=dtype)

    x = torch.linspace(pb["x_min"], pb["x_max"], 200, device=dev, dtype=dtype).reshape(-1, 1)
    t0 = torch.zeros_like(x)
    tT = torch.full_like(x, pb["T"])

    with torch.no_grad():
        V = pb["potential_fn"](t0, x)
        psi0_r, psi0_i, _ = pb["exact"](pb["n_level"], t0, x)
        psiT_r, psiT_i, _ = pb["exact"](pb["n_level"], tT, x)

        return dict(
            V_min=float(V.min().cpu()),
            V_max=float(V.max().cpu()),
            psi0_norm=float(torch.sqrt(torch.mean(psi0_r**2 + psi0_i**2)).cpu()),
            psiT_norm=float(torch.sqrt(torch.mean(psiT_r**2 + psiT_i**2)).cpu()),
        )
