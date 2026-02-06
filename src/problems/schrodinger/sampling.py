from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch


def sample_points(
    Nf: int,
    Nb: int,
    N0: int,
    *,
    x_min: float,
    x_max: float,
    T: float,
    device: torch.device,
    dtype: torch.dtype,
    example: str = "box",
    barrier_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Muestrea:
      - Interior (t_f, x_f)
      - Bordes (t_b, x_b) en x=x_min y x=x_max
      - Inicial (t_0, x_0) en t=0

    Para 'barrier' permite oversampling cerca de bordes de la barrera.
    """
    example = str(example).lower()
    x_min = float(x_min)
    x_max = float(x_max)
    T = float(T)

    t_f = torch.rand(Nf, 1, device=device, dtype=dtype) * T

    if example == "barrier" and barrier_params is not None:
        frac_focus = float(barrier_params.get("focus_frac", 0.35))
        window = float(barrier_params.get("focus_window", 0.05 * (x_max - x_min)))

        L = float(barrier_params.get("L", x_max - x_min))
        bw = float(barrier_params.get("barrier_width", L / 5.0))
        xc = float(barrier_params.get("x_center", 0.5 * L))
        x1 = xc - 0.5 * bw
        x2 = xc + 0.5 * bw

        N_focus = int(round(Nf * frac_focus))
        N_uni = Nf - N_focus

        x_uni = x_min + torch.rand(N_uni, 1, device=device, dtype=dtype) * (x_max - x_min)

        N1 = N_focus // 2
        N2 = N_focus - N1

        def sample_window(center: float, N: int) -> torch.Tensor:
            a = max(x_min, center - window)
            b = min(x_max, center + window)
            return a + torch.rand(N, 1, device=device, dtype=dtype) * (b - a)

        x_focus = torch.cat([sample_window(x1, N1), sample_window(x2, N2)], dim=0)
        x_f = torch.cat([x_uni, x_focus], dim=0)

        perm = torch.randperm(x_f.shape[0], device=device)
        x_f = x_f[perm]
        t_f = t_f[perm]
    else:
        x_f = x_min + torch.rand(Nf, 1, device=device, dtype=dtype) * (x_max - x_min)

    t_b = torch.rand(Nb, 1, device=device, dtype=dtype) * T
    Nb0 = Nb // 2
    Nb1 = Nb - Nb0
    xb0 = torch.full((Nb0, 1), x_min, device=device, dtype=dtype)
    xb1 = torch.full((Nb1, 1), x_max, device=device, dtype=dtype)
    x_b = torch.cat([xb0, xb1], dim=0)

    t_0 = torch.zeros(N0, 1, device=device, dtype=dtype)
    x_0 = x_min + torch.rand(N0, 1, device=device, dtype=dtype) * (x_max - x_min)

    return (t_f, x_f), (t_b, x_b), (t_0, x_0)
