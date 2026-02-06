from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_example(potential_type: str) -> str:
    """Normaliza strings del notebook a {'ho','box','barrier'}."""
    if potential_type is None:
        raise ValueError("potential_type no puede ser None")
    s = str(potential_type).strip().lower()
    if s in {"ho", "harmonic", "harmonic_oscillator", "qho", "oscillator"}:
        return "ho"
    if s in {"box", "well", "infinite_well", "pozo", "pozo_infinito"}:
        return "box"
    if s in {"barrier", "barrera", "rect_barrier", "rectangular_barrier"}:
        return "barrier"
    return s


def build_eq_params(
    potential_type: str,
    *,
    L: float = 1.0,
    T: float = 0.2,
    mass: float = 1.0,
    hbar: float = 1.0,
    n_level: int = 1,
    omega: float = 1.0,
    V0: float = 50.0,
    barrier_width: Optional[float] = None,
    x_center: Optional[float] = None,
    num_grid: int = 400,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Construye eq_params coherente y autocontenido.

    Convención de dominio:
      - ho      : x in [-L, L]
      - box     : x in [0, L]
      - barrier : x in [0, L]
    """
    example = normalize_example(potential_type)

    L = float(L)
    T = float(T)
    mass = float(mass)
    hbar = float(hbar)
    n_level = int(n_level)

    if barrier_width is None:
        barrier_width = L / 5.0
    if x_center is None:
        x_center = 0.5 * L

    if x_min is None or x_max is None:
        if example == "ho":
            x_min = -L if x_min is None else float(x_min)
            x_max = +L if x_max is None else float(x_max)
        else:
            x_min = 0.0 if x_min is None else float(x_min)
            x_max = L if x_max is None else float(x_max)

    eq_params: Dict[str, Any] = dict(
        example=example,
        L=L,
        T=T,
        mass=mass,
        hbar=hbar,
        n_level=n_level,
        x_min=float(x_min),
        x_max=float(x_max),
        omega=float(omega),
        V0=float(V0),
        barrier_width=float(barrier_width),
        x_center=float(x_center),
        num_grid=int(num_grid),
    )
    eq_params.update(extra)
    validate_eq_params(eq_params)
    return eq_params


def validate_eq_params(eq_params: Dict[str, Any]) -> None:
    """Chequeos rápidos para evitar incoherencias típicas (sobre todo barrera)."""
    ex = str(eq_params.get("example", "")).lower()
    L = float(eq_params.get("L", 1.0))
    x_min = float(eq_params.get("x_min", 0.0))
    x_max = float(eq_params.get("x_max", L))

    if not (x_max > x_min):
        raise ValueError(f"Dominio inválido: x_min={x_min} x_max={x_max}")

    if ex == "ho":
        if x_min >= 0 or x_max <= 0:
            raise ValueError("Para HO se espera un dominio centrado en 0, e.g. [-L, L].")

    if ex == "barrier":
        bw = float(eq_params.get("barrier_width", L / 5.0))
        xc = float(eq_params.get("x_center", 0.5 * L))
        if not (0.0 <= xc <= L):
            raise ValueError(f"x_center debe estar en [0, L]. Recibido x_center={xc}, L={L}")
        if not (0.0 < bw < L):
            raise ValueError(f"barrier_width debe estar en (0, L). Recibido bw={bw}, L={L}")
        x1 = xc - 0.5 * bw
        x2 = xc + 0.5 * bw
        if x1 < 0.0 or x2 > L:
            raise ValueError(
                f"La barrera se sale del dominio: [x1,x2]=[{x1},{x2}] fuera de [0,{L}]. "
                "Ajusta barrier_width o x_center."
            )
