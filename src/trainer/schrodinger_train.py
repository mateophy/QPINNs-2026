import time
import torch
from torch.optim import Adam, LBFGS
from torch.nn import MSELoss

from src.nn.pde import schrodinger_operator
from src.problems.schrodinger.problem import build_problem


# -------------------------
# Helpers
# -------------------------
def _parse_dtype(dtype_like):
    if isinstance(dtype_like, torch.dtype):
        return dtype_like
    if isinstance(dtype_like, str):
        s = dtype_like.lower()
        if "64" in s:
            return torch.float64
        if "32" in s:
            return torch.float32
    return torch.float64


def _to_device_dtype(*tensors, device, dtype):
    out = []
    for t in tensors:
        if t is None:
            out.append(None)
        else:
            out.append(t.to(device=device, dtype=dtype))
    return out


def _augment_barrier_focus_points(eq_params, device, dtype, t_f, x_f, focus_frac=0.45, margin_frac=0.30):
    """
    Reemplaza una fracción de puntos de colocation por puntos concentrados
    alrededor de la barrera: [xc-w/2, xc+w/2] con un margen.
    """
    N = x_f.shape[0]
    n_focus = int(N * focus_frac)
    if n_focus <= 0:
        return t_f, x_f

    x_min = float(eq_params.get("x_min", 0.0))
    x_max = float(eq_params.get("x_max", 1.0))
    L = x_max - x_min

    xc = float(eq_params.get("x_center", 0.5 * (x_min + x_max)))
    w = float(eq_params.get("barrier_width", 0.2 * L))

    a = xc - 0.5 * w
    b = xc + 0.5 * w
    margin = margin_frac * w

    lo = max(x_min, a - margin)
    hi = min(x_max, b + margin)

    x_focus = lo + (hi - lo) * torch.rand(n_focus, 1, device=device, dtype=dtype)

    # tiempos: mezcla uniforme + sesgo a t~0
    t_uniform = torch.rand(n_focus, 1, device=device, dtype=dtype)
    t_bias = torch.rand(n_focus, 1, device=device, dtype=dtype) ** 2
    t_mix = 0.5 * t_uniform + 0.5 * t_bias
    t_focus = t_mix * float(eq_params["T"])

    idx = torch.randperm(N, device=device)[:n_focus]
    t_f = t_f.clone()
    x_f = x_f.clone()
    t_f[idx] = t_focus
    x_f[idx] = x_focus
    return t_f, x_f


def _norm_loss(model, x_grid, times, device, dtype):
    """
    Penaliza promedio_k ( ||psi(t_k)||^2 - 1 )^2  usando integral trapz en x.
    """
    losses = []
    for tval in times:
        t = torch.full_like(x_grid, float(tval), device=device, dtype=dtype)
        psi = model(torch.cat((t, x_grid), dim=1))
        prob = psi[:, 0:1] ** 2 + psi[:, 1:2] ** 2  # |psi|^2
        norm = torch.trapz(prob.squeeze(-1), x_grid.squeeze(-1))
        losses.append((norm - 1.0) ** 2)
    return torch.mean(torch.stack(losses))


def _gauge_fix_loss(model, t_samples, x0, device, dtype):
    """
    Fija gauge de fase con una restricción suave:
      Im{psi(t, x0)} ~ 0  para varios tiempos.
    Esto NO cambia la física (fase global), pero ayuda a que Im no colapse.
    """
    t = torch.as_tensor(t_samples, device=device, dtype=dtype).reshape(-1, 1)
    x = torch.full_like(t, float(x0), device=device, dtype=dtype)
    psi = model(torch.cat((t, x), dim=1))
    im = psi[:, 1:2]
    return torch.mean(im ** 2)


# -------------------------
# Trainer
# -------------------------
def train(model, N_f=7000, N_b=500, N_0=500):
    """
    Trainer Schrödinger 1D (ho / box / barrier).

    Features:
      - warmup/curriculum de PDE (w_pde sube gradualmente)
      - término de normalización (anti-solución trivial)
      - IC en malla fija (opcional)
      - focus de muestreo cerca de la barrera (opcional)
      - chunking para PDE (útil para DV)
      - LBFGS opcional al final

    NUEVO (para mejorar Im):
      - args["w_pde_i"], args["w_bc_i"], args["w_ic_i"] (pesos para la parte imaginaria)
      - gauge fix opcional: args["w_gauge"] > 0
    """
    if not hasattr(model, "args") or "eq_params" not in model.args:
        raise KeyError("Falta model.args['eq_params'].")

    args = model.args
    eq_params = args["eq_params"]

    # device / dtype
    device_like = args.get("device", None)
    device = torch.device(device_like) if device_like is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.get("dtype", torch.float64))

    model.to(device)
    model = model.double() if dtype == torch.float64 else model.float()

    problem = build_problem(eq_params, device=device, dtype=dtype)
    ex = str(eq_params.get("example", "")).lower()

    # optimizer
    lr = float(args.get("lr", 1e-3))
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    mse = MSELoss()

    # pesos base
    w_pde_base = float(args.get("w_pde", 1.0))
    w_bc       = float(args.get("w_bc", 1.0))
    w_ic       = float(args.get("w_ic", 300.0))
    w_norm     = float(args.get("w_norm", 50.0))

    # NUEVO: pesos para Im (default = 1 => comportamiento viejo)
    w_pde_i = float(args.get("w_pde_i", 1.0))
    w_bc_i  = float(args.get("w_bc_i",  1.0))
    w_ic_i  = float(args.get("w_ic_i",  1.0))

    # gauge fix opcional (recomendado para HO eigenstate)
    w_gauge = float(args.get("w_gauge", 0.0))
    gauge_x0 = float(args.get("gauge_x0", 0.5 * (float(problem["x_min"]) + float(problem["x_max"]))))
    gauge_K  = int(args.get("gauge_K", 5))  # #tiempos por época para gauge

    # warmup
    epochs = int(getattr(model, "epochs", args.get("epochs", 3000)))
    warmup_epochs = int(args.get("warmup_epochs", max(200, epochs // 5)))

    # IC en grid fijo
    ic_grid = bool(args.get("ic_grid", True))

    # chunking para PDE (DV)
    pde_chunk_size = int(args.get("pde_chunk_size", 0))  # 0 => sin chunking

    # calcular L2 cada l2_every épocas (para DV, acelera)
    l2_every = int(args.get("l2_every", 1))

    # hard BC
    hard_bc = bool(args.get("hard_bc", False))

    # malla validación para L2(t=T)
    N_val = int(args.get("N_val", 400))
    x_val = torch.linspace(problem["x_min"], problem["x_max"], N_val, device=device, dtype=dtype).reshape(-1, 1)
    t_val = torch.full_like(x_val, problem["T"])
    with torch.no_grad():
        psi_r_exact_T, psi_i_exact_T, _ = problem["exact"](problem["n_level"], t_val, x_val)

    # malla para normalización
    N_norm = int(args.get("N_norm", 400))
    x_norm = torch.linspace(problem["x_min"], problem["x_max"], N_norm, device=device, dtype=dtype).reshape(-1, 1)

    # tiempos para normalización
    norm_times = args.get("norm_times", None)
    if norm_times is None:
        norm_times = [0.0, float(problem["T"])]
    else:
        norm_times = [float(tt) for tt in norm_times]
        # si vienen como fracciones 0..1, conviértelas a tiempo real
        if len(norm_times) > 0 and max(norm_times) <= 1.0:
            norm_times = [tt * float(problem["T"]) for tt in norm_times]

    # IC exacta en grid (si aplica)
    if ic_grid:
        x_0_fixed = torch.linspace(problem["x_min"], problem["x_max"], N_0, device=device, dtype=dtype).reshape(-1, 1)
        t_0_fixed = torch.zeros_like(x_0_fixed)
        with torch.no_grad():
            psi0_r_fixed, psi0_i_fixed, _ = problem["exact"](problem["n_level"], t_0_fixed, x_0_fixed)

    # histories
    if not hasattr(model, "loss_history"):
        model.loss_history = []
    if not hasattr(model, "l2_rel_history"):
        model.l2_rel_history = []

    PRINT_EVERY = int(args.get("print_every", 100))
    t0_clock = time.time()
    last_l2_rel = None

    for epoch in range(1, epochs + 1):
        model.train()

        # curriculum PDE
        if warmup_epochs <= 0:
            pde_factor = 1.0
        else:
            pde_factor = min(1.0, epoch / warmup_epochs)
        w_pde = w_pde_base * pde_factor

        # sample
        (t_f, x_f), (t_b, x_b), (t_0, x_0) = problem["sample"](N_f, N_b, N_0)
        t_f, x_f, t_b, x_b, t_0, x_0 = _to_device_dtype(t_f, x_f, t_b, x_b, t_0, x_0, device=device, dtype=dtype)

        # focus en barrera
        if ex == "barrier":
            focus_frac = float(args.get("focus_frac", 0.45))
            margin_frac = float(args.get("focus_margin_frac", 0.30))
            t_f, x_f = _augment_barrier_focus_points(eq_params, device, dtype, t_f, x_f,
                                                     focus_frac=focus_frac, margin_frac=margin_frac)

        # IC exacta
        if ic_grid:
            t_0 = t_0_fixed
            x_0 = x_0_fixed
            psi0_r = psi0_r_fixed
            psi0_i = psi0_i_fixed
        else:
            psi0_r, psi0_i, _ = problem["exact"](problem["n_level"], t_0, x_0)

        opt.zero_grad()

        # -------------------------
        # PDE residual (con chunking opcional)
        # -------------------------
        if pde_chunk_size and pde_chunk_size > 0:
            loss_chunks = []
            for i0 in range(0, t_f.shape[0], pde_chunk_size):
                ti = t_f[i0:i0 + pde_chunk_size]
                xi = x_f[i0:i0 + pde_chunk_size]
                _, _, rR, rI = schrodinger_operator(
                    model, ti, xi,
                    potential_fn=problem["potential_fn"],
                    mass=problem["mass"],
                    hbar=problem["hbar"],
                )
                loss_i = mse(rR, torch.zeros_like(rR)) + w_pde_i * mse(rI, torch.zeros_like(rI))
                loss_chunks.append(loss_i)
            loss_pde = torch.mean(torch.stack(loss_chunks))
        else:
            _, _, rR, rI = schrodinger_operator(
                model, t_f, x_f,
                potential_fn=problem["potential_fn"],
                mass=problem["mass"],
                hbar=problem["hbar"],
            )
            loss_pde = mse(rR, torch.zeros_like(rR)) + w_pde_i * mse(rI, torch.zeros_like(rI))

        # -------------------------
        # BC
        # -------------------------
        if hard_bc and ex in ("box", "barrier"):
            loss_bc = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            psi_b = model(torch.cat((t_b, x_b), dim=1))
            loss_bc = mse(psi_b[:, 0:1], torch.zeros_like(psi_b[:, 0:1])) + \
                      w_bc_i * mse(psi_b[:, 1:2], torch.zeros_like(psi_b[:, 1:2]))

        # -------------------------
        # IC
        # -------------------------
        psi_0 = model(torch.cat((t_0, x_0), dim=1))
        loss_ic = mse(psi_0[:, 0:1], psi0_r) + w_ic_i * mse(psi_0[:, 1:2], psi0_i)

        # -------------------------
        # Norma
        # -------------------------
        loss_norm = _norm_loss(model, x_norm, norm_times, device, dtype)

        # -------------------------
        # Gauge fix (opcional)
        # -------------------------
        if w_gauge > 0.0:
            # tiempos aleatorios en [0,T]
            t_samp = torch.rand(gauge_K, device=device, dtype=dtype) * float(problem["T"])
            loss_gauge = _gauge_fix_loss(model, t_samp, gauge_x0, device, dtype)
        else:
            loss_gauge = torch.tensor(0.0, device=device, dtype=dtype)

        # total
        loss = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic + w_norm * loss_norm + w_gauge * loss_gauge
        loss.backward()

        # step
        opt.step()

        model.loss_history.append(float(loss.detach().cpu().item()))

        # -------------------------
        # métrica L2 en t=T (cada l2_every)
        # -------------------------
        do_l2 = (l2_every <= 1) or (epoch % l2_every == 0) or (epoch == 1)
        if do_l2:
            with torch.no_grad():
                psi_pred_T = model(torch.cat((t_val, x_val), dim=1))
                err_r = psi_pred_T[:, 0:1] - psi_r_exact_T
                err_i = psi_pred_T[:, 1:2] - psi_i_exact_T
                l2_abs = torch.sqrt(torch.mean(err_r ** 2 + err_i ** 2))
                ref = torch.sqrt(torch.mean(psi_r_exact_T ** 2 + psi_i_exact_T ** 2)) + 1e-12
                last_l2_rel = float((l2_abs / ref).detach().cpu().item())

        # guardamos SIEMPRE un valor (para graficar vs épocas sin huecos)
        model.l2_rel_history.append(float(last_l2_rel) if last_l2_rel is not None else float("nan"))

        # logging
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            elapsed = time.time() - t0_clock
            l2_print = model.l2_rel_history[-1]
            msg = (
                f"It: {epoch}, Loss: {loss.item():.3e}, "
                f"Loss_res: {loss_pde.item():.3e},  Loss_bcs: {loss_bc.item():.3e}, "
                f"Loss_ut_ics: {loss_ic.item():.3e}, Loss_norm: {loss_norm.item():.3e}, "
                f"Loss_gauge: {loss_gauge.item():.3e}, "
                f"w_pde: {w_pde:.2e}, L^2_error: {l2_print:.3e}, lr: {opt.param_groups[0]['lr']:.3e}, "
                f"Time: {elapsed:.2e}"
            )
            if hasattr(model, "logger") and model.logger is not None:
                model.logger.print(msg)
            else:
                print(msg)

            if hasattr(model, "save_state"):
                model.save_state()

    # -------------------------
    # LBFGS opcional (muy útil en barrera)
    # -------------------------
    use_lbfgs = bool(args.get("use_lbfgs", ex == "barrier"))
    if use_lbfgs:
        max_iter = int(args.get("lbfgs_max_iter", 500))
        lr_lb = float(args.get("lbfgs_lr", 1.0))

        # fija un batch para LBFGS
        (t_f, x_f), (t_b, x_b), (t_0, x_0) = problem["sample"](N_f, N_b, N_0)
        t_f, x_f, t_b, x_b, t_0, x_0 = _to_device_dtype(t_f, x_f, t_b, x_b, t_0, x_0, device=device, dtype=dtype)

        if ex == "barrier":
            t_f, x_f = _augment_barrier_focus_points(
                eq_params, device, dtype, t_f, x_f,
                focus_frac=float(args.get("focus_frac", 0.45)),
                margin_frac=float(args.get("focus_margin_frac", 0.30)),
            )

        if ic_grid:
            t_0 = t_0_fixed
            x_0 = x_0_fixed
            psi0_r = psi0_r_fixed
            psi0_i = psi0_i_fixed
        else:
            psi0_r, psi0_i, _ = problem["exact"](problem["n_level"], t_0, x_0)

        lbfgs = LBFGS(model.parameters(), lr=lr_lb, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()

            _, _, rR, rI = schrodinger_operator(
                model, t_f, x_f,
                potential_fn=problem["potential_fn"],
                mass=problem["mass"],
                hbar=problem["hbar"],
            )
            loss_pde_lb = mse(rR, torch.zeros_like(rR)) + w_pde_i * mse(rI, torch.zeros_like(rI))

            if hard_bc and ex in ("box", "barrier"):
                loss_bc_lb = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                psi_b = model(torch.cat((t_b, x_b), dim=1))
                loss_bc_lb = mse(psi_b[:, 0:1], torch.zeros_like(psi_b[:, 0:1])) + \
                             w_bc_i * mse(psi_b[:, 1:2], torch.zeros_like(psi_b[:, 1:2]))

            psi_0 = model(torch.cat((t_0, x_0), dim=1))
            loss_ic_lb = mse(psi_0[:, 0:1], psi0_r) + w_ic_i * mse(psi_0[:, 1:2], psi0_i)

            loss_norm_lb = _norm_loss(model, x_norm, norm_times, device, dtype)

            if w_gauge > 0.0:
                t_samp = torch.rand(gauge_K, device=device, dtype=dtype) * float(problem["T"])
                loss_gauge_lb = _gauge_fix_loss(model, t_samp, gauge_x0, device, dtype)
            else:
                loss_gauge_lb = torch.tensor(0.0, device=device, dtype=dtype)

            loss_total = (
                w_pde_base * loss_pde_lb
                + w_bc * loss_bc_lb
                + w_ic * loss_ic_lb
                + w_norm * loss_norm_lb
                + w_gauge * loss_gauge_lb
            )
            loss_total.backward()
            return loss_total

        if hasattr(model, "logger") and model.logger is not None:
            model.logger.print(f"Starting LBFGS: max_iter={max_iter}")
        else:
            print(f"Starting LBFGS: max_iter={max_iter}")

        lbfgs.step(closure)

        if hasattr(model, "save_state"):
            model.save_state()
