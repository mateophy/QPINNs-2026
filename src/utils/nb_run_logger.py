# src/utils/nb_run_logger.py
from __future__ import annotations

from pathlib import Path
import os, time, json, sys, platform, subprocess, re, logging
import numpy as np

def _json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def _try_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    except Exception as e:
        return f"<unavailable: {e}>"

def _to_1d_float_array(x):
    """Convierte listas/tensores/arrays a vector float64; si no se puede, devuelve None."""
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        if x.dtype == object:
            # intenta aplanar elementos numéricos
            vals = []
            for v in x.flatten().tolist():
                try:
                    vals.append(float(v))
                except Exception:
                    return None
            return np.asarray(vals, dtype=np.float64)
        return np.asarray(x, dtype=np.float64).reshape(-1)

    if isinstance(x, list):
        # lista de dicts -> intenta extraer "loss" o "total"
        if len(x) > 0 and isinstance(x[0], dict):
            key = None
            for cand in ["loss", "Loss", "total", "loss_total"]:
                if cand in x[0]:
                    key = cand
                    break
            if key is None:
                return None
            vals = []
            for d in x:
                vals.append(float(d.get(key)))
            return np.asarray(vals, dtype=np.float64)

        # lista normal -> float
        vals = []
        for v in x:
            try:
                vals.append(float(v))
            except Exception:
                return None
        return np.asarray(vals, dtype=np.float64)

    # número escalar
    try:
        return np.asarray([float(x)], dtype=np.float64)
    except Exception:
        return None


class NBRunLogger:
    """
    Logger de corridas para notebooks (VS Code/Jupyter):
    - RUN_DIR organizado por potential/solver/seed
    - artifacts/: JSON/NPY/LOG
    - figures/: auto-guardado robusto (aunque no llames plt.show())
    - parsea logs del entrenamiento tipo: "It: ..., Loss: ..., L^2_error: ..."
    """

    def __init__(self, potential: str, solver: str, seed: int = 0, tag: str = "",
                 root: str | Path = "./results"):
        self.potential = str(potential)
        self.solver = str(solver)
        self.seed = int(seed)
        self.tag = str(tag)

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{ts}_seed{self.seed}" + (f"_{self.tag}" if self.tag else "")

        self.root = Path(root)
        self.run_dir = self.root / self.potential / self.solver / run_name
        self.fig_dir = self.run_dir / "figures"
        self.art_dir = self.run_dir / "artifacts"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.art_dir.mkdir(parents=True, exist_ok=True)

        self._fig_seen = set()
        self._hooked = False
        self._log_attached = False

        self.save_env_info()

    # ---------- IO ----------
    def save_env_info(self):
        info = {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _try_cmd(["git", "rev-parse", "HEAD"]),
            "git_status": _try_cmd(["git", "status", "--porcelain"]),
        }
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            info["torch_version"] = None
        try:
            import pennylane as qml
            info["pennylane_version"] = qml.__version__
        except Exception:
            info["pennylane_version"] = None

        self.save_json("env_info.json", info)

    def save_json(self, name: str, data):
        p = self.art_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)
        return p

    def save_npy(self, name: str, arr):
        p = self.art_dir / name
        np.save(p, np.asarray(arr))
        return p

    def save_text(self, name: str, text: str):
        p = self.art_dir / name
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    # ---------- Config / params ----------
    def log_config(self, eq_params=None, args=None, extra: dict | None = None):
        if eq_params is not None:
            self.save_json("eq_params.json", eq_params)
        if args is not None:
            self.save_json("model_args.json", args)
        if extra:
            self.save_json("extra.json", extra)

    def log_param_count(self, model):
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        quantum = sum(
            p.numel() for n, p in model.named_parameters()
            if ("quantum_layer" in n) and p.requires_grad
        )
        classical = total - quantum
        out = {"total": int(total), "quantum": int(quantum), "classical": int(classical)}
        self.save_json("param_count.json", out)
        return out

    # ---------- Histories ----------
    def log_histories(self, model):
        loss_raw = getattr(model, "loss_history", None)
        l2_raw   = getattr(model, "l2_rel_history", None)

        loss = _to_1d_float_array(loss_raw)
        l2   = _to_1d_float_array(l2_raw)

        # guarda también metainfo por si están "muestreados"
        meta = {
            "loss_history_len": None if loss is None else int(loss.size),
            "l2_rel_history_len": None if l2 is None else int(l2.size),
            "loss_history_type": None if loss_raw is None else str(type(loss_raw)),
            "l2_rel_history_type": None if l2_raw is None else str(type(l2_raw)),
        }
        self.save_json("history_meta.json", meta)

        if loss is not None:
            self.save_npy("loss_history.npy", loss)
        if l2 is not None:
            self.save_npy("l2_rel_history.npy", l2)

        return meta

    # ---------- Logging a archivo y parseo ----------
    def attach_python_logging(self, level=logging.INFO, also_console=True):
        """Captura logging a artifacts/python.log y opcionalmente lo deja visible en consola."""
        if self._log_attached:
            return

        log_path = self.art_dir / "python.log"
        root = logging.getLogger()
        root.setLevel(level)

        # 1) Archivo
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
        fh.setFormatter(fmt)
        root.addHandler(fh)

        # 2) Consola (notebook)
        if also_console:
            import sys
            has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
            if not has_stream:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(level)
                sh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
                root.addHandler(sh)

        # 3) Propagación del logger del proyecto
        logging.getLogger("src.utils.logger").setLevel(level)
        logging.getLogger("src.utils.logger").propagate = True

        self._log_attached = True
        self.save_json("logging_capture.json", {"python_log": str(log_path), "also_console": also_console})


    def parse_schrodinger_train_log(self, log_name="python.log"):
        """
        Parsea líneas tipo:
        It: 600, Loss: 3.277e-01, Loss_res: ..., Loss_bcs: ..., Loss_ut_ics: ..., Loss_norm: ...,
        ... L^2_error: 2.037e-02, lr: ...
        """
        log_path = self.art_dir / log_name
        if not log_path.exists():
            raise FileNotFoundError(f"No existe {log_path}. ¿Llamaste attach_python_logging() antes de entrenar?")

        pat = re.compile(
            r"It:\s*(?P<it>\d+),\s*Loss:\s*(?P<loss>[0-9eE\+\-\.]+).*?"
            r"Loss_res:\s*(?P<loss_res>[0-9eE\+\-\.]+).*?"
            r"Loss_bcs:\s*(?P<loss_bcs>[0-9eE\+\-\.]+).*?"
            r"Loss_ut_ics:\s*(?P<loss_ic>[0-9eE\+\-\.]+).*?"
            r"Loss_norm:\s*(?P<loss_norm>[0-9eE\+\-\.]+).*?"
            r"L\^2_error:\s*(?P<l2>[0-9eE\+\-\.]+).*?"
            r"lr:\s*(?P<lr>[0-9eE\+\-\.]+)"
        )

        it, loss, loss_res, loss_bcs, loss_ic, loss_norm, l2, lr = ([] for _ in range(8))

        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = pat.search(line)
            if not m:
                continue
            it.append(int(m.group("it")))
            loss.append(float(m.group("loss")))
            loss_res.append(float(m.group("loss_res")))
            loss_bcs.append(float(m.group("loss_bcs")))
            loss_ic.append(float(m.group("loss_ic")))
            loss_norm.append(float(m.group("loss_norm")))
            l2.append(float(m.group("l2")))
            lr.append(float(m.group("lr")))

        parsed = {
            "it": np.asarray(it, dtype=int),
            "loss": np.asarray(loss, dtype=np.float64),
            "loss_res": np.asarray(loss_res, dtype=np.float64),
            "loss_bcs": np.asarray(loss_bcs, dtype=np.float64),
            "loss_ic": np.asarray(loss_ic, dtype=np.float64),
            "loss_norm": np.asarray(loss_norm, dtype=np.float64),
            "l2": np.asarray(l2, dtype=np.float64),
            "lr": np.asarray(lr, dtype=np.float64),
        }

        # guarda NPY + CSV simple
        for k,v in parsed.items():
            self.save_npy(f"trainlog_{k}.npy", v)

        # CSV
        csv_path = self.art_dir / "trainlog_parsed.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("it,loss,loss_res,loss_bcs,loss_ic,loss_norm,l2,lr\n")
            for i in range(len(parsed["it"])):
                f.write(f"{parsed['it'][i]},{parsed['loss'][i]},{parsed['loss_res'][i]},"
                        f"{parsed['loss_bcs'][i]},{parsed['loss_ic'][i]},{parsed['loss_norm'][i]},"
                        f"{parsed['l2'][i]},{parsed['lr'][i]}\n")

        self.save_json("trainlog_meta.json", {"n_points": int(len(parsed["it"])), "csv": str(csv_path)})
        return parsed

    # ---------- FIGURE AUTOSAVE robusto ----------
    def _save_all_open_figures(self, prefix="fig"):
        import matplotlib.pyplot as plt
        for num in plt.get_fignums():
            fig = plt.figure(num)
            key = (id(fig), num)
            if key in self._fig_seen:
                continue
            fname = f"{prefix}_{len(self._fig_seen)+1:03d}.png"
            out = self.fig_dir / fname
            fig.savefig(out, dpi=220, bbox_inches="tight")
            self._fig_seen.add(key)

    def flush_figures(self):
        """Llama esto al final si quieres forzar el guardado."""
        self._save_all_open_figures(prefix="fig")

    def enable_autosave_figures_each_cell(self):
        """
        Funciona en VS Code/Jupyter incluso si NO llamas plt.show():
        - parchea matplotlib_inline.backend_inline.flush_figures
        - registra post_run_cell como backup
        - parchea plt.close para guardar antes de cerrar
        """
        if self._hooked:
            return

        import matplotlib.pyplot as plt

        # 1) antes de close()
        _orig_close = plt.close
        def _close(fig=None):
            try:
                self._save_all_open_figures(prefix="fig")
            except Exception:
                pass
            return _orig_close(fig)
        plt.close = _close

        # 2) backend inline (lo que usa VS Code normalmente)
        try:
            import matplotlib_inline.backend_inline as bi
            if hasattr(bi, "flush_figures"):
                _orig_flush = bi.flush_figures
                def _flush(*a, **k):
                    try:
                        self._save_all_open_figures(prefix="fig")
                    except Exception:
                        pass
                    return _orig_flush(*a, **k)
                bi.flush_figures = _flush
        except Exception:
            pass

        # 3) post_run_cell como respaldo
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                def _post(_):
                    self._save_all_open_figures(prefix="fig")
                ip.events.register("post_run_cell", _post)
        except Exception:
            pass

        self._hooked = True
        self.save_json("figure_capture.json", {"enabled": True, "fig_dir": str(self.fig_dir)})

    # ---------- Notebook copy ----------
    def copy_notebook_checked(self, notebook_path: str | Path, max_age_sec: int = 30):
        """
        Copia el ipynb al RUN_DIR como executed.ipynb, pero primero verifica
        que el archivo fue guardado recientemente (Ctrl+S).
        """
        import shutil, time as _time

        src = Path(notebook_path)
        if not src.exists():
            raise FileNotFoundError(f"No encuentro el notebook: {src}")

        age = _time.time() - src.stat().st_mtime
        if age > max_age_sec:
            raise RuntimeError(
                f"Tu .ipynb en disco parece NO estar guardado (age={age:.1f}s > {max_age_sec}s).\n"
                "En VS Code: presiona Ctrl+S y vuelve a ejecutar esta celda."
            )

        dst = self.run_dir / "executed.ipynb"
        shutil.copy2(src, dst)
        return dst

