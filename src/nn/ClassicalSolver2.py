import os
import torch
import torch.nn as nn
from src.utils.logger import Logging


class _NullLogger:
    def get_output_dir(self):
        os.makedirs("./results/models/checkpoints/_tmp", exist_ok=True)
        return "./results/models/checkpoints/_tmp"

    def print(self, msg: str):
        print(msg)


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


class ClassicalSolver2(nn.Module):
    """
    Red clásica para Schrödinger 1D:
      input  : (t, x)  -> Tensor (N,2)
      output : (Re psi, Im psi) -> Tensor (N,2)

    Soporta hard-BC Dirichlet (psi=0 en x_min/x_max) para box/barrier:
      psi_hat(t,x) = g(x) * NN(t,x)
      g(x) = s*(1-s) con s=(x-x_min)/(x_max-x_min)
    """

    def __init__(self, args, logger: Logging = None, data=None, device=None):
        super().__init__()
        self.args = args
        self.data = data

        # logger (opcional)
        self.logger = logger if logger is not None else _NullLogger()

        # device/dtype
        self.device = torch.device(
            device if device is not None
            else args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = _parse_dtype(args.get("dtype", torch.float64))

        self.batch_size = int(self.args.get("batch_size", 64))
        self.epochs = int(self.args.get("epochs", 2000))
        self.loss_history = []
        self.l2_rel_history = []

        # arquitectura
        self.classic_network = self.args["classic_network"]  # e.g. [2,64,64,64,2]
        h = self.classic_network[-2]

        self.preprocessor = nn.Sequential(
            nn.Linear(self.classic_network[0], h),
            nn.Tanh(),
            nn.Linear(h, h),
        )

        self.hidden = nn.Sequential(
            nn.Linear(h, h),
        )

        self.postprocessor = nn.Sequential(
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, self.classic_network[-1]),
        )

        self.activation = nn.Tanh()

        # init + mover a device/dtype
        self._initialize_logging()
        self._initialize_weights()
        self.to(self.device)
        if self.dtype == torch.float64:
            self.double()
        else:
            self.float()

    def _initialize_weights(self):
        # Xavier en todas las capas lineales
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        self.logger.print(f"checkpoint path: {self.log_path=}")

    def _hard_bc_gate(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1) coordenada espacial
        eq = self.args.get("eq_params", {})
        x_min = float(eq.get("x_min", 0.0))
        x_max = float(eq.get("x_max", 1.0))
        L = (x_max - x_min) if (x_max - x_min) != 0 else 1.0

        s = (x - x_min) / L  # [0,1]
        g = s * (1.0 - s)    # 0 en bordes, suave en interior
        return g

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        """
        tx: Tensor (N,2) con columnas [t, x]
        return: Tensor (N,2) = (Re psi, Im psi)
        """
        if tx.dim() != 2 or tx.shape[1] != 2:
            raise ValueError(f"Expected (N,2) [t,x], got {tuple(tx.shape)}")

        tx = tx.to(self.device, dtype=self.dtype)
        t = tx[:, 0:1]
        x = tx[:, 1:2]

        # red base
        z = self.preprocessor(tx)
        z = self.hidden(self.activation(z))
        out = self.postprocessor(self.activation(z))

        # hard BC opcional para box/barrier
        hard_bc = bool(self.args.get("hard_bc", False))
        ex = str(self.args.get("eq_params", {}).get("example", "")).lower()

        if hard_bc and ex in ("box", "barrier"):
            g = self._hard_bc_gate(x)
            out = g * out

        return out

    def save_state(self):
        os.makedirs(self.log_path, exist_ok=True)
        model_path_state = os.path.join(self.log_path, "model.pth")
        model_path_weights = os.path.join(self.log_path, "model_weights.pth")

        state = {
            "args": self.args,
            "state_dict": self.state_dict(),
            "loss_history": self.loss_history,
            "l2_rel_history": self.l2_rel_history,
            "log_path": self.log_path,
        }
        torch.save(state, model_path_state)
        torch.save(self.state_dict(), model_path_weights)
        self.logger.print(f"Model state saved to {model_path_state}")
        self.logger.print(f"Model weights saved to {model_path_weights}")

    @classmethod
    def load_state(cls, file_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cpu")
        ckpt = torch.load(file_path, map_location=map_location)
        model = cls(ckpt["args"], logger=None)
        model.load_state_dict(ckpt["state_dict"])
        model.loss_history = ckpt.get("loss_history", [])
        model.l2_rel_history = ckpt.get("l2_rel_history", [])
        return model

