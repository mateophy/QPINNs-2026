import os
import torch
import torch.nn as nn

import pennylane as qml
import matplotlib.pyplot as plt

from src.utils.logger import Logging
from src.nn.DVQuantumLayer import DVQuantumLayer


def complex_mse_loss_magnitude(input_complex, target_complex):
    complex_diff = input_complex - target_complex
    return torch.mean(complex_diff.real**2 + complex_diff.imag**2)


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


class DVPDESolver(nn.Module):
    """
    Input : (B,2) = [t,x]
    Output: (B,2) = [Re,Im]

    Opcional:
      - hard_bc: g(x) * psi_hat para box/barrier
      - global_phase: psi -> psi * exp(-i omega t)
      - global_phase_spatial_only: la red base se evalúa con t=0 (autoestados)
    """

    def __init__(self, args, logger: Logging, data=None, device=None):
        super().__init__()
        self.args = args
        self.data = data
        self.logger = logger

        self.device = torch.device(
            device if device is not None
            else args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = _parse_dtype(args.get("dtype", torch.float64))

        self.batch_size = int(self.args.get("batch_size", 64))
        self.epochs = int(self.args.get("epochs", 1000))
        self.loss_history = []
        self.l2_rel_history = []

        # DV
        self.num_qubits = int(self.args["num_qubits"])
        self.encoding = self.args.get("encoding", "angle")
        self.draw_quantum_circuit_flag = True

        # ---- Modo estacionario (clave para autoestados: HO/box/barrier) ----
        self.global_phase = bool(self.args.get("global_phase", False))
        self.global_phase_spatial_only = bool(self.args.get("global_phase_spatial_only", False))

        if self.global_phase:
            # omega entrenable
            omega0 = float(self.args.get("global_phase_omega_init", 0.0))
            self.global_phase_omega = nn.Parameter(
                torch.tensor(omega0, device=self.device, dtype=self.dtype)
            )

        # Arquitectura clásica (simple, robusta)
        # classic_network = [2, hidden, 2] (si pasas más, aquí usamos hidden=classic_network[-2])
        self.classic_network = self.args["classic_network"]
        hidden = int(self.classic_network[-2])

        self.preprocessor = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.num_qubits),
        ).to(self.device, dtype=self.dtype)

        self.quantum_layer = DVQuantumLayer(self.args).to(self.device)

        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        ).to(self.device, dtype=self.dtype)

        # Opt
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=float(self.args.get("lr", 1e-3)),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=1000
        )

        # Loss
        if self.args.get("problem") in ["complex_wave", "schrodinger"]:
            self.loss_fn = complex_mse_loss_magnitude
        else:
            self.loss_fn = torch.nn.MSELoss()

        self._initialize_logging()
        self._initialize_weights_all_linear()

        # mover modelo
        self.to(self.device)
        if self.dtype == torch.float64:
            self.double()
        else:
            self.float()

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()

    def _initialize_weights_all_linear(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------
    # Hard BC (box/barrier)
    # -------------------------
    def _apply_hard_bc_if_needed(self, tx: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if not bool(self.args.get("hard_bc", False)):
            return out

        eq = self.args.get("eq_params", {})
        ex = str(eq.get("example", "")).lower()
        if ex not in ("box", "barrier"):
            return out

        x = tx[:, 1:2]
        x_min = float(eq.get("x_min", 0.0))
        x_max = float(eq.get("x_max", 1.0))
        L = max(1e-12, x_max - x_min)

        # g(x) normalizado: max ~ 1
        g = 4.0 * (x - x_min) * (x_max - x) / (L * L)
        p = float(self.args.get("hard_bc_power", 1.0))
        if p != 1.0:
            g = g.clamp(min=0.0) ** p

        return g * out

    # -------------------------
    # Global phase rotation
    # -------------------------
    def _apply_global_phase(self, tx: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if not self.global_phase:
            return out
        t = tx[:, 0:1]
        omega = self.global_phase_omega

        # exp(-i omega t) = cos(ωt) - i sin(ωt)
        c = torch.cos(omega * t)
        s = torch.sin(omega * t)

        re = out[:, 0:1]
        im = out[:, 1:2]

        # (re + i im)*(c - i s) = (re c + im s) + i(im c - re s)
        re2 = re * c + im * s
        im2 = im * c - re * s
        return torch.cat([re2, im2], dim=1)


    # -------------------------
    # Forward
    # -------------------------
    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        if tx.dim() != 2 or tx.shape[1] != 2:
            raise ValueError(f"Expected (B,2) [t,x], got {tuple(tx.shape)}")

        tx = tx.to(self.device, dtype=self.dtype)

        # Si queremos autoestado: la red base solo ve t=0
        if self.global_phase and self.global_phase_spatial_only:
            t0 = torch.zeros_like(tx[:, 0:1])
            tx_base = torch.cat([t0, tx[:, 1:2]], dim=1)
        else:
            tx_base = tx

        pre = self.preprocessor(tx_base)
        if self.args.get("angle_squash", False):
            pre = torch.pi * torch.tanh(pre)

        if self.draw_quantum_circuit_flag:
            self.draw_quantum_circuit(pre)
            self.draw_quantum_circuit_flag = False

        q_out = self.quantum_layer(pre).to(self.device, dtype=pre.dtype)
        out = self.postprocessor(q_out)

        # 1) aplica fase global (si está activa)
        out = self._apply_global_phase(tx, out)

        # 2) aplica hard_bc (si aplica)
        out = self._apply_hard_bc_if_needed(tx, out)

        return out

    # -------------------------
    # Save / Draw
    # -------------------------
    def save_state(self):
        state = {
            "args": self.args,
            "classic_network": self.classic_network,
            "quantum_layer": self.quantum_layer.state_dict(),
            "preprocessor": self.preprocessor.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "loss_history": self.loss_history,
            "l2_rel_history": getattr(self, "l2_rel_history", []),
            "log_path": self.log_path,
        }

        model_path_state = os.path.join(self.log_path, "model.pth")
        model_path_weights = os.path.join(self.log_path, "model_weights.pth")

        with open(model_path_state, "wb") as f:
            torch.save(state, f)
        self.logger.print(f"Model state saved to {model_path_state}")

        torch.save(self.state_dict(), model_path_weights)
        self.logger.print(f"Model weights saved to {model_path_weights}")

    def draw_quantum_circuit(self, x):
        try:
            self.logger.print("The circuit used in the study:")
            if getattr(self.quantum_layer, "params", None) is not None:
                fig, ax = qml.draw_mpl(self.quantum_layer.circuit)(x[0])
                plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                plt.close()
        except Exception as e:
            self.logger.print(f"Failed to draw quantum circuit: {str(e)}")




