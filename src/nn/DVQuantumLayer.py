import torch
import torch.nn as nn
import pennylane as qml

# ------------------------------------------------------------
# Helpers: intentar conseguir batch_input sin casarnos con versión
# ------------------------------------------------------------
def _get_batch_input():
    # PennyLane moderno: qml.batch_input
    if hasattr(qml, "batch_input"):
        return qml.batch_input
    # PennyLane: pennylane.transforms.batch_input
    try:
        from pennylane.transforms import batch_input  # type: ignore
        return batch_input
    except Exception:
        return None


class DVQuantumLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_qubits = int(args["num_qubits"])
        self.num_quantum_layers = int(args["num_quantum_layers"])
        self.shots = args.get("shots", None)
        self.q_ansatz = args["q_ansatz"]
        self.problem = args.get("problem", "schrodinger")
        self.encoding = args.get("encoding", "angle")

        # ruido (si lo usas)
        self.noise_flag = bool(args.get("noise", False))

        # batching (lo nuevo)
        self.use_batching = bool(args.get("dv_batching", True))
        self._batch_input = _get_batch_input()

        # contador útil para debug
        self.shots_done = 0

        # --------------------------
        # Parámetros entrenables
        # --------------------------
        if self.q_ansatz in ["layered_circuit", "alternating_layer_tdcnot"]:
            self.params = nn.Parameter(
                torch.empty(self.num_quantum_layers, self.num_qubits * 4, dtype=torch.float32)
            )
        elif self.q_ansatz == "sim_circ_19":
            self.params = nn.Parameter(
                torch.empty(self.num_quantum_layers, self.num_qubits * 2, dtype=torch.float32)
            )
        elif self.q_ansatz == "farhi":
            self.params = nn.Parameter(
                torch.empty(self.num_quantum_layers, (2 * self.num_qubits - 2), dtype=torch.float32)
            )
        elif self.q_ansatz == "sim_circ_15":
            self.params = nn.Parameter(
                torch.empty(self.num_quantum_layers, self.num_qubits * 4, dtype=torch.float32)
            )
        elif self.q_ansatz == "sim_circ_5":
            self.params = nn.Parameter(
                torch.empty(self.num_quantum_layers, (3 * self.num_qubits) * self.num_qubits, dtype=torch.float32)
            )
        else:
            raise ValueError(f"Parameters are not initialized. Check q_ansatz='{self.q_ansatz}'.")

        self._initialize_weights()

        # --------------------------
        # Device + QNode
        # --------------------------
        diff_method = "best" if self.noise_flag else "backprop"

        if self.noise_flag:
            # Si vas a usar ruido, aquí normalmente necesitarías qiskit/noise models.
            # Para no romper ambientes donde no esté qiskit, lo dejamos explícito.
            raise NotImplementedError(
                "noise=True no está soportado en esta versión batcheada. "
                "Pon noise=False para entrenar rápido y estable."
            )
        else:
            self.dev = qml.device("default.qubit", wires=self.num_qubits)
            self.circuit = qml.QNode(self._quantum_circuit, self.dev, interface="torch", diff_method=diff_method)
        self.dv_batching_mode = args.get("dv_batching_mode", "broadcast")  # "broadcast" o "loop"
        self._broadcast_ok = None  # cache: True/False

        

    def _initialize_weights(self):
        if self.q_ansatz == "farhi":
            torch.nn.init.xavier_normal_(self.params.view(self.num_quantum_layers, (2 * self.num_qubits - 2)))
        elif self.q_ansatz in ["sim_circ_15", "layered_circuit", "alternating_layer_tdcnot"]:
            torch.nn.init.xavier_normal_(self.params.view(self.num_quantum_layers, self.num_qubits * 4))
        elif self.q_ansatz == "sim_circ_19":
            torch.nn.init.xavier_normal_(self.params.view(self.num_quantum_layers, self.num_qubits * 2))
        elif self.q_ansatz == "sim_circ_5":
            torch.nn.init.xavier_normal_(self.params.view(self.num_quantum_layers, (3 * self.num_qubits) * self.num_qubits))
        else:
            raise ValueError(f"Invalid q_ansatz value: {self.q_ansatz}")

    # --------------------------
    # Circuito
    # --------------------------
    def _quantum_circuit(self, x):
        if self.encoding == "amplitude":
            qml.templates.AmplitudeEmbedding(x, wires=range(self.num_qubits), normalize=True, pad_with=0.0)
        else:
            qml.templates.AngleEmbedding(x, wires=range(self.num_qubits), rotation="X")

        if self.q_ansatz == "layered_circuit":
            for layer in range(self.num_quantum_layers):
                self.layered_circuit(self.params[layer])
        elif self.q_ansatz == "alternating_layer_tdcnot":
            for layer in range(self.num_quantum_layers):
                self.alternating_layer_tdcnot(self.params[layer])
        elif self.q_ansatz == "sim_circ_19":
            for layer in range(self.num_quantum_layers):
                self.sim_circ_19(self.params[layer])
        elif self.q_ansatz == "farhi":
            for layer in range(self.num_quantum_layers):
                self.farhi_ansatz(self.params[layer])
        elif self.q_ansatz == "sim_circ_15":
            for layer in range(self.num_quantum_layers):
                self.create_sim_circuit_15(self.params[layer])
        elif self.q_ansatz == "sim_circ_5":
            for layer in range(self.num_quantum_layers):
                self.create_circuit_5(self.params[layer])
        else:
            raise ValueError(f"Unsupported q_ansatz={self.q_ansatz}")

        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    # --------------------------
    # Forward (batcheado si se puede)
    # --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_qubits)
        return: (B, num_qubits) expvals
        """
        # contador (si lo usas en logs)
        self.shots_done += x.shape[0]

        # -------------------------------
        # 1) Ruta rápida: broadcasting
        # -------------------------------
        if self.dv_batching_mode == "broadcast" and (self._broadcast_ok is not False):
            try:
                res = self.circuit(x)  # <-- batcheo por broadcasting

                # res suele ser list/tuple de largo num_qubits, cada item shape (B,)
                if isinstance(res, (list, tuple)):
                    out = torch.stack([r for r in res], dim=1)  # (B, num_qubits)
                else:
                    out = res

                if out.dim() == 1:
                    out = out.unsqueeze(0)

                self._broadcast_ok = True
                return out

            except Exception:
                # si broadcasting no está soportado, caemos al loop
                self._broadcast_ok = False

        # -------------------------------
        # 2) Fallback seguro: loop (lento)
        # -------------------------------
        return torch.stack([torch.stack(self.circuit(sample)) for sample in x])




    # --------------------------
    # Ansätze (conservados)
    # --------------------------
    def layered_circuit(self, params):
        assert params is not None and len(params) == self.num_qubits * 4, "params must be 4*num_qubits"
        param_idx = 0
        for q in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=q); param_idx += 1
            qml.RX(params[param_idx], wires=q); param_idx += 1

        qml.Barrier(wires=range(self.num_qubits))
        for q in range(self.num_qubits):
            qml.CNOT(wires=[q, (q + 1) % self.num_qubits])

        qml.Barrier(wires=range(self.num_qubits))
        for q in range(self.num_qubits):
            qml.RX(params[param_idx], wires=q); param_idx += 1
            qml.RZ(params[param_idx], wires=q); param_idx += 1

    def alternating_layer_tdcnot(self, params):
        assert params is not None and len(params) == self.num_qubits * 4, "params must be 4*num_qubits"
        param_idx = 0

        def build_tdcnot(ctrl, tgt):
            nonlocal param_idx
            qml.RY(params[param_idx], wires=ctrl); param_idx += 1
            qml.RY(params[param_idx], wires=tgt);  param_idx += 1
            qml.CNOT(wires=[ctrl, tgt])
            qml.RZ(params[param_idx], wires=ctrl); param_idx += 1
            qml.RZ(params[param_idx], wires=tgt);  param_idx += 1

        for i in range(self.num_qubits - 1)[::2]:
            build_tdcnot(i, (i + 1) % self.num_qubits)

        qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits)[1::2]:
            build_tdcnot(i, (i + 1) % self.num_qubits)

    def sim_circ_19(self, params):
        def add_rotations():
            pc = 0
            for i in range(self.num_qubits):
                qml.RX(params[pc], wires=i); pc += 1
            for i in range(self.num_qubits):
                qml.RZ(params[pc], wires=i); pc += 1
            qml.Barrier(wires=range(self.num_qubits))

        def add_entangling_gates():
            qml.CNOT(wires=[self.num_qubits - 1, 0])
            for i in reversed(range(1, self.num_qubits)):
                qml.CNOT(wires=[i - 1, i])

        add_rotations()
        add_entangling_gates()

    def farhi_ansatz(self, params):
        if len(params) != (2 * self.num_qubits - 2):
            raise ValueError("Insufficient parameters for farhi ansatz")

        def RXX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RX(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        def RZX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        pc = 0
        for i in range(self.num_qubits - 1):
            RXX(params[pc], wires=[self.num_qubits - 1, i]); pc += 1
        for i in range(self.num_qubits - 1):
            RZX(params[pc], wires=[self.num_qubits - 1, i]); pc += 1

    def create_sim_circuit_15(self, params):
        if params is None or len(params) != 4 * self.num_qubits:
            raise ValueError("params must be 4*num_qubits")

        p = 0
        for i in range(self.num_qubits):
            qml.RY(params[p], wires=i); p += 1
        qml.Barrier(wires=range(self.num_qubits))

        # bloque entangling (simplificado, conserva tu idea)
        for i in reversed(range(self.num_qubits)):
            qml.CRZ(params[p % len(params)], wires=[i, (i + 1) % self.num_qubits])
        qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits):
            qml.RX(params[p % len(params)], wires=i); p += 1
        qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits):
            ctrl = (i + self.num_qubits - 1) % self.num_qubits
            tgt = (ctrl + 3) % self.num_qubits
            qml.CRZ(params[p % len(params)], wires=[ctrl, tgt]); p += 1

        qml.Barrier(wires=range(self.num_qubits))

    def create_circuit_5(self, params):
        expected = (3 * self.num_qubits) * self.num_qubits
        if params is None or len(params) != expected:
            raise ValueError(f"Expected {expected} params, got {len(params)}")

        p = 0
        for i in range(self.num_qubits):
            qml.RX(params[p], wires=i); p += 1
        for i in range(self.num_qubits):
            qml.RZ(params[p], wires=i); p += 1

        qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits - 1, -1, -1):
            for j in range(self.num_qubits - 1, -1, -1):
                if j != i:
                    qml.CRZ(params[p], wires=[i, j]); p += 1

        qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits):
            qml.RX(params[p], wires=i); p += 1
        for i in range(self.num_qubits):
            qml.RZ(params[p], wires=i); p += 1

        qml.Barrier(wires=range(self.num_qubits))

