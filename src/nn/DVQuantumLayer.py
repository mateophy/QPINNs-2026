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
        # Parámetros entrenables (CONSISTENTES)
        # --------------------------
        n = self.num_qubits
        if n < 1:
            raise ValueError("num_qubits debe ser >= 1")

        if self.q_ansatz == "sim_circ_19":
            self.params_per_layer = 2 * n

        elif self.q_ansatz == "farhi":
            if n < 2:
                raise ValueError("farhi requiere num_qubits >= 2")
            self.params_per_layer = 2 * n - 2

        elif self.q_ansatz in ["layered_circuit", "sim_circ_15"]:
            self.params_per_layer = 4 * n

        elif self.q_ansatz == "alternating_layer_tdcnot":
            if n < 2:
                raise ValueError("alternating_layer_tdcnot requiere num_qubits >= 2")
            # Con n impar, tu patrón genera (n-1) bloques por capa; con n par genera n bloques.
            blocks = n - (n % 2)   # n par -> n; n impar -> n-1
            self.params_per_layer = 4 * blocks  # 4 params por bloque (RY,RY,RZ,RZ)

        elif self.q_ansatz == "sim_circ_5":
            # Tu circuito usa: 2n (RX/RZ) + n(n-1) (CRZ i!=j) + 2n (RX/RZ)
            self.params_per_layer = n * (n - 1) + 4 * n  # = n^2 + 3n

        else:
            raise ValueError(f"Parameters are not initialized. Check q_ansatz='{self.q_ansatz}'.")

        self.params = nn.Parameter(
            torch.empty(self.num_quantum_layers, self.params_per_layer, dtype=torch.float32)
        )

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
        torch.nn.init.xavier_normal_(
            self.params.view(self.num_quantum_layers, self.params.shape[1])
        )

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
    # Ansätze (consistentes)
    # --------------------------
    def layered_circuit(self, params):
        expected = getattr(self, "params_per_layer", 4 * self.num_qubits)
        if params is None or len(params) != expected:
            raise ValueError(
                f"[layered_circuit] Expected {expected} params per layer, got {0 if params is None else len(params)}. "
                "Ajusta self.params_per_layer en __init__."
            )

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

        # Fail-fast: no params desperdiciados
        if param_idx != expected:
            raise ValueError(f"[layered_circuit] Used {param_idx} params but expected {expected}.")

    def alternating_layer_tdcnot(self, params):
        # Con n impar, tu patrón actual genera (n-1) bloques por capa, no n.
        n = self.num_qubits
        expected = getattr(self, "params_per_layer", 4 * (n - (n % 2)))
        if params is None or len(params) != expected:
            raise ValueError(
                f"[alternating_layer_tdcnot] Expected {expected} params per layer, got {0 if params is None else len(params)}. "
                "Ajusta self.params_per_layer en __init__ (para n impar debe ser 4*(n-1))."
            )

        param_idx = 0

        def build_tdcnot(ctrl, tgt):
            nonlocal param_idx
            qml.RY(params[param_idx], wires=ctrl); param_idx += 1
            qml.RY(params[param_idx], wires=tgt);  param_idx += 1
            qml.CNOT(wires=[ctrl, tgt])
            qml.RZ(params[param_idx], wires=ctrl); param_idx += 1
            qml.RZ(params[param_idx], wires=tgt);  param_idx += 1

        # Pares (0,1), (2,3), ...
        for i in range(self.num_qubits - 1)[::2]:
            build_tdcnot(i, (i + 1) % self.num_qubits)

        qml.Barrier(wires=range(self.num_qubits))

        # Pares (1,2), (3,4), ...
        for i in range(self.num_qubits)[1::2]:
            build_tdcnot(i, (i + 1) % self.num_qubits)

        # Fail-fast
        if param_idx != expected:
            raise ValueError(f"[alternating_layer_tdcnot] Used {param_idx} params but expected {expected}.")

    def sim_circ_19(self, params):
        expected = getattr(self, "params_per_layer", 2 * self.num_qubits)
        if params is None or len(params) != expected:
            raise ValueError(
                f"[sim_circ_19] Expected {expected} params per layer, got {0 if params is None else len(params)}. "
                "Ajusta self.params_per_layer en __init__."
            )

        def add_rotations():
            pc = 0
            for i in range(self.num_qubits):
                qml.RX(params[pc], wires=i); pc += 1
            for i in range(self.num_qubits):
                qml.RZ(params[pc], wires=i); pc += 1
            qml.Barrier(wires=range(self.num_qubits))

            if pc != expected:
                raise ValueError(f"[sim_circ_19] Used {pc} params but expected {expected}.")

        def add_entangling_gates():
            qml.CNOT(wires=[self.num_qubits - 1, 0])
            for i in reversed(range(1, self.num_qubits)):
                qml.CNOT(wires=[i - 1, i])

        add_rotations()
        add_entangling_gates()

    def farhi_ansatz(self, params):
        expected = getattr(self, "params_per_layer", (2 * self.num_qubits - 2))
        if params is None or len(params) != expected:
            raise ValueError(
                f"[farhi] Expected {expected} params per layer, got {0 if params is None else len(params)}. "
                "Ajusta self.params_per_layer en __init__."
            )

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

        if pc != expected:
            raise ValueError(f"[farhi] Used {pc} params but expected {expected}.")

    def create_sim_circuit_15(self, params):
        expected = getattr(self, "params_per_layer", 4 * self.num_qubits)
        if params is None or len(params) != expected:
            raise ValueError(
                f"[sim_circ_15] Expected {expected} params per layer, got {0 if params is None else len(params)}. "
                "Ajusta self.params_per_layer en __init__."
            )

        # Importante: consumo SECUENCIAL (sin %), para que el conteo sea 1:1
        p = 0
        for i in range(self.num_qubits):
            qml.RY(params[p], wires=i); p += 1
        qml.Barrier(wires=range(self.num_qubits))

        # bloque entangling CRZ con parámetros secuenciales
        for i in reversed(range(self.num_qubits)):
            qml.CRZ(params[p], wires=[i, (i + 1) % self.num_qubits]); p += 1
        qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits):
            qml.RX(params[p], wires=i); p += 1
        qml.Barrier(wires=range(self.num_qubits))

        # evita ctrl==tgt cuando num_qubits divide 3 (ej. 3)
        shift = 3 % self.num_qubits
        if shift == 0:
            shift = 1

        for i in range(self.num_qubits):
            ctrl = (i + self.num_qubits - 1) % self.num_qubits
            tgt = (ctrl + shift) % self.num_qubits
            qml.CRZ(params[p], wires=[ctrl, tgt]); p += 1

        qml.Barrier(wires=range(self.num_qubits))

        if p != expected:
            raise ValueError(f"[sim_circ_15] Used {p} params but expected {expected}.")

    def create_circuit_5(self, params):
        # El circuito realmente usa: 2n + n(n-1) + 2n = n(n-1) + 4n
        n = self.num_qubits
        expected = getattr(self, "params_per_layer", (n * (n - 1) + 4 * n))
        if params is None or len(params) != expected:
            raise ValueError(
                f"[sim_circ_5] Expected {expected} params per layer, got {0 if params is None else len(params)}. "
                "Ajusta self.params_per_layer en __init__ (NO uses 3*n*n)."
            )

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

        if p != expected:
            raise ValueError(f"[sim_circ_5] Used {p} params but expected {expected}.")

