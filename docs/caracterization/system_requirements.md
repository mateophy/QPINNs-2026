# Profiling of the system needs.
After several reviews of the system, now we understand the needs of the system needed to run.

## Characteristics.
We now point out the stuff we know afterhand.

- **Size of our circuits:** Now we characterize `sim_circuit_19` consisting in basis of 15 gates, _ordered on a ring topology_, revolving around _CRX (can be exchanged for other entangling gates)_, _RX_ and _RZ_ gates and an _angle/amplitude embedding (can be to a layer of gates _RX_ acting on all wires)_ for data batching.
- **Fidelity:** We need to ensure the entangling and expresivity capabilities of the circuits to use. Simulators supported on pennylane: _default.mixed, lighting.qubit, qiskit.aer, and qulacs.simulator,_ were used to estimate the fidelity behaviour of the circuits, but we're still open for further recommendations on other alternatives to ensure accurate results.
- **Quiskit:** No the code it's entirely written on pennylane but we're open to the idea of rewriting code to match popular frameworks.
- **Epochs / Iterations:** For low frequency oscillators _(quantum harmonic oscillator on mode n = 0)_ around 2000 Epochs were needed with batch sizes of 17 random points _(7 interior, 5 on boundary and 5 on initial conditions)_, using the same conditions, for more oscillatory solutions _(quantum harmonic oscillator on n=3,4)_ were needed up to 20000 epochs with more neurons and deeper circuits.
- **Noise behaviour:** Multiple tests were made with pennylane noise models for: _reset error, amplitude damping and channel depolarizing_ with probabilities up to 20%, using up to 20~50 shots the QCPINN has no issue to train, and the loss vs epochs evolution suffers minor changes, _we evidence random peaks appearing with the induction of these errors, and the need for slightly more epochs to obtaine comparable results, around an increase of 5%_.

## Pipeline
Now let's talk about whats happening on the task we're simulating.

The HQCPINN consists hybrid PINN scheme where quantum neuron layers meet classical neuron layers, this classical part serves as a pre/post-processor for the quantum layers take place. For each neuron, the gates present on the circuit comes from the same operation, but the parameters used by each gate it's independent on the scheme, serving as learnable pararameters for the neural networks. Not going further, the entangling capabilities of qubits opens the possibility for the reduction of the needed parameters to achieve the representation of phenomena described.
On detail, the process that's been studied it's the construction of a Hybrid Quantum-Classical PINN, where the classical part serves as pre and post -processors, with quantum layers using a basis circuit, which is actively changing parameters, but the operations remain the same.
