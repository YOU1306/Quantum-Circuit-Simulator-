# Project1
(Hi, I am Rajat)
🚀 Quantum Circuit Simulator (C/CUDA + Interactive Visualization)

A high-performance Quantum Circuit Simulator implementing core quantum-computing operations using a C/CUDA backend, combined with an interactive visualization layer for building, testing, and understanding quantum circuits.

This project aims to provide a fast, modular, and educational framework for simulating quantum gates, qubits, and multi-qubit systems—while also visualizing state evolution in real-time.

🧠 Project Overview

This simulator replicates the behavior of quantum circuits using:

🔹 Backend (C/CUDA)

Efficient matrix and state-vector operations

GPU-accelerated computation for faster simulation

Support for multi-qubit systems

Modular design to add new gates easily

Handles:

Single-qubit gates (X, Y, Z, H)

Multi-qubit gates (CNOT, SWAP)

Rotation gates (Rx, Ry, Rz)

Measurement operation

Optimized linear algebra routines for quantum state evolution

🔹 Visualization Layer

Interactive circuit builder

Step-by-step simulation viewer

State vector visualization (Bloch sphere / amplitude bar representation)

Timeline-based gate placement

Ideal for teaching, experimentation, and prototyping

⚙️ Key Features

GPU acceleration via CUDA for scalable simulation

Custom quantum gate engine supporting unitary operations

State-vector simulation for N-qubit circuits

Error-free reversible operations through unitary transformations

Interactive UI for building circuits (drag-and-drop)

Live updates showing how gates affect the quantum state

Modular design: easily extend with new gates or backends

🧩 Architecture
Quantum-Circuit-Simulator/
│
├── backend/                 # C/CUDA implementation
│   ├── matrix_ops.cu        # GPU matrix multiplication
│   ├── state_vector.c       # State vector operations
│   ├── gates.c              # Gate definitions
│   └── utils/               # Helpers, memory mgmt
│
├── visualization/           
│   ├── circuit_viewer/      # UI for creating circuits
│   ├── state_visualizer/    # Visual display of amplitudes/bloch sphere
│   └── controls/            # Run, pause, step buttons
│
├── examples/                # Sample circuits (Bell state, teleportation)
└── README.md                # You're reading it now!

🔬 Example: Bell State Circuit
|0> --[H]----●----
             |
|0> ---------[X]--


You can simulate this using:

applyGate(H, 0);
applyCNOT(0, 1);
simulate(state_vector);

🛠️ Tech Stack

C/C++ → Core computation

CUDA → GPU acceleration

Java / JavaUI or Unity (depending on final setup) → Visualization

Makefile → Build system

📦 How to Build & Run
1. Build CUDA backend
nvcc backend/matrix_ops.cu -o quantum_backend

2. Run a sample circuit
./quantum_backend examples/bell_state.qc

3. Launch Visualization UI

(Explain based on your UI — JavaFX/Unity)

java -jar QuantumVisualizer.jar

🧪 Sample Circuits Included

Bell State

Quantum Teleportation

Deutsch-Jozsa Algorithm

Grover’s Search (Basic version)

📘 Future Enhancements

Density matrix simulation

Noise model (decoherence + gate errors)

Drag-and-drop circuit designer

Custom shader-based GPU visualizer

Export to Qiskit-compatible format

🤝 Contributors

Rajat Pal
CSE – Quantum Computing + GPU Simulation Developer
Backend (C/CUDA), Circuit Logic, and Visualization Layer

⭐ Support

If you like the project, please ⭐ star the repository — it helps a lot!
