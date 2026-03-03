# Neural Siege: High-Throughput Vectorized Multi-Agent Simulation

Neural Siege is a research-grade, massively parallel 2D grid simulation designed for studying multi-agent reinforcement learning (MARL), evolutionary dynamics, and heterogeneous policy architectures. Built on a PyTorch-native "Struct-of-Arrays" (SoA) engine, it enables thousands of independent agents to interact, learn via PPO, and evolve through mutation-driven lineage within a single GPU-accelerated environment.

[![PyTorch](https://img.shields.io/badge/Backend-PyTorch_2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python)](https://www.python.org/)
[![Performance](https://img.shields.io/badge/Optimization-vmap_/_TorchScript-orange)](https://pytorch.org/docs/stable/func.html)

## Key Highlights
- **Vectorized Physics & Combat:** Core engine resolves movement conflicts, raycasting, and focus-fire damage using pure tensor operations, minimizing Python-loop overhead.
- **Heterogeneous Policy Architectures:** Supports multiple brain types including `MirrorBrain` (Propose-Reflect architecture), `TronBrain` (Cross-Attention fusion), and standard `TransformerBrain`.
- **Inference Optimization:** Implements `torch.func.vmap` for vectorized inference across thousands of agents with *independent* parameter sets, and TorchScript support for low-latency execution.
- **Advanced Lineage Tracking:** Captures full genealogical trees (parent-child relationships) and rare mutation events to study behavioral propagation and trait evolution.
- **Research-Ready Telemetry:** Decoupled telemetry system producing atomic JSONL event logs, CSV snapshots, and PPO diagnostic streams without stalling the simulation.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [End-to-End Pipeline](#end-to-end-pipeline)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Brain Architectures](#brain-architectures)
7. [Configuration](#configuration)
8. [Telemetry & Persistence](#telemetry--persistence)
9. [Reproducibility & Determinism](#reproducibility--determinism)
10. [Limitations & Future Work](#limitations--future-work)
11. [Citation](#citation)

---

## System Architecture

The project follows a decoupled architecture where state management, physical resolution, and learning run independently.

```mermaid
flowchart TD
    subgraph Engine
        Registry[AgentsRegistry: SoA Tensor Store]
        Grid[Grid Tensor: 3-Channel Spatial Map]
        TickEngine[TickEngine: Combat & Physics Resolver]
    end

    subgraph Intelligence
        Brain[Policy Architectures: Mirror/Tron/Transformer]
        VMap[vmap Inference Orchestrator]
    end

    subgraph Learning
        PPO[Per-Agent PPO Runtime: Local Optimizers]
    end

    subgraph Output
        Telem[Telemetry Session: JSONL/CSV]
        Viewer[Pygame Viewer: Real-time UI]
        CP[Checkpoint Manager: Atomic State Saves]
    end

    Registry <--> Grid
    TickEngine --> Registry
    TickEngine --> Intelligence
    Intelligence --> Learning
    Learning --> Registry
    TickEngine --> Telem
    Registry --> Viewer
End-to-End Pipeline
The simulation follows a Combat-First execution lifecycle, ensuring that agents killed in a tick cannot perform movement actions in that same tick.
code
Mermaid
sequenceDiagram
    participant S as Simulation Loop
    participant B as Brains (GPU)
    participant E as Engine (Physics)
    participant L as PPO Runtime
    participant T as Telemetry

    rect rgb(240, 240, 240)
    Note right of S: Phase 1: Observation
    S->>E: Build Observations (Raycasting + Rich Context)
    S->>E: Build Action Masks
    end

    rect rgb(220, 230, 250)
    Note right of S: Phase 2: Decision
    S->>B: Batch Inference (vmap/Ensemble)
    B-->>S: Sample Actions (Logits + Values)
    end

    rect rgb(230, 250, 230)
    Note right of S: Phase 3: Resolution
    S->>E: Resolve Combat (Damage & Deaths)
    S->>E: Resolve Movement (Conflict Mitigation)
    S->>E: Apply Environment (Zones/Metabolism)
    end

    rect rgb(250, 240, 230)
    Note right of S: Phase 4: Learning & Records
    S->>L: Record Trajectories & Update Optimizers
    S->>T: Flush Event Chunks
    S->>E: Respawn (Cloning + Mutation)
    end
Repository Structure
code
Text
Neural-Siege/
├── agent/                  # Neural network policy architectures
│   ├── ensemble.py         # vmap/batching inference logic
│   ├── mirror_brain.py     # Two-pass "Reflection" transformer
│   ├── tron_brain.py       # Cross-attention sensor-fusion brain
│   └── transformer_brain.py# Standard attention-based controller
├── engine/                 # Core simulation logic
│   ├── agent_registry.py   # SoA state management
│   ├── grid.py             # Spatial tensor initialization
│   ├── mapgen.py           # Procedural terrain generation
│   ├── tick.py             # Main simulation stepper (hot loop)
│   ├── ray_engine/         # Optimized vectorized raycasting
│   └── game/               # Logic for movement masks and rules
├── rl/                     # Reinforcement learning components
│   └── ppo_runtime.py      # Per-agent independent PPO training
├── ui/                     # Graphical visualization
│   ├── viewer.py           # Pygame-based UI orchestrator
│   └── camera.py           # Coordinate transformation logic
├── utils/                  # Infra utilities
│   ├── checkpointing.py    # Atomic save/resume logic
│   ├── persistence.py      # Multiprocessing background logging
│   ├── telemetry.py        # Event recording and lineage tracking
│   └── profiler.py         # Torch/NVIDIA performance monitoring
├── config.py               # Global hyperparameters and env parsing
└── main.py                 # Application entry point
Installation
Prerequisites
Python 3.9+
CUDA-compatible GPU (recommended for large populations)
ffmpeg (optional, for video recording)
Setup
code
Bash
# Clone the repository
git clone https://github.com/TODO_REPLACE_WITH_REPO.git
cd Neural-Siege

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy pygame opencv-python imageio pyarrow
Quickstart
Headless Training (Performance Mode)
To run a high-speed training session without GUI:
code
Bash
python main.py
Interactive Visualization
To run with the Pygame viewer enabled:
code
Bash
# Set environment variable or modify config.py
export FWS_UI=1
python main.py
Resume from Checkpoint
code
Bash
export FWS_CHECKPOINT_PATH="results/sim_YYYY-MM-DD_HH-MM-SS/checkpoints/latest.txt"
python main.py
Brain Architectures
Neural Siege supports distinct architectural paradigms for agent control:
Architecture	Paradigm	Key Mechanism
MirrorBrain	Propose-Reflect	Pass 1 builds an action; Pass 2 builds a "reflection token" to edit the proposal based on internal uncertainty.
TronBrain	Sensor Fusion	Multi-stage Transformer that encodes rays, encodes semantic context, and fuses them via cross-attention.
TransformerBrain	Global Attention	Standard Transformer encoder treating rays as a sequence and rich features as a global context token.
Configuration
The simulation is controlled via config.py and environment variables.
Variable	Description	Default
FWS_MAX_AGENTS	Maximum agent capacity in the registry.	700
FWS_USE_VMAP	Enables torch.func.vmap for inference.	True
FWS_PPO_WINDOW_TICKS	Rollout horizon before an RL update.	512
FWS_RESP_FLOOR_PER_TEAM	Minimum population maintained by respawner.	190
FWS_MUT_STD	Standard deviation for mutation noise.	0.05
FWS_RAY_TOKENS	Number of LIDAR rays per agent.	32
Telemetry & Persistence
Neural Siege utilizes a non-blocking background process for logging to ensure disk I/O does not bottleneck the GPU.
agent_life.csv: A periodic snapshot of every agent's lifetime stats (K/D, damage, parent ID, lifespan).
events/events_XXXX.jsonl: High-fidelity records of every birth, death, and attack event.
lineage_edges.csv: A parent-child adjacency list, enabling reconstruction of evolutionary trees.
ppo_training_telemetry.csv: Detailed diagnostics (KL divergence, explained variance, loss components).
Reproducibility & Determinism
The system enforces a strict reproducibility policy:
Global Seeding: random, numpy, and torch (CPU/CUDA) are seeded via FWS_SEED.
RNG Checkpointing: Random states are serialized in checkpoints to ensure deterministic resumes.
Atomic Persistence: All JSON and state files are written using a temp-and-replace strategy to prevent corruption.
Note: Absolute bit-level determinism on GPU may be subject to non-deterministic CUDA kernels (e.g., atomic additions in index_add_). Use torch.use_deterministic_algorithms(True) for strict research requirements (may incur performance penalties).
Limitations & Future Work
Grid Resolution: Currently optimized for 2D discrete grids. Extension to continuous space is not currently supported.
Memory Consumption: Per-agent brain storage (without parameter sharing) scales linearly with MAX_AGENTS.
Communication: Agents currently do not have an explicit communication channel (radio/messages), though they can observe ally positions.
Citation
If you use this simulation in your research, please cite it as:
code
Bibtex
@software{NeuralSiege2026,
  author = {TODO: Your Name/Lab Name},
  title = {Neural Siege: High-Throughput Vectorized Multi-Agent Simulation},
  year = {2026},
  url = {https://github.com/TODO_REPLACE_WITH_REPO}
}
License
Distributed under the MIT License. See LICENSE for more information.
