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
