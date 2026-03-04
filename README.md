# Neural Siege (`Infinite_War_Simulation`) — Repository Summary

## Purpose

**Neural Siege** (`Infinite_War_Simulation`) is a research-oriented, high-throughput multi-agent combat simulation framework for grid-based environments. It supports two core workflows:

1. **High-speed simulation execution** (headless by default, optional UI inspection), with GPU-accelerated tensorized runtime paths.
2. **Online reinforcement learning** via **per-agent PPO**, where each agent maintains independent policy and optimizer state (no parameter sharing).

The repository is notable not only for simulation and learning features, but also for strong **operational robustness** for a research prototype: atomic checkpointing, resumable runs, append-safe telemetry, crash hygiene, and artifact continuity across resumes.

---

## What the Repository Implements

At a high level, the codebase combines:

* a **vectorized simulation engine** for grid combat,
* **multiple transformer-family policy architectures**,
* an **integrated per-agent PPO runtime**, and
* **structured telemetry + persistence + checkpointing** designed for long-running experiments.

This makes it suitable for experimentation across:

* multi-agent reinforcement learning (MARL),
* evolutionary / lineage dynamics,
* policy-architecture comparisons (`TransformerBrain`, `TronBrain`, `MirrorBrain`),
* systems/observability questions in simulation-heavy ML workflows.

---

## Repository Structure and Component Responsibilities

| Component                | Responsibility                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `main.py`                | Runtime entrypoint. Reads `FWS_*` config, initializes fresh or resumed state, selects UI/headless mode, orchestrates shutdown, summaries, and checkpointing. |
| `config.py`              | Environment-variable-driven configuration system with validation, profile overrides, and compact startup summary reporting.                                  |
| `engine/`                | Core simulation runtime: grid state, agent registry, tick loop, map generation, raycasting, action masking, spawn/respawn logic.                             |
| `agent/`                 | Policy architectures (`TransformerBrain`, `TronBrain`, `MirrorBrain`), observation-schema helpers, and bucketed batched inference (`ensemble.py`).           |
| `rl/ppo_runtime.py`      | Per-agent PPO runtime: rollout buffering, GAE/advantage computation, multi-epoch updates, optimizer/scheduler state, checkpoint serialization.               |
| `utils/telemetry.py`     | Structured telemetry system (CSV/JSONL/event streams), lineage tracking, summary files, and append-safe resume behavior.                                     |
| `utils/checkpointing.py` | Atomic checkpoint save/load with completion markers (`DONE`), manifests, `latest.txt`, pruning, and resume-path resolution.                                  |
| `utils/persistence.py`   | Background writer process for non-blocking persistence of core run CSVs (e.g., stats and death logs).                                                        |
| `ui/`                    | Optional Pygame viewer for interactive inspection/debugging.                                                                                                 |
| `recorder/`              | Optional video/frame recording and schema-related utilities (extra dependencies required).                                                                   |
| `lineage_tree.py`        | Offline lineage visualization/analysis utility (e.g., parent-child graph exploration).                                                                       |

Supporting utilities also include profiling/sanitization helpers and source aggregation tooling (e.g., `dump_py_to_text.py`).

---

## Core Runtime Architecture and Data Flow

A central design choice is the use of **two synchronized state representations**:

* **Grid tensor**: spatial/environment representation (terrain, zones, occupancy, etc.)
* **Agent registry tensor store**: per-agent state (position, health, team, cooldowns, brain type, and related attributes)

This split enables efficient vectorized world operations while preserving structured per-agent state, but it also creates a strict correctness requirement: **grid and registry views must remain synchronized**.

### Tick Lifecycle (High-Level)

1. **Observation + Action-Mask Construction**

   * Raycasting and rich semantic feature extraction are computed for alive agents.
   * Invalid/illegal actions are masked before action selection.

2. **Policy Inference**

   * Agents are grouped into architecture-compatible buckets.
   * Inference runs through `agent.ensemble.ensemble_forward()`.
   * `torch.func.vmap` may be used when supported and safe; otherwise execution falls back to a canonical loop path.

3. **Environment Update**

   * Actions are applied (movement, combat resolution, zone effects/scoring, respawn logic).
   * Rewards are computed when PPO is enabled.

4. **PPO Integration (Optional)**

   * Rollout elements are recorded (`obs`, logits, actions, rewards, dones, masks, values, etc.).
   * After a configured rollout window, per-agent PPO performs GAE/advantage estimation and optimization updates.

5. **Telemetry + Persistence + Checkpointing**

   * Stats/events are queued and written asynchronously to avoid stalling the simulation loop.
   * Checkpoints are written atomically (temp directory + completion marker + finalization/replace).

---

## Runtime Modes

The framework supports multiple runtime modes with different operational behavior:

### 1) Headless Mode (default)

* Highest throughput path
* Preferred for long runs, remote execution, and benchmarking
* Produces normal run artifacts (results, telemetry, checkpoints) unless disabled by config

### 2) UI Viewer Mode (`FWS_UI=1`)

* Pygame-based interactive visualization and debugging
* Useful for behavior inspection
* Adds rendering overhead and GPU→CPU synchronization costs (not suitable for throughput benchmarking)

### 3) Inspector No-Output Mode (`FWS_INSPECTOR_MODE=ui_no_output`)

* Viewer-oriented inspection path without creating standard run artifacts
* Avoids normal results/checkpoint/telemetry directory creation
* Useful for visual debugging without contaminating experiment output folders

---

## Configuration Model (High-Level)

Configuration is driven by environment variables prefixed with **`FWS_`**, with profile support (e.g., `default`, `debug`, `train_fast`, `train_quality`).

Major configuration domains include:

* **Run / Resume**

  * seed, results directory, checkpoint path
  * output continuity on resume
  * strict append-schema checks for resumed CSV workflows

* **Simulation**

  * grid dimensions, start population, max agents
  * spawn mode
  * map / zone configuration

* **Model / Inference**

  * brain family selection
  * action-space size and observation layout
  * CUDA / AMP toggles
  * `vmap` toggles and thresholds

* **PPO**

  * enable flag
  * rollout window
  * learning rate, epochs, minibatches
  * entropy/KL-related training controls

* **Telemetry / UI / Recording**

  * telemetry enablement and granularity
  * rich PPO diagnostics
  * UI and inspector mode behavior
  * optional video recording

* **Checkpointing**

  * checkpoint cadence
  * on-exit checkpointing
  * retention / pruning policy

`config.py` performs validation and emits a compact startup summary banner for quick run verification.

---

## Inputs, Outputs, and Artifacts

### Inputs

The framework does **not require an external dataset** for baseline simulation runs.

Primary inputs are:

* `FWS_*` environment configuration
* optional checkpoint path for resume
* procedurally generated map/world state
* runtime spawn/respawn logic and agent populations

### Outputs (Typical Non-Inspector Runs)

A run typically creates a timestamped results directory containing some or all of the following (depending on configuration and runtime mode):

* core run metadata and summaries (`config.json`, `summary.json`)
* primary CSV logs (`stats.csv`, `dead_agents_log.csv`)
* `checkpoints/` with atomic checkpoint folders (`checkpoint.pt`, `manifest.json`, `DONE`, optional `PINNED`, `latest.txt`)
* `telemetry/` CSVs and event JSONL streams
* optional rich PPO diagnostics (`ppo_training_telemetry.csv`)
* optional video/frame outputs and lineage analysis artifacts (when enabled)

These artifacts support post-run analysis of:

* learning dynamics,
* agent lifecycle and lineage,
* simulation behavior,
* runtime stability/performance.

---

## Training and Evaluation Posture

### PPO Training Model

The framework supports **per-agent PPO** (**no parameter sharing**), enabling behavioral diversity and lineage-level experimentation at the cost of higher memory and compute overhead.

### Evaluation State

The repository emphasizes **telemetry and artifact generation** over a dedicated standalone evaluation pipeline. In practice, users evaluate runs by inspecting CSV/JSONL outputs and building custom analysis notebooks/scripts.

### Benchmarking Guidance (Practical)

Throughput and learning behavior should be measured under controlled conditions, including:

* headless vs UI mode
* telemetry/video enabled vs disabled
* grid size and population size
* device + AMP settings
* fresh vs resumed runs
* config/profile parity across comparisons

---

## Key Design Trade-offs

### Per-Agent PPO vs Shared Policy

* **Pros:** behavioral diversity, lineage dynamics, heterogeneous adaptation
* **Cons:** linear growth in optimizer/state overhead and higher memory/compute cost

### Dual State Representation (Grid + Registry)

* **Pros:** efficient vectorized updates and spatial queries with structured agent state
* **Cons:** requires strict synchronization invariants and careful debugging

### Environment-Variable Configuration (`FWS_*`)

* **Pros:** easy scripting and reproducible shell-based launches
* **Cons:** lower discoverability and weaker typing than a first-class CLI/config schema

### Background Telemetry/Persistence Writer

* **Pros:** reduces I/O stalls in the hot simulation loop
* **Cons:** queue pressure and durability behavior require operational tuning and clear expectations

### Append-Safe Resume Continuity

* **Pros:** preserves a single artifact lineage across interrupted/resumed runs
* **Cons:** depends on schema stability (or explicit schema migration/versioning)

---

## Current Limitations and Maturity Gaps

* No pinned dependency manifest observed (e.g., `requirements.txt`, `pyproject.toml`, lockfile)
* No visible automated test/CI coverage in the summarized snapshot
* Per-agent PPO scaling costs grow with population size
* UI mode is unsuitable for throughput benchmarking
* Telemetry schema evolution can break append-resume workflows without migration support
* No first-class evaluation/analysis scripts for benchmark and learning-curve reporting

---

## Summary Assessment

Neural Siege is a **systems-heavy, feature-rich MARL simulation framework** with unusually strong operational discipline for a research prototype (checkpoint safety, resume continuity, append-safe telemetry, and artifact hygiene).

It is well-positioned for experimentation in:

* multi-agent reinforcement learning,
* evolutionary policy dynamics,
* policy architecture comparisons (Transformer / Tron / Mirror variants),
* observability-aware simulation systems research.

### Highest-Leverage Improvements (Next)

To mature further as a research platform, the most valuable next steps are:

1. **Dependency pinning** (canonical environment spec / lockfile)
2. **Automated regression tests** (especially checkpoint-resume and telemetry schema stability)
3. **Evaluation/analysis tooling** (reproducible reporting scripts/notebooks)
4. **Benchmark baselines** (throughput + learning telemetry under controlled configs)
5. **Telemetry schema versioning / migration support** for long-lived resume workflows
