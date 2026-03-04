# Neural Siege (`Infinite_War_Simulation`)

High-throughput, GPU-accelerated multi-agent grid combat simulation with **per-agent PPO training**, **transformer-family agent policies**, **append-safe telemetry**, and **atomic checkpoint/resume workflows**.

> **Source note:** This README is based on an aggregated Python-source snapshot (37 `.py` files). Non-Python assets (e.g., `LICENSE`, dependency lockfiles, CI configs) may be absent from the snapshot and are marked with TODOs where relevant.

[![Project Status](https://img.shields.io/badge/status-research%20prototype-informational)](#)
[![Python](https://img.shields.io/badge/python-TODO-blue)](#)
[![PyTorch](https://img.shields.io/badge/pytorch-TODO-red)](#)
[![CUDA](https://img.shields.io/badge/cuda-optional-76B900)](#)
[![License](https://img.shields.io/badge/license-TODO-lightgrey)](#)

## Key Highlights

* **Vectorized tick engine** using PyTorch tensors for large-scale grid combat simulation with synchronized **grid + agent registry** state representations.
* **Multiple policy architectures** (`TronBrain`, `MirrorBrain`, `TransformerBrain`) with bucketed batched inference via `agent.ensemble.ensemble_forward()`.
* **Reliability-first `vmap` acceleration** with safe fallback to canonical loop execution when `torch.func`/TorchScript compatibility is not satisfied.
* **Per-agent PPO runtime (no parameter sharing)** with slot-local optimizers/schedulers, rollout buffers, checkpointable state, and action-mask-aware updates.
* **Operational robustness**: atomic checkpoints (`DONE` marker), `latest.txt`, retention pruning, resume continuity, crash traces, and on-exit checkpointing.
* **Append-friendly telemetry** with CSV/JSONL event streams, lineage artifacts, counters/summaries, and **rich PPO diagnostics CSV**.
* **Dual runtime modes**: headless throughput mode, interactive Pygame viewer, and **inspector no-output mode** for visualization without run artifacts.

## Table of Contents

* [Overview](#overview)
* [Repository Contents](#repository-contents)
* [Architecture](#architecture)
* [Runtime Pipeline](#runtime-pipeline)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
* [Inputs Outputs and Artifacts](#inputs-outputs-and-artifacts)
* [Training and Execution](#training-and-execution)
* [Evaluation and Validation](#evaluation-and-validation)
* [Results and Benchmarks](#results-and-benchmarks)
* [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
* [Limitations and Known Issues](#limitations-and-known-issues)
* [Troubleshooting](#troubleshooting)
* [Reproducibility and Determinism](#reproducibility-and-determinism)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Overview

**Neural Siege** is a research-oriented, grid-based multi-agent combat simulation designed to support both:

1. **High-throughput headless simulation** (CPU/GPU, tensorized execution), and
2. **Online reinforcement learning** via an integrated **per-agent PPO runtime**.

The project is optimized for long-running experiments and operational safety:

* resumable checkpoints,
* append-safe telemetry,
* crash hygiene,
* deterministic seeding,
* configurable runtime modes.

This makes it suitable for experiments in:

* multi-agent reinforcement learning (MARL),
* heterogeneous policy populations,
* evolution/lineage dynamics,
* systems-level performance vs observability trade-offs.

## Repository Contents

| Area                     | Purpose                                 | Notes                                                                                     |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------- |
| `main.py`                | Runtime orchestration                   | Startup, fresh/resume flows, UI/headless selection, shutdown, summaries, checkpoint hooks |
| `config.py`              | `FWS_*` configuration + validation      | Profiles, defaults, runtime summary banner, sanity checks                                 |
| `engine/`                | Core simulation runtime                 | Grid, registry, tick loop, map generation, raycasting, move masking, spawn/respawn        |
| `agent/`                 | Policy architectures + inference fusion | `TransformerBrain`, `TronBrain`, `MirrorBrain`, `obs_spec`, `ensemble_forward`            |
| `rl/ppo_runtime.py`      | Per-agent PPO runtime                   | Rollouts, GAE/updates, optimizer state, checkpointable PPO state, diagnostics queue       |
| `utils/telemetry.py`     | Structured telemetry subsystem          | CSV/JSONL events, counters, lineage, summaries, rich PPO telemetry                        |
| `utils/persistence.py`   | Background results writer               | Queue-based logging for core CSV streams (`stats.csv`, death logs)                        |
| `utils/checkpointing.py` | Atomic checkpoint save/load/resume      | `DONE` marker, manifests, `latest.txt`, pruning, RNG restore                              |
| `ui/`                    | Interactive Pygame viewer               | Visualization and inspection paths                                                        |
| `recorder/`              | Optional recording/schema utilities     | Video/frame output + optional Arrow schema helpers                                        |
| `lineage_tree.py`        | Offline lineage visualization utility   | Plotly-based lineage analysis workflow                                                    |
| `dump_py_to_text.py`     | Developer utility                       | Aggregates Python sources into a single snapshot text file                                |

## Architecture

```mermaid
flowchart TD
    A[config.py / FWS_*] --> B[main.py]

    B --> C[TickEngine]
    B --> D[CheckpointManager]
    B --> E[ResultsWriter]
    B --> F[TelemetrySession]
    B --> G[Viewer (optional)]
    B --> H[Recorder (optional)]

    C --> I[AgentsRegistry]
    C --> J[Grid Tensor]
    C --> K[Mapgen / Zones / Spawn / Respawn]
    C --> L[Ray Engine + Move Mask]
    C --> M[SimulationStats]

    C --> N[Bucketed inference via ensemble_forward]
    N --> O[TronBrain]
    N --> P[MirrorBrain]
    N --> Q[TransformerBrain]

    C --> R[PerAgentPPORuntime (optional)]
    R --> F
    R --> I

    D --> C
    D --> I
    D --> M

    F --> S[telemetry/*.csv + events/*.jsonl]
    E --> T[stats.csv / dead_agents_log.csv / config.json]
    D --> U[checkpoints/ckpt_t*/]
    H --> V[video / frames outputs]
```

### Architecture Notes

* The engine maintains **two synchronized state views**: a **grid tensor** and an **agent registry tensor store**. Consistency between them is a core invariant.
* Policy inference is **bucketed by compatible brain architecture** and executed through `ensemble_forward`, which attempts `torch.func.vmap` when safe and falls back to a loop when needed.
* PPO is integrated into the tick lifecycle and receives rollout elements from runtime execution when enabled.
* Telemetry and persistence are decoupled from the hot path to reduce disk I/O stalls.

## Runtime Pipeline

```mermaid
flowchart TD
    A[Set env vars / profile] --> B[python main.py]
    B --> C{Checkpoint path provided?}

    C -- No --> D[Fresh init\nGrid + map + zones + spawn]
    C -- Yes --> E[Load checkpoint\nRestore world + RNG + PPO state]

    D --> F[Create TickEngine]
    E --> F

    F --> G{Inspector no-output mode?}
    G -- Yes --> H[Viewer-only path\nNo artifacts]
    G -- No --> I[Create run_dir + ResultsWriter]

    I --> J[CheckpointManager]
    J --> K[TelemetrySession]

    K --> L{UI enabled?}
    L -- Yes --> M[Viewer.run()]
    L -- No --> N[Headless loop]

    M --> O[engine.run_tick()]
    N --> O

    O --> P[Observe + raycast + action masks]
    P --> Q[Bucketed policy inference]
    Q --> R[Combat / movement / zones / rewards]
    R --> S[PPO record/update (optional)]
    S --> T[Periodic telemetry + checkpoints]

    T --> U[Shutdown / interrupt / crash]
    U --> V[On-exit checkpoint (optional)]
    U --> W[Flush summaries + close writers]
```

### Runtime Modes

| Mode                | Trigger                                      | Intended use                                                      | Artifacts |
| ------------------- | -------------------------------------------- | ----------------------------------------------------------------- | --------- |
| Headless (default)  | `FWS_UI=0`                                   | Throughput / long runs / remote execution                         | Yes       |
| UI Viewer           | `FWS_UI=1`                                   | Interactive debugging / visualization                             | Yes       |
| Inspector No-Output | `FWS_INSPECTOR_MODE=ui_no_output` (or alias) | Visual inspection without modifying results/checkpoints/telemetry | No        |

## Repository Structure

```text
Infinite_War_Simulation/
├── agent/
│   ├── __init__.py
│   ├── ensemble.py
│   ├── mirror_brain.py
│   ├── obs_spec.py
│   ├── transformer_brain.py
│   └── tron_brain.py
├── engine/
│   ├── game/
│   │   └── move_mask.py
│   ├── ray_engine/
│   │   ├── raycast_32.py
│   │   ├── raycast_64.py
│   │   └── raycast_firsthit.py
│   ├── __init__.py
│   ├── agent_registry.py
│   ├── grid.py
│   ├── mapgen.py
│   ├── respawn.py
│   ├── spawn.py
│   └── tick.py
├── recorder/
│   ├── __init__.py
│   ├── recorder.py
│   ├── schemas.py
│   └── video_writer.py
├── rl/
│   ├── __init__.py
│   └── ppo_runtime.py
├── simulation/
│   └── stats.py
├── ui/
│   ├── __init__.py
│   ├── camera.py
│   └── viewer.py
├── utils/
│   ├── checkpointing.py
│   ├── persistence.py
│   ├── profiler.py
│   ├── sanitize.py
│   └── telemetry.py
├── __init__.py
├── config.py
├── dump_py_to_text.py
├── lineage_tree.py
└── main.py
```

## Installation

> **Observed snapshot state:** No dependency manifest (`requirements.txt`, `pyproject.toml`, `environment.yml`) was included. The commands below are a safe starting point and should be replaced by a pinned environment spec.

### Base environment (minimal runtime)

```bash
# Assumption: run from repo root containing the Infinite_War_Simulation folder
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install numpy torch
```

### Optional dependencies (UI / video / analysis)

```bash
# UI viewer
pip install pygame-ce

# Video/frame IO (optional)
pip install opencv-python imageio

# Analytics / lineage / schema tooling (optional)
pip install plotly pyarrow
```

### Smoke test

```bash
cd Infinite_War_Simulation
python -c "import config; print(config.summary_str())"
```

## Quickstart

### 1) Headless run (default)

```bash
cd Infinite_War_Simulation
python main.py
```

### 2) UI viewer mode

```bash
cd Infinite_War_Simulation
# Linux/macOS
FWS_UI=1 python main.py

# Windows PowerShell
# $env:FWS_UI='1'; python main.py
```

### 3) Resume from checkpoint (resume continuity enabled by default)

```bash
cd Infinite_War_Simulation
FWS_CHECKPOINT_PATH="results/sim_YYYY-MM-DD_HH-MM-SS/checkpoints/ckpt_tXXXXXX_YYYY-MM-DD_HH-MM-SS" python main.py
```

### 4) Inspector no-output mode (visualize without creating artifacts)

```bash
cd Infinite_War_Simulation
FWS_UI=1 FWS_INSPECTOR_MODE=ui_no_output python main.py
```

## Configuration

Neural Siege is **environment-variable driven**. Most runtime knobs live in `config.py` and are exposed as `FWS_*` variables.

### High-signal configuration subset (observed)

| Category   | Config Variable                   | Env Var                               |      Default | Purpose                                                               |
| ---------- | --------------------------------- | ------------------------------------- | -----------: | --------------------------------------------------------------------- |
| Run        | `PROFILE`                         | `FWS_PROFILE`                         |    `default` | Profile overrides (`default`, `debug`, `train_fast`, `train_quality`) |
| Run        | `RNG_SEED`                        | `FWS_SEED`                            |         `42` | Deterministic startup seed                                            |
| Run        | `RESULTS_DIR`                     | `FWS_RESULTS_DIR`                     |    `results` | Base output directory                                                 |
| Resume     | `CHECKPOINT_PATH`                 | `FWS_CHECKPOINT_PATH`                 |         `""` | Resume source (empty = fresh run)                                     |
| Resume     | `RESUME_OUTPUT_CONTINUITY`        | `FWS_RESUME_OUTPUT_CONTINUITY`        |       `True` | Append into original run folder on resume                             |
| Resume     | `RESUME_APPEND_STRICT_CSV_SCHEMA` | `FWS_RESUME_APPEND_STRICT_CSV_SCHEMA` |       `True` | Fail on CSV header/schema mismatch during append                      |
| Checkpoint | `CHECKPOINT_EVERY_TICKS`          | `FWS_CHECKPOINT_EVERY_TICKS`          |      `50000` | Periodic checkpoint cadence (`0` disables)                            |
| Checkpoint | `CHECKPOINT_ON_EXIT`              | `FWS_CHECKPOINT_ON_EXIT`              |       `True` | Save checkpoint during clean shutdown                                 |
| Runtime    | `USE_CUDA` / `DEVICE`             | `FWS_CUDA`                            |         auto | CUDA enable / device selection                                        |
| Runtime    | `AMP_ENABLED`                     | `FWS_AMP`                             |       `True` | Mixed precision (CUDA path)                                           |
| Runtime    | `USE_VMAP`                        | `FWS_USE_VMAP`                        |       `True` | Enable `torch.func.vmap` path in bucketed inference                   |
| Runtime    | `VMAP_MIN_BUCKET`                 | `FWS_VMAP_MIN_BUCKET`                 |          `8` | Minimum bucket size before attempting `vmap`                          |
| Sim        | `GRID_WIDTH`, `GRID_HEIGHT`       | `FWS_GRID_W`, `FWS_GRID_H`            | `100`, `100` | Grid dimensions                                                       |
| Sim        | `START_AGENTS_PER_TEAM`           | `FWS_START_PER_TEAM`                  |        `300` | Initial population per team                                           |
| Sim        | `MAX_AGENTS`                      | `FWS_MAX_AGENTS`                      |        `700` | Registry capacity                                                     |
| Sim        | `SPAWN_MODE`                      | `FWS_SPAWN_MODE`                      |    `uniform` | Spawn strategy (`uniform` / `symmetric`)                              |
| Model      | `BRAIN_KIND`                      | `FWS_BRAIN`                           |       `tron` | Primary brain family                                                  |
| Model      | `NUM_ACTIONS`                     | `FWS_NUM_ACTIONS`                     |         `41` | Discrete action space size                                            |
| PPO        | `PPO_ENABLED`                     | `FWS_PPO_ENABLED`                     |       `True` | Enable per-agent PPO runtime                                          |
| PPO        | `PPO_WINDOW_TICKS`                | `FWS_PPO_TICKS`                       |        `512` | Rollout window length                                                 |
| PPO        | `PPO_LR`                          | `FWS_PPO_LR`                          |       `3e-4` | Learning rate                                                         |
| PPO        | `PPO_EPOCHS`                      | `FWS_PPO_EPOCHS`                      |          `4` | PPO epochs per update                                                 |
| PPO        | `PPO_MINIBATCHES`                 | `FWS_PPO_MINIB`                       |          `8` | Minibatches per update                                                |
| Telemetry  | `TELEMETRY_ENABLED`               | `FWS_TELEMETRY`                       |       `True` | Global telemetry switch                                               |
| Telemetry  | `TELEMETRY_PPO_RICH_CSV`          | `FWS_TELEM_PPO_RICH_CSV`              |       `True` | Rich PPO diagnostics CSV                                              |
| UI         | `ENABLE_UI`                       | `FWS_UI`                              |      `False` | Toggle Pygame viewer                                                  |
| UI         | `INSPECTOR_MODE`                  | `FWS_INSPECTOR_MODE`                  |        `off` | Inspector mode behavior                                               |
| UI         | `RECORD_VIDEO`                    | `FWS_RECORD_VIDEO`                    |      `False` | Optional video recording                                              |

### Observation layout (current snapshot)

```text
OBS_DIM = RAY_TOKEN_COUNT * RAY_FEAT_DIM + (RICH_BASE_DIM + INSTINCT_DIM)
        = 32 * 8 + (23 + 4)
        = 283
```

## Inputs Outputs and Artifacts

### Inputs

Neural Siege does **not require an external dataset** for baseline runs. Primary inputs are:

* `FWS_*` runtime configuration variables
* optional checkpoint path (`FWS_CHECKPOINT_PATH`)
* procedurally generated world/map state
* runtime-created agent populations and brain instances

### Core runtime data flow (conceptual)

* Tick engine builds **observations** and **action masks** for alive agents.
* Bucketed inference produces action logits and values (when PPO is enabled).
* Actions are sampled and applied to the environment (movement/combat/zones/respawn).
* PPO runtime records trajectory fragments and performs periodic updates.
* Telemetry and persistence subsystems emit structured artifacts for analysis.

### Typical output artifacts (non-inspector mode)

```text
results/
└── sim_YYYY-MM-DD_HH-MM-SS/
    ├── config.json
    ├── stats.csv
    ├── dead_agents_log.csv
    ├── summary.json
    ├── crash_trace.txt                  # crash path only
    ├── checkpoints/
    │   ├── latest.txt
    │   └── ckpt_t..._YYYY-MM-DD_HH-MM-SS/
    │       ├── checkpoint.pt
    │       ├── manifest.json
    │       ├── DONE
    │       └── PINNED                   # optional
    ├── telemetry/
    │   ├── run_meta.json
    │   ├── agent_life.csv
    │   ├── lineage_edges.csv
    │   ├── agent_static.csv
    │   ├── tick_summary.csv
    │   ├── move_summary.csv
    │   ├── counters.csv
    │   ├── telemetry_summary.csv
    │   ├── ppo_training_telemetry.csv   # optional / config-gated
    │   ├── mutation_events.csv
    │   └── events/
    │       └── events_*.jsonl[.gz]
    └── video.* / frames_*               # optional recorder outputs
```

> Exact artifact presence depends on runtime mode and config toggles (telemetry, UI, video, reporting).

## Training and Execution

There is no separate build step in the observed snapshot; execution is Python-driven.

### PPO-enabled training lifecycle (high level)

* Tick engine computes observations, masks, and actions for alive agents.
* Rollout elements are recorded into the per-agent PPO runtime:

  * observations,
  * masked logits,
  * values,
  * actions,
  * rewards,
  * dones,
  * masks,
  * bootstrap values.
* PPO runtime performs **independent per-agent updates** (no parameter sharing).
* Rich PPO diagnostic rows can be flushed into `telemetry/ppo_training_telemetry.csv`.

### Checkpointing and resume behavior

* **Atomic checkpoint writes** via temporary directory + atomic replace/rename
* Completion guarded by **`DONE` marker**
* `latest.txt` maintained for convenient “resume latest” workflows
* Resume path supports:

  * checkpoint directory,
  * direct `checkpoint.pt`,
  * checkpoints root containing `latest.txt`
* **Resume output continuity** (default `True`) appends into the original run directory instead of fragmenting artifacts across multiple runs

### Suggested run patterns

```bash
# Throughput-oriented headless run
cd Infinite_War_Simulation
FWS_UI=0 FWS_TELEMETRY=1 FWS_CHECKPOINT_EVERY_TICKS=50000 python main.py

# Visual debug run (reduced scale)
FWS_UI=1 FWS_GRID_W=64 FWS_GRID_H=64 FWS_START_PER_TEAM=80 python main.py

# Resume in-place with strict CSV append checks
FWS_CHECKPOINT_PATH="results/.../checkpoints/ckpt_t..._..." \
FWS_RESUME_OUTPUT_CONTINUITY=1 \
FWS_RESUME_APPEND_STRICT_CSV_SCHEMA=1 \
python main.py
```

## Evaluation and Validation

No standalone `eval.py` script was observed in the source snapshot. Validation is primarily performed through **runtime checks**, **telemetry artifacts**, and **post-run analysis**.

### Built-in safeguards (observed)

* `config.py` validation (range/sanity/profile checks)
* shape/device assertions in PPO rollout and update paths
* observation schema checks in `agent/obs_spec.py` and `MirrorBrain`
* checkpoint completeness guard (`DONE` marker required)
* optional strict CSV append schema checks for resume continuity
* best-effort telemetry flush/close behavior on shutdown

### Practical evaluation workflow

1. Confirm startup banner (`config.summary_str()`) matches intended experiment config.
2. Verify `summary.json` and `telemetry/run_meta.json` (seed/device/resume/config snapshot).
3. Inspect `telemetry/ppo_training_telemetry.csv` for learning dynamics (KL, entropy, clip fraction, explained variance, gradient norms).
4. Compare `stats.csv` and telemetry summaries across controlled runs.
5. Use `lineage_tree.py` for lineage-focused analysis from telemetry lineage files.

## Results and Benchmarks

> **No benchmark numbers are included here because none were provided in the source snapshot.** Fill the template below with measured runs.

### Benchmark template

| Run ID | Date         | Profile      | Brain    | Grid      | Start/Team | Device | AMP  | UI    | Avg TPS | Final Tick | PPO   | Resume   | Notes  |
| ------ | ------------ | ------------ | -------- | --------- | ---------: | ------ | ---- | ----- | ------: | ---------: | ----- | -------- | ------ |
| `TODO` | `YYYY-MM-DD` | `default`    | `tron`   | `100x100` |      `300` | `cuda` | `on` | `off` |  `TODO` |     `TODO` | `yes` | `fresh`  | `TODO` |
| `TODO` | `YYYY-MM-DD` | `train_fast` | `mirror` | `100x100` |      `300` | `cuda` | `on` | `off` |  `TODO` |     `TODO` | `yes` | `resume` | `TODO` |

### Benchmark interpretation notes

* Compare runs only when core config and hardware are controlled.
* UI, rich telemetry, and video recording materially affect throughput.
* For resumed runs, record whether metrics are cumulative and whether output continuity was enabled.

## Design Decisions and Trade-offs

### Per-agent PPO (no parameter sharing)

* **Why:** maximize behavioral diversity and agent-level autonomy.
* **Trade-off:** high memory/optimizer-state overhead as population scales.

### Dual state representation (grid + registry)

* **Why:** efficient vectorized spatial updates plus structured per-agent attributes.
* **Trade-off:** strict synchronization invariants are required to prevent subtle state drift.

### Environment-variable configuration (`FWS_*`)

* **Why:** reproducible shell-based experiment launches and profile overrides.
* **Trade-off:** weaker discoverability and typing than a first-class CLI/config schema.

### Background writer process + append-safe resume continuity

* **Why:** preserve throughput and support long-run artifact continuity.
* **Trade-off:** schema stability matters; append workflows can fail fast after schema changes (by design in strict mode).

### Reliability-first `vmap` path

* **Why:** performance improvement when safe, correctness preserved via fallback.
* **Trade-off:** mixed execution paths can complicate profiling/debugging across environments.

## Limitations and Known Issues

* No dependency manifest was present in the snapshot (`requirements.txt` / `pyproject.toml` / lockfile missing).
* No automated test suite / CI config was observed in the snapshot.
* Per-agent PPO scales linearly in model + optimizer state cost.
* UI mode introduces GPU→CPU synchronization overhead and should not be used for throughput benchmarking.
* Telemetry append continuity requires stable schemas across code versions.
* Action/observation schema coupling across config/engine/agent modules increases change risk.
* Optional dependency surface (Pygame, OpenCV, ImageIO, Plotly, PyArrow) increases environment variability.

## Troubleshooting

### High-frequency issues

* **Checkpoint load refused**: verify checkpoint directory contains both `checkpoint.pt` and `DONE`.
* **Resume created a new run folder unexpectedly**: check `FWS_RESUME_OUTPUT_CONTINUITY` and `FWS_RESUME_FORCE_NEW_RUN`.
* **CSV append resume failed**: schema changed; either migrate artifacts or start a fresh run (`FWS_RESUME_FORCE_NEW_RUN=1`).
* **UI is slow / stuttering**: reduce grid/population, disable video, or use headless mode for throughput runs.
* **No telemetry files generated**: confirm `FWS_TELEMETRY=1` and ensure inspector no-output mode is not active.
* **Optional import errors (`pygame`, `cv2`, `imageio`, `pyarrow`, `plotly`)**: install only the extras needed for your selected workflow.

### Useful diagnostics

```bash
cd Infinite_War_Simulation

# Print compact runtime summary
python -c "import config; print(config.summary_str())"

# Resolve checkpoint path before resume (examples)
python - <<'PY'
from utils.checkpointing import resolve_checkpoint_path
print(resolve_checkpoint_path(r'results/.../checkpoints'))
PY
```

## Reproducibility and Determinism

### What is implemented (observed)

* Deterministic startup seeding via `FWS_SEED` / `RNG_SEED`
* Checkpoint save/load captures and restores RNG state (Python / NumPy / PyTorch; CUDA where available)
* Atomic checkpoint writes with completion markers
* Best-effort summary/telemetry shutdown paths

### Determinism caveats

* CUDA kernels and mixed precision may still introduce nondeterminism depending on hardware, drivers, and backend behavior.
* UI mode and interrupt timing can affect runtime timing and event ordering.
* Optional telemetry/video settings can alter runtime timing enough to affect learning trajectories.

### Recommended experiment metadata to record

* exact `FWS_*` overrides
* Git commit SHA
* Python / PyTorch / CUDA versions
* GPU model + driver
* whether resume continuity was used

## Contributing

Contributions should preserve the project’s **reliability-first** behavior (checkpoint safety, telemetry robustness, and reproducibility-minded workflows).

### Minimum expectations

* Keep changes surgical, especially in performance-critical modules.
* Preserve or explicitly migrate artifact schemas (CSV/JSONL/checkpoint manifests).
* Document new `FWS_*` knobs.
* Validate both fresh-start and resume flows when touching runtime state, checkpointing, or telemetry.
* Note reproducibility/performance implications in PR descriptions when relevant.

### Recommended PR checklist

* [ ] Fresh run works
* [ ] Resume from checkpoint works
* [ ] Telemetry outputs remain readable / append-safe
* [ ] No obvious headless throughput regression
* [ ] README / config docs updated

## Citation

If you use this repository in research or internal experimentation, cite the codebase version used.

```bibtex
@misc{neural_siege_infinite_war_simulation,
  title        = {{Neural Siege (Infinite_War_Simulation)}},
  author       = {{TODO: Add author / team name}},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {TODO: Add repository URL, commit SHA, and release tag}
}
```

## License

**TODO: Add license information.**
No license file was present in the Python-source snapshot used to build this README.

## Acknowledgements

This project builds on a practical stack commonly used in simulation and ML systems, including:

* PyTorch (tensor compute / GPU acceleration)
* NumPy (numeric utilities)
* Pygame (interactive viewer)
* OpenCV / ImageIO (optional video/frame outputs)
* PyArrow (optional typed schema tooling)
* Plotly (optional lineage visualization)
