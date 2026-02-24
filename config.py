"""
config.py (or similar): Central Configuration + Environment Overrides

What this file is
-----------------
This module defines *almost all* tunable parameters for a large-scale multi-agent
reinforcement learning simulation (your "Final War Sim" style world). It is designed
to be:

1) **Reproducible**: a single file defines the defaults for a run.
2) **Scriptable**: every important knob can be overridden using environment variables,
   so you can do hyperparameter sweeps from Bash / PowerShell / SLURM without editing code.
3) **Stable contracts**: some constants (like observation layout dimensions) are treated
   as "ABI-like" interfaces with the neural network. Changing those breaks checkpoints.

Key pattern
-----------
- Define safe environment parsers (_env_bool/_env_int/_env_float/_env_str).
- Define defaults (pure Python constants).
- Allow environment variables to override defaults.
- Apply "profiles" (preset bundles) at the end as optional macro-overrides.
- Provide a summary_str() to print a one-line run signature for logs.

Important: "Don't change any code"
----------------------------------
Per your instruction, the *logic and values* below are preserved.
I am only adding professional, beginner-friendly explanation comments and structure docs.

Engineering philosophy (hard facts)
-----------------------------------
- Environment-driven configs are standard in production ML systems because:
  * they reduce code churn
  * allow orchestration systems to set parameters at runtime
  * improve reproducibility by logging env var values
- Keeping observation/action dimensions stable is critical because:
  * neural network weights depend on exact input/output shapes
  * checkpoints are shape-dependent
- Many defaults below are chosen to balance:
  * simulation realism vs. compute cost
  * stability of PPO training vs. exploration
  * UI smoothness vs. GPU->CPU transfer overhead
"""

from __future__ import annotations

import os
import math
import torch

# =============================================================================
# 🛠️ ENV-PARSING UTILITIES (PRODUCTION-GRADE SECRETS)
# =============================================================================
# These helper functions safely parse environment variables.
#
# Why do this?
# ------------
# In research/engineering workflows, you often want to run:
#   - parameter sweeps
#   - ablation studies
#   - multiple seeds
#   - different map sizes
# ...without editing code.
#
# A typical pattern is:
#   FWS_MAX_AGENTS=2000 FWS_SEED=0 python main.py
#
# This file defines typed parsers so:
# - invalid values do NOT crash the run (they fall back to defaults)
# - code remains clean (no repeated try/except everywhere)
# - there is a single source of truth for defaults
#
# NOTE: These parsers are intentionally permissive (fall back silently).
# In strict production systems, you may prefer hard failures for invalid values.
# Here, permissive behavior helps long experiments keep running.

def _env_bool(key: str, default: bool) -> bool:
    """
    Read an environment variable and interpret it as boolean.

    Accepted "truthy" strings (case-insensitive):
        "1", "true", "yes", "y", "on"

    If the variable is missing, return the provided default.

    Example:
        FWS_CUDA=1  -> True
        FWS_CUDA=0  -> False (not in truthy set)
    """
    v = os.getenv(key)
    if v is None: return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_float(key: str, default: float) -> float:
    """
    Read an environment variable as float.
    If missing or invalid, return default.

    Why float parsing matters:
    - RL hyperparameters (LR, entropy, gamma) are floats.
    - Map probabilities and rates are floats.
    """
    v = os.getenv(key)
    if v is None: return float(default)
    try: return float(v)
    except Exception: return float(default)

def _env_int(key: str, default: int) -> int:
    """
    Read an environment variable as int.
    If missing or invalid, return default.

    Used for:
    - grid sizes
    - tick frequencies
    - counts (agents, zones, etc.)
    """
    v = os.getenv(key)
    if v is None: return int(default)
    try: return int(v)
    except Exception: return int(default)

def _env_str(key: str, default: str) -> str:
    """
    Read an environment variable as string.
    If missing, return default.

    Note: we do not strip() here because sometimes spacing is meaningful,
    but downstream typically strips where needed.
    """
    v = os.getenv(key)
    return default if v is None else str(v)

def _env_is_set(key: str) -> bool:
    """
    Returns True if the environment variable exists (even if empty),
    otherwise False.

    This is different from checking truthiness:
    - You may want to distinguish between:
        "env var not provided" vs. "env var provided but empty"
    """
    return os.getenv(key) is not None

# =============================================================================
# 🧪 EXPERIMENT TRACKING & SEEDING
# =============================================================================
# Profiles act as master macros to override batches of settings at once.
# Typical use:
#   FWS_PROFILE=debug python main.py
#
# Note: In this codebase you also have a profiler utility using FWS_PROFILE as a boolean.
# Here, FWS_PROFILE is being used as a profile name ("default"/"debug"/...).
# This is a naming collision risk in large projects; one fix is to rename one of them
# (e.g., FWS_RUN_PROFILE vs FWS_TORCH_PROFILE). But per instructions, we keep code unchanged.
PROFILE: str = _env_str("FWS_PROFILE", "default").strip().lower()

# Tag used by ResultsWriter to organize output directories for research logging.
# This usually becomes part of a folder name, experiment tracking key, etc.
EXPERIMENT_TAG: str = _env_str("FWS_EXPERIMENT_TAG", "god_level_run").strip()

# Master Random Seed.
# - 0: "stochastic" run (often used to increase variety / emergent behavior)
# - 42: deterministic-ish (best for debugging)
#
# Engineering note:
# True determinism can still be hard with CUDA due to non-deterministic kernels,
# parallel reductions, etc. But using a fixed seed is still extremely useful.
RNG_SEED: int = _env_int("FWS_SEED", 42)

# Base directory for all results (logs, checkpoints, telemetry, etc.)
RESULTS_DIR: str = _env_str("FWS_RESULTS_DIR", "results").strip()

# Optional: resume from a checkpoint (directory containing DONE+checkpoint.pt,
# or a direct checkpoint.pt path).
CHECKPOINT_PATH: str = _env_str("FWS_CHECKPOINT_PATH", "").strip()

# Autosave: set to >0 to autosave checkpoints while the viewer is running.
AUTOSAVE_EVERY_SEC = int(os.environ.get("FWS_AUTOSAVE_EVERY_SEC", "3600"))

# ----------------------------------------------------------------------
# Tick-based checkpointing (works in BOTH UI and headless)
# ----------------------------------------------------------------------
# These knobs control saving model+world state periodically.
# Benefits:
# - You can resume after crashes/power loss
# - You can analyze intermediate behaviors
# - You can do "time travel debugging" by reproducing a scenario
CHECKPOINT_EVERY_TICKS = _env_int("FWS_CHECKPOINT_EVERY_TICKS", 10000)  # 0 disables
CHECKPOINT_ON_EXIT = _env_bool("FWS_CHECKPOINT_ON_EXIT", True)
CHECKPOINT_KEEP_LAST_N = _env_int("FWS_CHECKPOINT_KEEP_LAST_N", 2)  # keep latest N (non-pinned)
CHECKPOINT_PIN_ON_MANUAL = _env_bool("FWS_CHECKPOINT_PIN_ON_MANUAL", True)
CHECKPOINT_PIN_TAG = _env_str("FWS_CHECKPOINT_PIN_TAG", "manual")
CHECKPOINT_TRIGGER_FILE = _env_str("FWS_CHECKPOINT_TRIGGER_FILE", "checkpoint.now")

# ----------------------------------------------------------------------
# Headless console reporting
# ----------------------------------------------------------------------
# When UI is off, you still need observability.
# This defines how frequently you print stats and how detailed they are.
HEADLESS_PRINT_EVERY_TICKS = _env_int("FWS_HEADLESS_PRINT_EVERY_TICKS", 100)  # 0 disables
HEADLESS_PRINT_LEVEL = _env_int("FWS_HEADLESS_PRINT_LEVEL", 2)               # 0=min,1=std,2=detail
HEADLESS_PRINT_GPU = _env_bool("FWS_HEADLESS_PRINT_GPU", True)

# =============================================================================
# 📡 SCIENTIFIC RECORDING / TELEMETRY (NON-INVASIVE, CONFIG-FIRST)
# =============================================================================
# Telemetry is OPTIONAL and must not change simulation outcomes.
# It writes append-only event logs + periodic snapshots for later analysis.
#
# "Append-only" + chunk files are standard for:
# - crash safety (partial run still yields usable logs)
# - avoiding corruption (atomic writes)
# - large-scale data handling (streaming, batching)
TELEMETRY_ENABLED: bool = _env_bool("FWS_TELEMETRY", True)

# Optional extra tag to distinguish telemetry-heavy runs from normal runs.
TELEMETRY_TAG: str = _env_str("FWS_TELEMETRY_TAG", "").strip()

# Versioning schemas is critical: once you have logs in the wild, you need to
# know how to parse them.
TELEMETRY_SCHEMA_VERSION: str = _env_str("FWS_TELEM_SCHEMA", "2").strip()

# Meta logs: run-level metadata and agent statics
TELEMETRY_WRITE_RUN_META: bool = _env_bool("FWS_TELEM_RUN_META", True)            # tiny + safe
TELEMETRY_WRITE_AGENT_STATIC: bool = _env_bool("FWS_TELEM_AGENT_STATIC", True)   # can grow; opt-in

# A tick summary is usually a compact per-tick aggregate table.
TELEMETRY_TICK_SUMMARY_EVERY: int = _env_int("FWS_TELEM_TICK_SUMMARY_EVERY", 100)  # 0=off (opt-in)

# --- Frequencies (ticks) ---
# These are essentially "sampling rates". Higher frequency = more detail but more disk I/O.
TELEMETRY_TICK_METRICS_EVERY: int = _env_int("FWS_TELEM_TICK_EVERY", 10)          # per-tick summary rows
TELEMETRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_SNAPSHOT_EVERY", 10)          # grid occupancy snapshots
TELEMETRY_REGISTRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_REG_EVERY", 10)      # registry snapshots
TELEMETRY_VALIDATE_EVERY: int = _env_int("FWS_TELEM_VALIDATE_EVERY", 50)          # invariant checks
TELEMETRY_PERIODIC_FLUSH_EVERY: int = _env_int("FWS_TELEM_FLUSH_EVERY", 200)      # force flush

# --- Buffers / chunk sizes (written as chunk files for atomicity) ---
# Chunking helps performance because writing 1 line at a time is slow.
TELEMETRY_EVENT_CHUNK_SIZE: int = _env_int("FWS_TELEM_EVENT_CHUNK", 50000)
TELEMETRY_TICK_CHUNK_SIZE: int = _env_int("FWS_TELEM_TICK_CHUNK", 5000)

# --- Event category toggles ---
# You can reduce logging volume by disabling high-frequency categories.
TELEMETRY_LOG_BIRTHS: bool = _env_bool("FWS_TELEM_BIRTHS", True)
TELEMETRY_LOG_DEATHS: bool = _env_bool("FWS_TELEM_DEATHS", True)
TELEMETRY_LOG_DAMAGE: bool = _env_bool("FWS_TELEM_DAMAGE", True)
TELEMETRY_LOG_KILLS: bool = _env_bool("FWS_TELEM_KILLS", True)
TELEMETRY_LOG_MOVES: bool = _env_bool("FWS_TELEM_MOVES", True)   # can be huge
TELEMETRY_LOG_PPO: bool = _env_bool("FWS_TELEM_PPO", True)

# Move event sampling (used when TELEMETRY_LOG_MOVES is enabled).
# Aggregates are always cheap; per-agent move events are gated by *_EVERY and capped by *_MAX.
TELEMETRY_MOVE_EVENTS_EVERY: int = _env_int("FWS_TELEM_MOVE_EVERY", 0)        # 0=off (aggregates only)
TELEMETRY_MOVE_EVENTS_MAX_PER_TICK: int = _env_int("FWS_TELEM_MOVE_MAX", 256) # hard cap
TELEMETRY_MOVE_EVENTS_SAMPLE_RATE: float = _env_float("FWS_TELEM_MOVE_RATE", 1.0)  # 0..1, deterministic sampler

# Generic extension counters stream (tick,key,value) in telemetry/counters.csv
TELEMETRY_COUNTERS_EVERY: int = _env_int("FWS_TELEM_COUNTERS_EVERY", 0)       # 0=off

# Damage logging mode:
# - "victim_sum": one row per victim per tick (smaller)
# - "per_hit": one row per attacker->victim (bigger)
TELEMETRY_DAMAGE_MODE: str = _env_str("FWS_TELEM_DMG_MODE", "victim_sum").strip().lower()

# Output formats (raw):
# These strings are effectively "feature flags" in your telemetry writer.
TELEMETRY_EVENTS_FORMAT: str = _env_str("FWS_TELEM_EVENTS_FMT", "jsonl").strip().lower()
TELEMETRY_TICKS_FORMAT: str = _env_str("FWS_TELEM_TICKS_FMT", "csv").strip().lower()
TELEMETRY_SNAPSHOT_FORMAT: str = _env_str("FWS_TELEM_SNAP_FMT", "npz").strip().lower()

# Validation strictness: 0=off | 1=basic | 2=strict
# Invariants are sanity checks. Example invariants:
# - hp >= 0
# - positions within bounds
# - alive flags consistent with hp
TELEMETRY_VALIDATE_LEVEL: int = _env_int("FWS_TELEM_VALIDATE", 1)
TELEMETRY_ABORT_ON_ANOMALY: bool = _env_bool("FWS_TELEM_ABORT", False)

# Reports (end-of-run)
# Often: aggregate stats, plots, optional Excel export for quick analysis.
TELEMETRY_REPORT_ENABLE: bool = _env_bool("FWS_TELEM_REPORT", True)
TELEMETRY_REPORT_EXCEL: bool = _env_bool("FWS_TELEM_EXCEL", True)
TELEMETRY_REPORT_PNG: bool = _env_bool("FWS_TELEM_PNG", True)

# =============================================================================
# 💻 HARDWARE ACCELERATION & TENSOR COMPILER
# =============================================================================
# These settings determine how PyTorch executes:
# - CPU vs GPU
# - precision mode (FP32 vs AMP FP16)
# - vectorization (vmap)

# USE_CUDA is only True if both:
# - user wants CUDA (FWS_CUDA=1 by default)
# - torch.cuda.is_available() is True (NVIDIA GPU + driver + CUDA runtime)
USE_CUDA = _env_bool("FWS_CUDA", True) and torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TORCH_DEVICE = DEVICE  # alias; often used by codebases for clarity/consistency

# Automatic Mixed Precision (AMP).
# AMP uses lower precision (FP16) for many GPU operations, which:
# - increases throughput
# - reduces VRAM usage
# - may slightly affect numerical stability (hence careful tuning in PPO)
AMP_ENABLED = _env_bool("FWS_AMP", True)

def amp_enabled() -> bool:
    """Small helper for readability: other modules can call config.amp_enabled()."""
    return AMP_ENABLED

# Tensor dtype used for many computations (especially model inference/training).
# FP16 is only selected when on CUDA + AMP enabled.
TORCH_DTYPE = torch.float16 if (USE_CUDA and AMP_ENABLED) else torch.float32

# PyTorch VMAP (Vectorized Map).
# vmap can reduce Python overhead by batching computations across agents.
USE_VMAP = _env_bool("FWS_USE_VMAP", True)
VMAP_MIN_BUCKET = _env_int("FWS_VMAP_MIN_BUCKET", 8)
VMAP_DEBUG = _env_bool("FWS_VMAP_DEBUG", False)

# =============================================================================
# 🌍 WORLD SCALE & MEMORY ALLOCATION
# =============================================================================
# Grid sizes define the world "canvas". Larger grids increase:
# - memory usage (maps/occupancy)
# - compute cost (raycasts, movement, interactions)
GRID_WIDTH  = _env_int("FWS_GRID_W", 64)
GRID_HEIGHT = _env_int("FWS_GRID_H", 64)

# Starting population per team.
# With two teams, total initial agents = 2 * START_AGENTS_PER_TEAM.
START_AGENTS_PER_TEAM = _env_int("FWS_START_PER_TEAM", 200)

# MAX_AGENTS controls pre-allocation in an SoA (Struct-of-Arrays) AgentRegistry.
# In SoA, you keep a fixed tensor for each attribute (x,y,hp,atk,...) sized [MAX_AGENTS].
# This is GPU-friendly: contiguous memory + vectorized operations.
MAX_AGENTS  = _env_int("FWS_MAX_AGENTS", 600)

# 0 = run simulation math as fast as CPU/GPU allows (no tick cap)
TICK_LIMIT = _env_int("FWS_TICK_LIMIT", 0)

# Target TPS (ticks per second). 0 often means "uncapped".
TARGET_TPS = _env_int("FWS_TARGET_TPS", 0)

# Strict schema width for the AgentRegistry. DO NOT CHANGE.
# This is a structural contract: code assumes there are exactly 10 base features.
AGENT_FEATURES = 10

# =============================================================================
# 🗺️ TOPOGRAPHY & STRATEGIC OBJECTIVES
# =============================================================================
# Map walls + zones define the "meta-game":
# - walls create choke points, line-of-sight dynamics, path constraints
# - heal zones create resource competition
# - capture points create objective-driven conflict

RANDOM_WALLS = _env_int("FWS_RAND_WALLS", 12)
WALL_SEG_MIN = _env_int("FWS_WALL_SEG_MIN", 6)
WALL_SEG_MAX = _env_int("FWS_WALL_SEG_MAX", 43)
WALL_AVOID_MARGIN = _env_int("FWS_WALL_MARGIN", 3)

# Probabilities governing wall generator style.
MAP_WALL_STRAIGHT_PROB = _env_float("FWS_MAP_WALL_STRAIGHT_PROB", 0.75)
MAP_WALL_GAP_PROB      = _env_float("FWS_MAP_WALL_GAP_PROB", 0.08)

# Heal Zones ("water holes")
HEAL_ZONE_COUNT      = _env_int("FWS_HEAL_COUNT", 16)
HEAL_ZONE_SIZE_RATIO = _env_float("FWS_HEAL_SIZE_RATIO", 10/128)
HEAL_RATE            = _env_float("FWS_HEAL_RATE", 0.0005)

# Capture Points ("King of the Hill")
CP_COUNT           = _env_int("FWS_CP_COUNT", 7)
CP_SIZE_RATIO      = _env_float("FWS_CP_SIZE_RATIO", 10/128)
CP_REWARD_PER_TICK = _env_float("FWS_CP_REWARD", 0.05)

# =============================================================================
# ⚔️ COMBAT BIOLOGY & CLASSES
# =============================================================================
# Unit IDs are numeric tags used in tensors. Integers are used because:
# - they are compact
# - they are easy to store in int tensors
# - they can index lookup tables
UNIT_SOLDIER_ID = 1
UNIT_ARCHER_ID  = 2
UNIT_SOLDIER    = UNIT_SOLDIER_ID
UNIT_ARCHER     = UNIT_ARCHER_ID

# HP/ATK tuning affects the "time-to-kill" (TTK), which heavily affects RL learning:
# - If TTK is too low: agents die instantly; credit assignment is harder
# - If TTK is too high: fights drag; learning may be slow and chaotic
MAX_HP     = _env_float("FWS_MAX_HP", 1.0)
SOLDIER_HP = _env_float("FWS_SOLDIER_HP", 1.0)
ARCHER_HP  = _env_float("FWS_ARCHER_HP", 0.65)  # glass cannon archetype

# Attack values
BASE_ATK    = _env_float("FWS_BASE_ATK", 0.35)
SOLDIER_ATK = _env_float("FWS_SOLDIER_ATK", 0.35)
ARCHER_ATK  = _env_float("FWS_ARCHER_ATK", 0.20)
MAX_ATK     = max(SOLDIER_ATK, ARCHER_ATK, BASE_ATK, 1e-6)  # avoid zero division risks downstream

# Engagement distances
ARCHER_RANGE = _env_int("FWS_ARCHER_RANGE", 4)
ARCHER_LOS_BLOCKS_WALLS = _env_bool("FWS_ARCHER_BLOCK_LOS", True)

# =============================================================================
# 🔋 METABOLISM (Anti-Stalemate Mechanic)
# =============================================================================
# Metabolism drains HP over time to prevent corner-camping.
# This creates a "pressure to act" which can:
# - increase exploration
# - force interaction around heal zones
METABOLISM_ENABLED       = _env_bool("FWS_META_ON", True)
META_SOLDIER_HP_PER_TICK = _env_float("FWS_META_SOLDIER", 0.00009)
META_ARCHER_HP_PER_TICK  = _env_float("FWS_META_ARCHER",  0.00005)

# =============================================================================
# 👁️ SENSORS & INSTINCT (THE AI'S "EYES")
# =============================================================================
# Raycasting is typically the most expensive per-tick operation:
# - It checks along lines in discrete grid space for obstacles/targets.
# - Complexity roughly grows with (agents * rays * vision_range).
VISION_RANGE_SOLDIER = _env_int("FWS_VISION_SOLDIER", 8)
VISION_RANGE_ARCHER  = _env_int("FWS_VISION_ARCHER", 10)

VISION_RANGE_BY_UNIT = {
    UNIT_SOLDIER_ID: VISION_RANGE_SOLDIER,
    UNIT_ARCHER_ID:  VISION_RANGE_ARCHER,
}

# Maximum steps for ray marching / LoS checks; must be >= max vision range.
RAYCAST_MAX_STEPS = max(max(VISION_RANGE_BY_UNIT.values()), 1)
RAY_MAX_STEPS     = RAYCAST_MAX_STEPS  # alias used elsewhere

# Instinct radius can be used to compute local density features (e.g., flanking detection).
INSTINCT_RADIUS = _env_int("FWS_INSTINCT_RADIUS", 12)

# =============================================================================
# 🧩 TENSOR OBSERVATION LAYOUT (STRICT CONTRACT)
# =============================================================================
# OBSERVATION FORMAT:
# - Ray tokens: RAY_TOKEN_COUNT rays
# - Each ray has RAY_FEAT_DIM features
# - Additional "rich" features appended (agent state, team context, zone context, etc.)
#
# The neural network expects OBS_DIM exactly. If this changes:
# - old checkpoints become incompatible
# - policy/value heads need resizing
RAY_TOKEN_COUNT = _env_int("FWS_RAY_TOKENS", 32)
RAY_FEAT_DIM    = 8
RAYS_FLAT_DIM   = RAY_TOKEN_COUNT * RAY_FEAT_DIM

RICH_BASE_DIM   = 23
INSTINCT_DIM    = 4
RICH_TOTAL_DIM  = RICH_BASE_DIM + INSTINCT_DIM

OBS_DIM = RAYS_FLAT_DIM + RICH_TOTAL_DIM

# Semantic indices help interpret the "rich" portion.
# These are *documentation + tooling aids* that allow you to map raw indices to meaning.
SEMANTIC_RICH_BASE_INDICES = {
    "own_context":    (0, 1, 2, 5, 6, 7, 8),
    "world_context":  (11, 20, 21, 22),
    "zone_context":   (9, 10),
    "team_context":   (3, 4, 12, 13, 14, 15),
    "combat_context": (16, 17, 18, 19),
}

SEMANTIC_TOKEN_ORDER = (
    "own_context", "world_context", "zone_context", "team_context", "combat_context", "instinct_context"
)

# Action space size: number of discrete actions the agent policy can output.
NUM_ACTIONS = _env_int("FWS_NUM_ACTIONS", 41)

# =============================================================================
# 🔄 POPULATION CONTROL (REINFORCEMENTS)
# =============================================================================
RESPAWN_ENABLED = _env_bool("FWS_RESPAWN", True)

# New wave-based respawn logic parameters:
RESP_FLOOR_PER_TEAM      = _env_int("FWS_RESP_FLOOR_PER_TEAM", 190)
RESP_MAX_PER_TICK        = _env_int("FWS_RESP_MAX_PER_TICK", 15)
RESP_PERIOD_TICKS        = _env_int("FWS_RESP_PERIOD_TICKS", 2000)
RESP_PERIOD_BUDGET       = _env_int("FWS_RESP_PERIOD_BUDGET", 40)
RESP_HYST_COOLDOWN_TICKS = _env_int("FWS_RESP_HYST_COOLDOWN_TICKS", 45)
RESP_WALL_MARGIN         = _env_int("FWS_RESP_WALL_MARGIN", 2)

SPAWN_ARCHER_RATIO       = _env_float("FWS_SPAWN_ARCHER_RATIO", 0.35)

# Legacy fallbacks (older algorithm knobs kept for backward compatibility):
RESPAWN_PROB_PER_DEAD        = _env_float("FWS_RESPAWN_PROB", 0.05)
RESPAWN_SPAWN_TRIES          = _env_int("FWS_RESPAWN_TRIES", 200)
RESPAWN_MUTATION_STD         = _env_float("FWS_MUT_STD", 0.05)
RESPAWN_CLONE_PROB           = _env_float("FWS_CLONE_PROB", 1)
RESPAWN_USE_TEAM_ELITE       = _env_bool("FWS_TEAM_ELITE", True)
RESPAWN_RESET_OPT_ON_RESPAWN = _env_bool("FWS_RESET_OPT", True)
RESPAWN_JITTER_RADIUS        = _env_int("FWS_RESP_JITTER", 3)
RESPAWN_COOLDOWN_TICKS       = _env_int("FWS_RESPAWN_CD", 500)
RESPAWN_BATCH_PER_TEAM       = _env_int("FWS_RESPAWN_BATCH", 1)
RESPAWN_ARCHER_SHARE         = _env_float("FWS_RESPAWN_ARCHER_SHARE", 0.50)
RESPAWN_INTERIOR_BIAS        = _env_float("FWS_RESPAWN_INTERIOR_BIAS", 0.40)

# =============================================================================
# 🏆 REWARD SHAPING (RL Feedback Loop)
# =============================================================================
# Reward shaping defines the "objective function" for learning.
# PPO learns to maximize expected cumulative reward (discounted by gamma).
TEAM_KILL_REWARD       = _env_float("FWS_REW_KILL",       2.0)
TEAM_DMG_DEALT_REWARD  = _env_float("FWS_REW_DMG_DEALT",  0.00)
TEAM_DEATH_PENALTY     = _env_float("FWS_REW_DEATH",     -1.5)
TEAM_DMG_TAKEN_PENALTY = _env_float("FWS_REW_DMG_TAKEN",  0.00)

PPO_REWARD_HP_TICK         = _env_float("FWS_PPO_REW_HP_TICK", 0.0005)
PPO_REWARD_KILL_INDIVIDUAL = _env_float("FWS_PPO_REW_KILL_AGENT", 10.0)
PPO_REWARD_DEATH           = _env_float("FWS_PPO_REW_DEATH", -10.0)
PPO_REWARD_CONTESTED_CP    = _env_float("FWS_PPO_REW_CONTEST", 0.05)

# =============================================================================
# 🧠 REINFORCEMENT LEARNING (PROXIMAL POLICY OPTIMIZATION)
# =============================================================================
# PPO is a policy gradient method with a clipped objective:
# - stable compared to vanilla policy gradients
# - widely used for large action spaces and continuous training loops
#
# Key PPO hyperparameters (very brief):
# - LR: learning rate
# - CLIP: trust region-ish constraint on policy update size
# - ENTROPY: encourages exploration (prevents collapse to deterministic too early)
# - GAMMA: discount factor (how far into the future reward matters)
# - LAMBDA: GAE parameter controlling bias/variance tradeoff
PPO_ENABLED       = _env_bool("FWS_PPO_ENABLED", True)
PPO_RESET_LOG     = _env_bool("FWS_PPO_RESET_LOG", True)
PPO_WINDOW_TICKS  = _env_int("FWS_PPO_TICKS", 256)

PPO_LR            = _env_float("FWS_PPO_LR", 3e-4)
PPO_LR_T_MAX      = _env_int("FWS_PPO_T_MAX", 10_000_000)
PPO_LR_ETA_MIN    = _env_float("FWS_PPO_ETA_MIN", 1e-6)

PPO_CLIP          = _env_float("FWS_PPO_CLIP", 0.2)
PPO_CLIP_EPS      = PPO_CLIP
PPO_ENTROPY_COEF  = _env_float("FWS_PPO_ENTROPY", 0.05)
PPO_VALUE_COEF    = _env_float("FWS_PPO_VCOEF", 0.5)
PPO_EPOCHS        = _env_int("FWS_PPO_EPOCHS", 4)
PPO_MINIBATCHES   = _env_int("FWS_PPO_MINIB", 8)
PPO_MAX_GRAD_NORM = _env_float("FWS_PPO_MAXGN", 0.5)
PPO_TARGET_KL     = _env_float("FWS_PPO_TKL", 0.02)
PPO_GAMMA         = _env_float("FWS_PPO_GAMMA", 0.995)
PPO_LAMBDA        = _env_float("FWS_PPO_LAMBDA", 0.95)
PPO_UPDATE_TICKS  = _env_int("FWS_PPO_UPDATE_TICKS", 5)

# Backward-compatible knob interaction:
# If FWS_PPO_ENTROPY_BONUS is set and FWS_PPO_ENTROPY is not set,
# use PPO_ENTROPY_BONUS as PPO_ENTROPY_COEF.
PPO_ENTROPY_BONUS = _env_float("FWS_PPO_ENTROPY_BONUS", PPO_ENTROPY_COEF)
if _env_is_set("FWS_PPO_ENTROPY_BONUS") and not _env_is_set("FWS_PPO_ENTROPY"):
    PPO_ENTROPY_COEF = float(PPO_ENTROPY_BONUS)

# Per-agent brains and mutation schedule (evolution-like dynamics)
PER_AGENT_BRAINS        = _env_bool("FWS_PER_AGENT_BRAINS", True)
MUTATION_PERIOD_TICKS   = _env_int("FWS_MUTATE_EVERY", 100000)
MUTATION_FRACTION_ALIVE = _env_float("FWS_MUTATE_FRAC", 0.002)

# =============================================================================
# 🤖 BRAIN ARCHITECTURE (THE NEURAL ENGINES)
# =============================================================================
# Team assignment:
# - historically: one architecture per team (Red vs Blue)
# - now: can mix architectures within a team (alternate or random)
BRAIN_KIND: str = os.getenv("FWS_BRAIN", "tron").strip().lower()
TEAM_BRAIN_ASSIGNMENT: bool = _env_bool("FWS_TEAM_BRAIN_ASSIGNMENT", True)

# Team brain assignment mode:
#   - "exclusive": classic behavior (team fixed brain)
#   - "mix": each team can spawn BOTH architectures
TEAM_BRAIN_ASSIGNMENT_MODE: str = _env_str("FWS_TEAM_BRAIN_MODE", "mix").strip().lower()

# If mixing:
#   - "alternate": deterministic 50/50 alternating assignment
#   - "random": probabilistic using TEAM_BRAIN_MIX_P_TRON
TEAM_BRAIN_MIX_STRATEGY: str = _env_str("FWS_TEAM_BRAIN_MIX_STRATEGY", "alternate").strip().lower()

# Probability that a newly created brain is Tron (else Mirror).
TEAM_BRAIN_MIX_P_TRON: float = _env_float("FWS_TEAM_BRAIN_MIX_P_TRON", 0.5)

# Seed for ONLY the brain-mix RNG (does not affect map spawning RNG).
TEAM_BRAIN_MIX_SEED: int = _env_int("FWS_TEAM_BRAIN_MIX_SEED", int(globals().get("RNG_SEED", 0)))

# --- Tron Transformer Hyperparameters ---
# d_model must typically be divisible by num_heads.
TRON_D_MODEL       = _env_int("FWS_TRON_DMODEL", 32)
TRON_HEADS         = _env_int("FWS_TRON_HEADS", 8)
TRON_DROPOUT       = _env_float("FWS_TRON_DROPOUT", 0.05)
TRON_RAY_LAYERS    = _env_int("FWS_TRON_RAY_LAYERS", 3)
TRON_SEM_LAYERS    = _env_int("FWS_TRON_SEM_LAYERS", 3)
TRON_FUSION_LAYERS = _env_int("FWS_TRON_FUSION_LAYERS", 2)
TRON_MLP_HIDDEN    = _env_int("FWS_TRON_MLP_HID", 256)
TRON_USE_ROPE      = _env_bool("FWS_TRON_ROPE", True)
TRON_USE_PRENORM   = _env_bool("FWS_TRON_PRENORM", True)

# =============================================================================
# 🪞 MIRROR TRANSFORMER HYPERPARAMETERS
# =============================================================================
# Mirror defaults to Tron values for safety (no behavior change unless overridden).
MIRROR_D_MODEL       = _env_int("FWS_MIRROR_DMODEL", int(TRON_D_MODEL))
MIRROR_HEADS         = _env_int("FWS_MIRROR_HEADS", int(TRON_HEADS))
MIRROR_DROPOUT       = _env_float("FWS_MIRROR_DROPOUT", float(TRON_DROPOUT))
MIRROR_RAY_LAYERS    = _env_int("FWS_MIRROR_RAY_LAYERS", int(TRON_RAY_LAYERS))
MIRROR_SEM_LAYERS    = _env_int("FWS_MIRROR_SEM_LAYERS", int(TRON_SEM_LAYERS))
MIRROR_FUSION_LAYERS = _env_int("FWS_MIRROR_FUSION_LAYERS", int(TRON_FUSION_LAYERS))
MIRROR_MLP_HIDDEN    = _env_int("FWS_MIRROR_MLP_HID", int(TRON_MLP_HIDDEN))
MIRROR_USE_ROPE      = _env_bool("FWS_MIRROR_ROPE", bool(TRON_USE_ROPE))
MIRROR_USE_PRENORM   = _env_bool("FWS_MIRROR_PRENORM", bool(TRON_USE_PRENORM))

# =============================================================================
# 🖥️ UI, VIEWER & SCREEN RECORDING (SMOOTH 60FPS)
# =============================================================================
ENABLE_UI  = _env_bool("FWS_UI", True)

# Viewer refresh throttling:
# Pulling GPU tensors back to CPU every tick can stall.
# Lower refresh rates reduce PCIe transfers and keep UI smooth.
VIEWER_STATE_REFRESH_EVERY = _env_int("FWS_VIEWER_STATE_REFRESH_EVERY", 3)
VIEWER_PICK_REFRESH_EVERY  = _env_int("FWS_VIEWER_PICK_REFRESH_EVERY", 3)

UI_FONT_NAME: str = _env_str("FWS_UI_FONT", "consolas")
VIEWER_CENTER_WINDOW: bool = _env_bool("FWS_VIEWER_CENTER_WINDOW", True)

CELL_SIZE  = _env_int("FWS_CELL_SIZE", 5)
HUD_WIDTH  = _env_int("FWS_HUD_W", 340)
TARGET_FPS = _env_int("FWS_TARGET_FPS", 60)

# Optional direct-to-disk recording
RECORD_VIDEO: bool     = _env_bool("FWS_RECORD_VIDEO", False)
VIDEO_FPS: int         = _env_int("FWS_VIDEO_FPS", 60)
VIDEO_SCALE: int       = _env_int("FWS_VIDEO_SCALE", 4)
VIDEO_EVERY_TICKS: int = _env_int("FWS_VIDEO_EVERY_TICKS", 1)

# Visual styling constants (RGB/RGBA tuples).
# Keeping these in config avoids scattering magic numbers across UI code.
UI_COLORS = {
    "bg": (15, 17, 22), "hud_bg": (10, 12, 16), "side_bg": (14, 16, 20),
    "grid": (35, 37, 42), "border": (80, 85, 95), "wall": (100, 105, 115),
    "empty": (20, 22, 28),

    "red_soldier": (240, 50, 50), "red_archer":  (255, 120, 0), "red": (240, 50, 50),
    "blue_soldier": (30, 160, 255), "blue_archer":  (0, 220, 180), "blue": (30, 160, 255),

    "archer_glyph": (255, 245, 120), "marker": (255, 255, 255),
    "text": (240, 240, 245), "text_dim": (160, 165, 175),
    "green": (50, 220, 120), "warn": (255, 170, 0),
    "bar_bg": (30, 35, 40), "bar_fg": (50, 220, 120),
    "graph_red": (240, 50, 50, 180), "graph_blue": (30, 160, 255, 180),
    "graph_grid": (50, 50, 60), "pause_text": (255, 200, 50)
}

# =============================================================================
# 🛠️ PROFILE OVERRIDE INJECTION
# =============================================================================
def _apply_profile_overrides() -> None:
    """
    Apply preset overrides based on PROFILE.

    How it works
    ------------
    - If PROFILE == "default": do nothing.
    - Else, look up preset rows in `presets`.
    - Each row is: (ENV_KEY, GLOBAL_VAR_NAME, VALUE)
    - Only apply preset values if the user did NOT explicitly set ENV_KEY.

    This provides a powerful pattern:
    - "debug" profile makes a smaller/faster world
    - "train_fast" profile maximizes throughput
    - "train_quality" profile increases model capacity

    Safety check
    ------------
    Transformer requirement:
        d_model % heads == 0
    because each attention head gets a slice of the embedding dimension.
    """
    if PROFILE == "default":
        return

    presets = {
        "debug": [
            ("FWS_GRID_W", "GRID_WIDTH", 80),
            ("FWS_GRID_H", "GRID_HEIGHT", 80),
            ("FWS_START_PER_TEAM", "START_AGENTS_PER_TEAM", 30),
            ("FWS_MAX_AGENTS", "MAX_AGENTS", 160),
            ("FWS_RAND_WALLS", "RANDOM_WALLS", 6),
            ("FWS_UI", "ENABLE_UI", True),
            ("FWS_TARGET_FPS", "TARGET_FPS", 30),
            ("FWS_RECORD_VIDEO", "RECORD_VIDEO", False),
            ("FWS_USE_VMAP", "USE_VMAP", False),
        ],
        "train_fast": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_TRON_DMODEL", "TRON_D_MODEL", 96),
            ("FWS_TRON_HEADS", "TRON_HEADS", 4),
            ("FWS_TRON_RAY_LAYERS", "TRON_RAY_LAYERS", 2),
            ("FWS_TRON_SEM_LAYERS", "TRON_SEM_LAYERS", 2),
            ("FWS_TRON_FUSION_LAYERS", "TRON_FUSION_LAYERS", 1),
            ("FWS_TRON_MLP_HID", "TRON_MLP_HIDDEN", 192),
        ],
        "train_quality": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_TRON_DMODEL", "TRON_D_MODEL", 192),
            ("FWS_TRON_HEADS", "TRON_HEADS", 6),
            ("FWS_TRON_RAY_LAYERS", "TRON_RAY_LAYERS", 4),
            ("FWS_TRON_SEM_LAYERS", "TRON_SEM_LAYERS", 4),
            ("FWS_TRON_FUSION_LAYERS", "TRON_FUSION_LAYERS", 2),
            ("FWS_TRON_MLP_HID", "TRON_MLP_HIDDEN", 384),
        ],
    }

    rows = presets.get(PROFILE)
    if not rows: return

    # globals() returns the module-level namespace dict.
    # This is how we programmatically override variables defined above.
    g = globals()

    # Only override if the user did NOT explicitly set the env var for that key.
    for env_key, var_name, value in rows:
        if not _env_is_set(env_key):
            g[var_name] = value

    # Safety: transformer embedding dimension must be divisible by number of heads.
    if int(g.get("TRON_D_MODEL", 0)) % max(int(g.get("TRON_HEADS", 1)), 1) != 0:
        raise ValueError(
            f"TRON_D_MODEL must be divisible by TRON_HEADS "
            f"(got {g.get('TRON_D_MODEL')} / {g.get('TRON_HEADS')})"
        )

# Apply profile presets immediately at import time.
# That means importing config produces final values.
_apply_profile_overrides()

def summary_str() -> str:
    """
    Return a compact one-line description of the run configuration.

    Why this is useful
    ------------------
    - Logs become searchable.
    - It's easy to confirm you're running what you think you are running.
    - This can be printed at startup and embedded into output folders/metadata.

    Returns
    -------
    str
        Example:
            [final_war_sim: GOD LEVEL] dev=cuda grid=128x128 start=200/team obs=... acts=... AMP=on
    """
    return (
        f"[final_war_sim: GOD LEVEL] "
        f"dev={DEVICE.type} "
        f"grid={GRID_WIDTH}x{GRID_HEIGHT} "
        f"start={START_AGENTS_PER_TEAM}/team "
        f"obs={OBS_DIM} acts={NUM_ACTIONS} "
        f"AMP={'on' if AMP_ENABLED else 'off'}"
    )