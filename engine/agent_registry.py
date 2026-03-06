from __future__ import annotations
# ──────────────────────────────────────────────────────────────────────────────
# Forward-annotation semantics (Python typing)
# ──────────────────────────────────────────────────────────────────────────────
# This directive ensures that all type annotations are stored as strings rather
# than evaluated immediately. In large systems—especially those with complex
# module graphs—this reduces import-time coupling and prevents NameError issues
# when annotations refer to classes or types defined later.
#
# Importantly, this changes only annotation evaluation behavior; it does not
# modify runtime tensor computations or model execution.
from __future__ import annotations
import weakref
# --- PERF PATCH C: module-level brain signature cache ---
# Rationale: _signature() is called for every alive agent every tick. It traverses
# model.named_modules() and joins strings. Architecture never changes between ticks.
#
# WeakKeyDictionary is lifecycle-safe: when a brain module is replaced in register()
# and the old reference is GC'd, its cache entry automatically vanishes.
# No manual invalidation is needed — brain identity changes on replacement.
_BRAIN_SIG_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# ──────────────────────────────────────────────────────────────────────────────
# Standard library imports
# ──────────────────────────────────────────────────────────────────────────────
# dataclass: provides a concise way to define classes used primarily as data
# containers. It auto-generates __init__, __repr__, equality, etc.
from dataclasses import dataclass
#
# List, Dict, Optional: typing primitives used for clarity and static tooling.
from typing import List, Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# PyTorch imports
# ──────────────────────────────────────────────────────────────────────────────
# torch: tensor computation library (CPU/GPU).
# torch.nn: neural network modules; used for typing and storing agent “brains”.
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Project configuration
# ──────────────────────────────────────────────────────────────────────────────
# This module is expected to contain global knobs such as:
#   • MAX_AGENTS     : maximum number of agents the registry can store
#   • TORCH_DTYPE    : dtype for the agent_data tensor (float32/float16/bfloat16)
#   • TORCH_DEVICE   : device string or torch.device (e.g. "cuda", "cpu")
#   • UNIT_SOLDIER / UNIT_ARCHER : unit subtype encodings (defaults 1.0/2.0)
#
# The code uses getattr(...) in places for defensive defaults.
import config

# =============================================================================
# Column layout (Struct-of-Arrays for GPU efficiency)
# =============================================================================
# This registry stores agent attributes in a single 2D tensor:
#
#   agent_data: shape (capacity, NUM_COLS)
#
# Each column corresponds to a specific attribute, and each row corresponds to
# a slot (agent index) in the registry.
#
# This is a classic “Struct-of-Arrays” (SoA) layout:
#   Instead of storing each agent as a Python object (Array-of-Structs), all
#   agents’ attributes are stored column-wise in a dense GPU-friendly tensor.
#
# Why SoA is valuable in ML + simulation settings:
#   • Coalesced memory access on GPU:
#       When you read an attribute for many agents (e.g., all X positions),
#       you access a contiguous slice, which is cache- and bandwidth-friendly.
#   • Vectorized operations:
#       Most simulation and RL pipelines want to operate on batches of agents
#       simultaneously. A SoA layout encourages pure tensor operations rather
#       than Python loops.
#   • Lower Python overhead:
#       Avoids per-agent Python objects, which become a bottleneck at scale.
#
# IMPORTANT CONSTRAINT:
#   The tensor dtype is configured by config.TORCH_DTYPE, which may be float16
#   under AMP (automatic mixed precision). Therefore, certain values that
#   require exact integer fidelity (like permanent unique IDs) should not be
#   stored solely in this float tensor. This is addressed below with agent_uids.
COL_ALIVE = 0       # float: 1.0 alive, 0.0 dead
COL_TEAM  = 1       # float: 2.0 red, 3.0 blue
COL_X     = 2       # float: x coordinate
COL_Y     = 3       # float: y coordinate
COL_HP    = 4       # float: current health points
COL_UNIT  = 5       # float: 1.0 = Soldier, 2.0 = Archer
COL_HP_MAX = 6      # float: maximum health points for this agent
COL_VISION = 7      # float: vision range in cells for this agent
COL_ATK    = 8      # float: attack power for this agent
COL_AGENT_ID = 9    # float: permanent unique ID for this agent
NUM_COLS = 10

# -----------------------------------------------------------------------------
# Configuration coherence: optional synchronization hook
# -----------------------------------------------------------------------------
# Some codebases store the number of agent features in configuration so that
# other modules can allocate tensors of compatible shape.
#
# This conditional ensures:
#   • If config.AGENT_FEATURES exists (i.e., the configuration expects it),
#     it is updated to match this module’s NUM_COLS.
#
# NOTE:
#   This does not create config.AGENT_FEATURES if it does not already exist.
#   It merely synchronizes it if present.
if hasattr(config, 'AGENT_FEATURES'):
    config.AGENT_FEATURES = NUM_COLS

# -----------------------------------------------------------------------------
# Team identifiers
# -----------------------------------------------------------------------------
# The grid/occupancy code elsewhere appears to encode team identity as:
#   2.0 = red
#   3.0 = blue
#
# These are floats because agent_data is float-typed. The choice of 2.0/3.0 is
# likely aligned with occupancy channel conventions in the grid representation.
TEAM_RED_ID  = 2.0
TEAM_BLUE_ID = 3.0

# -----------------------------------------------------------------------------
# Unit subtype identifiers
# -----------------------------------------------------------------------------
# Unit encodings are read from config if present; otherwise default to 1.0/2.0.
#
# Casting to float ensures compatibility with the float agent_data tensor.
UNIT_SOLDIER = float(getattr(config, "UNIT_SOLDIER", 1.0))
UNIT_ARCHER  = float(getattr(config, "UNIT_ARCHER", 2.0))

# =============================================================================
# Buckets: allow grouping agents with same NN architecture
# =============================================================================
# In multi-agent RL or simulation environments, it is common to have many agents
# sharing the same model architecture, but not necessarily the same instance.
#
# “Bucketing” groups agents by a signature that describes their NN structure.
# This enables batched inference where agents with identical architectures can
# be processed together more efficiently.
#
# Even if each agent has its own parameters (distinct nn.Module instance),
# inference can still be batched if the forward pass shape is compatible, though
# parameter batching is non-trivial without additional engineering. This design
# suggests the system may be preparing for architectural grouping as a first
# step toward throughput optimization.
@dataclass
class Bucket:
    # signature: a string describing the model architecture (not parameters),
    # used as a grouping key.
    signature: str

    # indices: LongTensor [K] containing the registry slots belonging to this
    # bucket. These are agent indices in the registry.
    indices: torch.Tensor

    # models: list of nn.Module objects corresponding to those indices, in the
    # same order. The list length is K, matching indices.
    models: List[nn.Module]
    # locs[j] == position of indices[j] within alive_idx, so alive_idx[locs] == indices.
    # This eliminates torch.searchsorted in the caller (tick.py) each tick.
    locs: torch.Tensor      # shape (K,), dtype long


class AgentsRegistry:
    """
    A high-throughput agent registry backed primarily by a dense tensor.

    Conceptual role in the system:
    ------------------------------
    This component functions as the “single source of truth” for agent state
    during simulation steps. The simulation grid tracks occupancy and agent ids,
    while this registry stores the full attribute set of each agent (HP, team,
    position, brain, etc.).

    Data model:
    -----------
    • Each agent resides in a fixed registry slot (0 <= slot < capacity).
    • The grid’s agent-id channel is expected to refer to these slots/ids.
    • The main numerical state is stored in `agent_data` with NUM_COLS columns.
    • Neural network “brains” are stored separately in a Python list, because
      nn.Module objects are not representable in a tensor.

    Performance rationale:
    ----------------------
    The SoA tensor layout allows vectorized batch operations over large agent
    populations and is conducive to GPU execution.

    Important precision consideration:
    ----------------------------------
    Mixed precision training/inference often uses float16/bfloat16 for speed.
    However, integer-like identifiers (unique IDs) cannot safely live in such
    dtypes without risk of precision loss. Therefore, the registry stores UIDs
    in an int64 side tensor (`agent_uids`) and keeps the float column
    COL_AGENT_ID as a display/compatibility field only.
    """

    def __init__(self, grid: torch.Tensor) -> None:
        # Store a reference to the grid tensor, enabling the registry to share
        # device placement and potentially coordinate updates with the grid.
        self.grid = grid

        # In PyTorch, tensors have a `.device` attribute indicating CPU or GPU.
        # Aligning all internal tensors to the grid device avoids device mismatch.
        self.device = grid.device

        # capacity is the maximum number of agents the registry can store.
        # It is determined by configuration.
        self.capacity = int(config.MAX_AGENTS)

        # Internal monotonically increasing counter for assigning globally unique
        # IDs. This is separate from the registry slot index.
        #
        # Motivation:
        #   • Slots can be reused after agents die.
        #   • A permanent unique ID can be used for analytics, lineage tracking,
        #     reproducibility, and debugging across long simulations.
        self._next_agent_id: int = 0

        # ---------------------------------------------------------------------
        # Main agent tensor (SoA layout)
        # ---------------------------------------------------------------------
        # agent_data: shape (capacity, NUM_COLS)
        #
        # dtype is configurable (may be float32, float16, bfloat16).
        # device is configurable as well. Note: the code uses config.TORCH_DEVICE
        # rather than self.device; this presumes the config and grid device match.
        #
        # The tensor is initialized with zeros; columns are populated on register().
        self.agent_data = torch.zeros(
            (self.capacity, NUM_COLS),
            dtype=config.TORCH_DTYPE,
            device=config.TORCH_DEVICE
        )

        # Mark all agents as not alive initially (explicitly).
        # This is redundant with zeros initialization but makes intent explicit.
        self.agent_data[:, COL_ALIVE] = 0.0

        # ---------------------------------------------------------------------
        # Permanent unique IDs stored as int64
        # ---------------------------------------------------------------------
        # Rationale:
        #   Under mixed precision (float16), large integers lose precision.
        #   Over long runs, agent IDs can become large enough that float16 cannot
        #   represent them exactly, causing collisions or incorrect identity.
        #
        # Therefore:
        #   • agent_uids is the authoritative UID store (int64).
        #   • COL_AGENT_ID in agent_data is auxiliary, possibly clamped.
        self.agent_uids = torch.full(
            (self.capacity,),
            -1,
            dtype=torch.int64,
            device=config.TORCH_DEVICE,
        )

        # ---------------------------------------------------------------------
        # Brains and generations stored separately (Python lists)
        # ---------------------------------------------------------------------
        # brains: per-slot nn.Module or None.
        #   Cannot be stored in a tensor; modules are Python objects with state.
        self.brains: List[Optional[nn.Module]] = [None] * self.capacity

        # generations: per-slot integer generation index (e.g., evolutionary runs).
        self.generations: List[int] = [0] * self.capacity

        # ---------------------------------------------------------------------
        # Expose column constants as instance attributes
        # ---------------------------------------------------------------------
        # This is a convenience so external code can access:
        #   registry.COL_HP, registry.COL_X, etc.
        # without relying on module-level constants.
        #
        # Note: It duplicates information already defined above; duplication is
        # acceptable here for ergonomics.
        self.COL_ALIVE, self.COL_TEAM, self.COL_X, self.COL_Y, self.COL_HP, self.COL_UNIT = 0, 1, 2, 3, 4, 5
        self.COL_HP_MAX, self.COL_VISION, self.COL_ATK, self.COL_AGENT_ID = 6, 7, 8, 9


    def clear(self) -> None:
        """
        Reset all agents while preserving registry capacity.

        Operational intent:
        -------------------
        This method returns the registry to a pristine state:
          • all numeric fields zeroed
          • all alive flags cleared
          • all unique IDs invalidated (-1)
          • all brains removed (None)
          • all generations reset
          • unique-id counter reset to 0

        Typical use cases:
        ------------------
        • Resetting an episode/environment without re-allocating tensors.
        • Clearing state between experiments while keeping GPU memory stable.

        Performance considerations:
        ---------------------------
        • zero_() is an in-place operation that avoids reallocations.
        • Reassigning Python lists does allocate list objects, but this is small
          relative to tensor allocations and avoids stale references.
        """
        self.agent_data.zero_()
        self.agent_data[:, COL_ALIVE] = 0.0
        self.agent_uids.fill_(-1)
        self.brains = [None] * self.capacity
        self.generations = [0] * self.capacity
        self._next_agent_id = 0

    def get_next_id(self) -> int:
        """
        Return the next available unique agent ID and increment the counter.

        Formal behavior:
        ---------------
        Let c be the current internal counter value. This method returns c and
        then updates the counter to c+1. Therefore, IDs are strictly increasing
        over time.

        Properties:
        -----------
        • Uniqueness (within a single process lifetime): guaranteed.
        • Monotonicity: guaranteed.
        • Reuse: not performed; IDs never repeat unless counter is reset via clear().

        Why not use slot index as ID?
        -----------------------------
        Because slots are reused when agents die, slot indices do not provide
        temporal uniqueness or lineage tracking across long simulations.
        """
        agent_id = self._next_agent_id
        self._next_agent_id += 1
        return agent_id

    def register(
        self,
        slot: int,
        *,
        agent_id: int,
        team_is_red: bool,
        x: int,
        y: int,
        hp: float,
        atk: float,
        brain: nn.Module,
        unit: float | int,
        hp_max: float,
        vision_range: int,
        generation: int = 0,
    ) -> None:
        """
        Register (or overwrite) an agent in a fixed registry slot.

        Parameter semantics:
        --------------------
        slot:
            The registry index where this agent will reside. Slots are fixed
            positions in the SoA tensor. The caller is responsible for choosing
            a free/appropriate slot.

        agent_id:
            Permanent unique identifier for the agent (authoritative ID stored
            in self.agent_uids). This is distinct from slot index.

        team_is_red:
            Boolean flag controlling team assignment:
              True  -> TEAM_RED_ID
              False -> TEAM_BLUE_ID

        x, y:
            Grid coordinates. Stored as floats in agent_data for uniform dtype,
            but semantically represent integer cell positions.

        hp:
            Current hit points of the agent.

        atk:
            Attack power, presumably used by combat logic.

        brain:
            Neural network module controlling agent policy/behavior.

        unit:
            Unit subtype (soldier/archer). Stored as float for compatibility.

        hp_max:
            Maximum hit points for normalization or constraints.

        vision_range:
            Agent-specific vision in grid cells (used by raycasting logic).

        generation:
            Integer generation marker, default 0. This suggests evolutionary
            mechanics (mutation/selection) or curriculum tracking.

        Guarantees:
        -----------
        After this method:
          • agent is marked alive in COL_ALIVE
          • agent attributes are written into agent_data
          • UID is written into agent_uids
          • brain is moved to the registry device and stored
          • generation is stored

        Precision / AMP safety:
        -----------------------
        The UID is stored in int64 (agent_uids). A float representation is also
        stored in COL_AGENT_ID, but clamped to dtype range when possible.
        """

        # Safety: ensure slot is in-bounds.
        assert 0 <= slot < self.capacity

        # Local alias for the agent_data tensor for concise indexing.
        d = self.agent_data

        # Mark slot alive.
        d[slot, COL_ALIVE] = 1.0

        # Assign team encoding according to boolean flag.
        d[slot, COL_TEAM]  = TEAM_RED_ID if team_is_red else TEAM_BLUE_ID

        # Store spatial coordinates.
        # Although semantically integer grid coordinates, they are stored as float
        # due to agent_data dtype constraints.
        d[slot, COL_X]     = float(x)
        d[slot, COL_Y]     = float(y)

        # Store health and attack parameters.
        d[slot, COL_HP]    = float(hp)
        d[slot, COL_ATK]   = float(atk)

        # Store unit subtype.
        d[slot, COL_UNIT]  = float(unit)

        # Store per-agent maximum HP and vision range.
        d[slot, COL_HP_MAX] = float(hp_max)
        d[slot, COL_VISION] = float(vision_range)

        # Store true UID in the int64 tensor (authoritative identity store).
        self.agent_uids[slot] = int(agent_id)

        # Also store a float representation of agent_id in agent_data.
        #
        # Motivation:
        #   Some downstream tooling or debugging UI may rely on agent_data only.
        #
        # Risk:
        #   If dtype is float16, large IDs cannot be represented exactly.
        #
        # Mitigation implemented here:
        #   Attempt to clamp to maximum representable float value of d.dtype
        #   using torch.finfo(d.dtype).max.
        try:
            max_f = torch.finfo(d.dtype).max
            d[slot, COL_AGENT_ID] = float(min(float(agent_id), float(max_f)))
        except Exception:
            # Fallback: store directly if finfo is not available or dtype is not
            # floating (though in practice it should be floating here).
            d[slot, COL_AGENT_ID] = float(agent_id)

        # Store the brain module in the per-slot list.
        #
        # brain.to(self.device) moves parameters to the same device as the grid,
        # ensuring inference happens without device mismatch.
        #
        # Note: this mutates brain in-place in the sense that it returns a module
        # placed on the target device; PyTorch modules are typically moved by
        # calling .to(...) which returns the same module reference.
        self.brains[slot]  = brain.to(self.device)

        # Store generation marker as a Python int.
        self.generations[slot] = int(generation)

    def kill(self, slots: torch.Tensor) -> None:
        """
        Mark agents as dead in the registry.

        Input:
        ------
        slots:
            A tensor of slot indices to kill.

        Behavior:
        ---------
        • If slots is None or empty, do nothing.
        • Otherwise, sets COL_ALIVE=0.0 for those slots.

        Separation of concerns:
        -----------------------
        The docstring notes that grid clearing happens in TickEngine. This is an
        important architectural detail: the registry is responsible for agent
        state, while the simulation engine is responsible for spatial occupancy
        updates. Decoupling these responsibilities reduces hidden side effects.
        """
        if slots is None or slots.numel() == 0:
            return
        self.agent_data[slots, COL_ALIVE] = 0.0

    def positions_xy(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Return a LongTensor of shape (N, 2) containing integer XY positions.

        Inputs:
        -------
        indices:
            1D tensor of agent slot indices whose positions should be returned.

        Output:
        -------
        A tensor of shape (N, 2) where:
            out[:, 0] = x coordinate
            out[:, 1] = y coordinate

        Implementation detail:
        ----------------------
        Positions are stored as floats in agent_data for dtype uniformity; here
        they are cast to long because they are used as integer grid indices in
        downstream computations (e.g., raycasting, grid indexing).
        """
        x = self.agent_data[indices, COL_X].to(torch.long)
        y = self.agent_data[indices, COL_Y].to(torch.long)
        return torch.stack((x, y), dim=1)

    @staticmethod
    def _signature(model: nn.Module) -> str:
        """
        Create a lightweight architecture fingerprint for a model.

        Objective:
        ----------
        Group agents whose brains share the same architecture so inference can be
        batched or otherwise optimized.

        Method:
        -------
        The signature is a string constructed from:
          • the model class name
          • the (in_features, out_features) pairs of any nn.Linear layers

        Example:
        --------
        A simple MLP might produce:
          "MyMLP|L(64,128)|L(128,64)|L(64,10)"

        Important note:
        ---------------
        This is *not* a hash of parameters. Two distinct model instances with the
        same layer shapes will have the same signature even if weights differ.
        That is correct for “architecture grouping”.

        Robustness:
        ----------
        The method is wrapped in try/except to avoid breaking if:
          • model.named_modules() behaves unexpectedly
          • modules are exotic or dynamically defined
          • any attribute access fails
        """
        cached = _BRAIN_SIG_CACHE.get(model)
        if cached is not None:
            return cached
        sig_parts: List[str] = [model.__class__.__name__]
        try:
            for _, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    sig_parts.append(f"L({m.in_features},{m.out_features})")
        except Exception:
            # If anything goes wrong, fall back to the class name only.
            # This still provides a coarse grouping.
            pass
        result = "|".join(sig_parts)
        try:
            _BRAIN_SIG_CACHE[model] = result
        except TypeError:
            pass  # non-weakly-referenceable objects (TorchScript) fall through gracefully
        return result

    def build_buckets(self, alive_idx: torch.Tensor) -> List[Bucket]:
        """
        Group alive agents by model signature for batched inference.

        Input:
        ------
        alive_idx:
            A 1D tensor of indices (slots) for agents currently considered alive.

        Output:
        -------
        A list of Bucket objects. Each bucket corresponds to a unique signature
        and contains:
          • indices: slot indices in that bucket
          • models: corresponding brain modules

        Algorithm:
        ----------
        1) Iterate over alive_idx (converted to Python list for iteration).
        2) For each agent slot:
            • retrieve brain module
            • if brain is None: mark agent dead and skip
            • else compute signature and append slot to dict[signature]
        3) Convert each dict entry into a Bucket:
            • indices stored as a LongTensor on the registry device
            • models list retrieved in the same order

        Design implications:
        --------------------
        The presence of per-agent brain modules suggests agents may undergo
        mutation, specialization, or per-agent learning. Grouping by signature
        enables future optimizations such as:
          • batching observation tensors for all agents in the same bucket
          • running inference in fewer forward passes
          • reducing Python overhead from per-agent calls

        Correctness note:
        -----------------
        If an alive agent has brain=None, the code forcefully sets its alive flag
        to 0.0. This indicates that the registry treats "missing brain" as an
        invalid agent state and resolves it by killing the agent.
        """
        # Each entry is (local_pos_in_alive_idx, global_slot_id).
        buckets_dict: Dict[str, List] = {}

        # Iterate over agent indices. alive_idx.tolist() materializes data to CPU
        # and creates a Python list; this is fine when the number of alive agents
        # is not enormous, but it is a deliberate trade-off for simplicity.
        for local_pos, i in enumerate(alive_idx.tolist()):
            brain = self.brains[i]

            # If brain is missing, declare the agent dead as a consistency fix.
            if brain is None:
                self.agent_data[i, COL_ALIVE] = 0.0
                continue

            # Compute signature key and append slot index into bucket dictionary.
            key = self._signature(brain)
            buckets_dict.setdefault(key, []).append((local_pos, i))

        out: List[Bucket] = []

        # Construct Bucket objects from the dictionary.
        for key, lst in buckets_dict.items():
            # Convert slot list into a LongTensor on the registry device.
            locs_list  = [lp for lp, _  in lst]
            slots_list = [gs for _,  gs in lst]

            idx  = torch.tensor(slots_list, dtype=torch.long, device=self.device)
            locs = torch.tensor(locs_list,  dtype=torch.long, device=self.device)

            models = [
                self.brains[j]
                for j in slots_list
                if j < len(self.brains) and self.brains[j] is not None
            ]

            # Only append non-empty buckets.
            if models:
                out.append(Bucket(signature=key, indices=idx, models=models, locs=locs))

        return out