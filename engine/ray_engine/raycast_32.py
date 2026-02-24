from __future__ import annotations
# ──────────────────────────────────────────────────────────────────────────────
# Future annotations
# ──────────────────────────────────────────────────────────────────────────────
# This import ensures that *type annotations* are treated as "forward references"
# by default (i.e., stored as strings rather than evaluated immediately).
#
# Why this matters:
#   • It avoids runtime NameError issues when annotating with classes/types that
#     are defined later in the file.
#   • It reduces import-order coupling and can slightly improve import-time
#     performance.
#   • It is particularly useful in larger codebases with interdependent modules.
#
# Note: This does not change program behavior for the core tensor computations
# below; it affects only annotation evaluation semantics.
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Typing imports
# ──────────────────────────────────────────────────────────────────────────────
# Optional[T] expresses that a value may be of type T or may be None.
# Here it is used to indicate that `max_steps_each` can be omitted.
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Numerical / ML library imports
# ──────────────────────────────────────────────────────────────────────────────
# torch: tensor library used for GPU-accelerated computation and automatic
#        differentiation. In this function we explicitly disable gradients
#        because raycasting features are observational and do not require grads.
import torch
#
# numpy: used here solely for generating evenly spaced angles and their
#        corresponding unit direction vectors. This is done once at import time.
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Project configuration
# ──────────────────────────────────────────────────────────────────────────────
# A module expected to hold global knobs such as:
#   • RAYCAST_MAX_STEPS   (int): maximum ray marching length
#   • MAX_HP              (float or int): maximum hit points
#   • TORCH_DTYPE         (torch dtype): e.g., torch.float32 / float16 / bfloat16
#
# The code uses getattr(...) to avoid hard-failing if these are not defined.
import config

# ──────────────────────────────────────────────────────────────────────────────
# Grid format documentation
# ──────────────────────────────────────────────────────────────────────────────
# This code assumes the environment is represented as a 3-channel grid tensor:
#
#   grid: torch.Tensor of shape (3, H, W)
#
# with semantic meaning per channel index:
#   channel 0: occupancy / tile-type encoding
#       0 = empty
#       1 = wall
#       2 = red team occupancy marker
#       3 = blue team occupancy marker
#
#   channel 1: hp (hit points), typically in [0, MAX_HP]
#       For empty cells or walls, hp is typically 0 or irrelevant.
#
#   channel 2: agent_id, encoded such that:
#       -1 indicates no agent
#       >=0 indicates an agent is present (agent id or index)
#
# In addition, this function receives:
#   unit_map: torch.Tensor of shape (H, W), dtype int32 (as documented)
#       Values are stated as ∈ {-1, 1, 2}. From usage below, it is used to
#       disambiguate agent subtypes (e.g., soldier vs archer or similar).
#
# IMPORTANT:
#   This function does not validate the correctness of these encodings; it
#   assumes they are consistent with the simulation.
# ──────────────────────────────────────────────────────────────────────────────


def _generate_32_directions() -> torch.Tensor:
    """
    Generates 32 unique direction vectors, evenly spaced around a circle.

    Conceptual overview:
    --------------------
    A raycast in a grid-world typically "marches" outward from an origin position
    along a chosen direction vector. Here, the system uses a fixed set of 32
    directions. This is a common design for "sensor rays" in reinforcement
    learning agents, analogous to a 2D LIDAR or range-finder.

    Mathematical description:
    -------------------------
    Consider the unit circle parameterization:

        v(θ) = (cos θ, sin θ)

    for θ ∈ [0, 2π). If we sample 32 angles equally spaced:

        θ_k = 2π * k / 32   for k = 0, 1, ..., 31

    then {v(θ_k)} forms a set of 32 directions uniformly distributed around the
    circle (uniform in angle, not necessarily uniform in grid coverage due to
    discretization).

    Implementation details:
    -----------------------
    • np.linspace(0, 2π, 32, endpoint=False) produces 32 evenly spaced angles.
      endpoint=False ensures 2π is excluded so that θ=0 and θ=2π are not both
      present (which would duplicate the same direction).
    • dx = cos(angles), dy = sin(angles) produce the Cartesian components.
    • np.stack([dx, dy], axis=1) builds an array of shape (32, 2).
    • Finally we convert to torch.Tensor float32.

    Note:
    -----
    These are *continuous* directions. During ray marching, the code multiplies
    by step length and then casts to long indices, effectively discretizing the
    ray onto the grid.
    """
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    dx = np.cos(angles)
    dy = np.sin(angles)
    final_dirs = np.stack([dx, dy], axis=1)
    return torch.tensor(final_dirs, dtype=torch.float32)


# DIRS32 is a module-level constant tensor holding the 32 direction unit vectors.
# It is computed once on import, preventing repeated numpy computation per call.
DIRS32 = _generate_32_directions()

# ──────────────────────────────────────────────────────────────────────────────
# Semantic class encoding for the ray "first hit"
# ──────────────────────────────────────────────────────────────────────────────
# For each ray, we want to describe what the ray hits first. The model uses a
# one-hot encoding over 6 classes:
#
#   0 = none           (no hit within max range)
#   1 = wall           (occupancy channel indicates wall)
#   2 = red-soldier    (team red + unit subtype 1)
#   3 = red-archer     (team red + unit subtype 2)
#   4 = blue-soldier   (team blue + unit subtype 1)
#   5 = blue-archer    (team blue + unit subtype 2)
#
# This mapping is derived from later logic that combines:
#   • tile-type (grid[0]) indicating red vs blue occupancy marker, and
#   • unit_map indicating subtype 1 vs subtype 2.
#
# The number of classes is therefore 6.
_TYPE_CLASSES = 6


@torch.no_grad()
def raycast32_firsthit(
    pos_xy: torch.Tensor,               # (N,2) long
    grid: torch.Tensor,                  # (3,H,W)
    unit_map: torch.Tensor,              # (H,W) int32 ∈{-1, 1, 2}
    max_steps_each: Optional[torch.Tensor] = None,  # (N,) long — per-agent vision
) -> torch.Tensor:
    """
    Compute "first-hit" ray features for N agents using 32 rays per agent.

    Output feature specification:
    -----------------------------
    For each agent i and each ray r (32 directions), we emit exactly 8 features:

      [ onehot6(none, wall, red-sold, red-arch, blue-sold, blue-arch),
        dist_norm,
        hp_norm ]

    where:
      • onehot6 is a length-6 one-hot vector describing the type of the first
        object hit (or none).
      • dist_norm is the first-hit distance normalized by the agent’s max range
        (per-agent max_steps_each), with 0 if no hit.
      • hp_norm is the hit-point value at the hit cell (if an agent is hit),
        normalized by MAX_HP, with 0 if no hit.

    Shape:
    ------
    This yields:
      32 rays × 8 dims = 256 floats per agent

    and therefore returns a tensor of shape:
      (N, 256)

    Important performance notes:
    ----------------------------
    • The function is vectorized: it computes all rays for all agents across all
      step distances in one batched tensor operation, minimizing Python loops.
    • It uses @torch.no_grad() because these are environment observation features
      and do not require gradient tracking.
    • The approach can be seen as a dense "ray march" over a discretized grid,
      selecting the earliest hit.

    Subtle semantic note:
    ---------------------
    This method ray-marches in *continuous* direction vectors but ultimately
    converts to integer grid indices via .long(), so multiple step values can map
    to the same cell for shallow angles. This is an expected artifact of simple
    integer casting discretization.
    """

    # -------------------------------------------------------------------------
    # Device and dtype policy
    # -------------------------------------------------------------------------
    # The raycast must occur on the same device as the grid to avoid expensive
    # device transfers and to support GPU execution.
    device = grid.device
    #
    # The code chooses a floating-point dtype from config (if provided), else
    # defaults to float32. This dtype is used primarily for the final features
    # and the one-hot tensor.
    dtype = getattr(config, "TORCH_DTYPE", torch.float32)

    # -------------------------------------------------------------------------
    # Normalize/validate input tensor placement and types
    # -------------------------------------------------------------------------
    # pos_xy: expected shape (N, 2), representing per-agent integer position
    # coordinates (x, y). We explicitly cast to long and place on the grid device.
    pos_xy = pos_xy.to(dtype=torch.long, device=device)
    #
    # N = number of agents.
    N = int(pos_xy.size(0))
    #
    # H, W from grid shape (3, H, W).
    H, W = int(grid.size(1)), int(grid.size(2))

    # -------------------------------------------------------------------------
    # Determine maximum ray length (global and per-agent)
    # -------------------------------------------------------------------------
    # Global maximum steps: a hard upper bound on ray-marching distance.
    # This is a configuration knob so you can trade off:
    #   • larger range (more information)
    #   • higher compute and memory
    R_global = int(getattr(config, "RAYCAST_MAX_STEPS", 10))

    # Per-agent maximum range:
    # If not provided, every agent uses the global cap.
    # If provided, it is clamped into [0, R_global] for safety.
    if max_steps_each is None:
        max_steps_each = torch.full((N,), R_global, device=device, dtype=torch.long)
    else:
        max_steps_each = torch.clamp(
            max_steps_each.to(device=device, dtype=torch.long), 0, R_global
        )

    # -------------------------------------------------------------------------
    # Construct ray-marching coordinates for all agents, all rays, all steps
    # -------------------------------------------------------------------------
    # dirs: (1, 32, 2) direction vectors, broadcastable across agents and steps.
    dirs = DIRS32.to(device).view(1, 32, 2)                     # (1,32,2)

    # base: (N, 1, 1, 2) base positions, converted to float for multiplication.
    # Although positions are integer grid coordinates, float is used for the
    # directional step addition. Discretization occurs after computing coords.
    base = pos_xy.view(N, 1, 1, 2).float()                      # (N,1,1,2)

    # steps: (1, 1, S, 1) where S = R_global.
    # This constructs step distances 1..R_global inclusive.
    # Step distance begins at 1 because step=0 would correspond to the origin cell.
    steps = torch.arange(
        1, R_global + 1,
        device=device,
        dtype=torch.float32
    ).view(1, 1, R_global, 1)  # (1,1,S,1)

    # coords: (N, 32, S, 2)
    # For each agent, each ray direction, and each step distance, compute:
    #   coord = base + dir * step
    #
    # Then cast to long so the coordinate becomes integer indices for grid lookup.
    # This is a simplified rasterization of a ray in a discrete lattice.
    coords = (base + dirs.view(1, 32, 1, 2) * steps).long()     # (N,32,S,2)

    # Split coords into x and y components and clamp to grid bounds:
    # x ∈ [0, W-1], y ∈ [0, H-1].
    #
    # clamp_ is in-place to reduce temporary allocations.
    x = coords[..., 0].clamp_(0, W - 1)                          # (N,32,S)
    y = coords[..., 1].clamp_(0, H - 1)                          # (N,32,S)

    # -------------------------------------------------------------------------
    # Construct the "active" mask for per-agent max steps
    # -------------------------------------------------------------------------
    # step_ids: (1, 1, S) contains step indices 1..S (as long).
    # We use these to mask out steps beyond each agent's max range.
    step_ids = torch.arange(
        1, R_global + 1,
        device=device,
        dtype=torch.long
    ).view(1, 1, R_global)

    # active: (N, 1, S)
    # active[i, :, s] is True iff step_ids[s] <= max_steps_each[i].
    # This ensures that agents with shorter vision do not "see" beyond their range.
    active = step_ids <= max_steps_each.view(N, 1, 1)            # (N,1,S)

    # -------------------------------------------------------------------------
    # Gather occupancy and hp values along the ray paths
    # -------------------------------------------------------------------------
    # occ: (N, 32, S) occupancy/tile-type along ray cells.
    # hp:  (N, 32, S) hp channel along ray cells.
    #
    # The indexing grid[0][y, x] leverages advanced indexing:
    #   • y and x are broadcasted index tensors
    #   • the result is a gathered tensor of matching shape.
    occ = grid[0][y, x]                                          # (N,32,S)
    hp = grid[1][y, x]                                           # (N,32,S)

    # -------------------------------------------------------------------------
    # Identify hits: walls and agents
    # -------------------------------------------------------------------------
    # is_wall: True wherever occ indicates wall (== 1) AND within active range.
    is_wall = (occ == 1) & active

    # has_agent: True wherever the agent_id channel indicates presence (>= 0)
    # AND within active range.
    #
    # Note: has_agent is computed from grid[2] (agent_id), not from occ.
    # This is important: an "agent tile" might be represented in occ but
    # agent_id channel is the authoritative presence check here.
    has_agent = (grid[2][y, x] >= 0) & active

    # -------------------------------------------------------------------------
    # Determine the index (step position) of the first wall hit per ray
    # -------------------------------------------------------------------------
    # We want, for each (agent, ray), the earliest step where a wall appears.
    #
    # Strategy:
    #   1) is_wall.any(dim=-1) tells whether there exists any wall hit along steps.
    #   2) argmax over a float-cast boolean mask returns the first index where
    #      the mask is 1 (True), because booleans become {0.0, 1.0}.
    #
    # If there is no wall hit, idx_wall is set to -1.
    idx_wall = torch.where(
        is_wall.any(dim=-1),
        is_wall.to(torch.float32).argmax(dim=-1),
        -1
    )

    # -------------------------------------------------------------------------
    # Determine the index (step position) of the first agent hit per ray
    # -------------------------------------------------------------------------
    # Same method as idx_wall but for agent hits.
    idx_agent = torch.where(
        has_agent.any(dim=-1),
        has_agent.to(torch.float32).argmax(dim=-1),
        -1
    )

    # -------------------------------------------------------------------------
    # Prepare result holders:
    #   first_kind: which class the first hit belongs to (type code)
    #   first_idx:  step index (0-based along the steps dimension) of first hit
    # -------------------------------------------------------------------------
    # first_kind:
    #   Initialized to 0 (meaning "none"). This aligns with class encoding.
    # first_idx:
    #   Initialized to -1 meaning "no valid hit".
    first_kind = torch.full((N, 32), 0, dtype=torch.int64, device=device)
    first_idx = torch.full((N, 32), -1, dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Determine which kind of hit occurs first (wall vs agent)
    # -------------------------------------------------------------------------
    # There are three mutually exclusive cases:
    #   • both_hit: ray hits at least one wall AND at least one agent
    #   • only_wall: wall exists, agent does not
    #   • only_agent: agent exists, wall does not
    both_hit = (idx_wall >= 0) & (idx_agent >= 0)
    only_wall = (idx_wall >= 0) & ~both_hit
    only_agent = (idx_agent >= 0) & ~both_hit

    # If both wall and agent exist along the ray, we choose the earlier (smaller)
    # step index. Ties are resolved in favor of wall (<=).
    if both_hit.any():
        earlier_is_wall = (idx_wall <= idx_agent)

        # first_idx for those rays becomes whichever index is earlier.
        first_idx[both_hit] = torch.where(earlier_is_wall, idx_wall, idx_agent)[both_hit]

        # Encode wall hits as class 1.
        first_kind[both_hit & earlier_is_wall] = 1

        # Temporarily encode "agent hit" as -2 for later refinement.
        # Why temporary?
        #   Because determining which *agent class* was hit requires looking up
        #   both the team marker (from grid[0]) and unit subtype (unit_map),
        #   which happens below using the gathered hit location.
        first_kind[both_hit & ~earlier_is_wall] = -2  # temp code for agent

    # If only a wall hit exists, it's trivially the first hit.
    if only_wall.any():
        first_idx[only_wall] = idx_wall[only_wall]
        first_kind[only_wall] = 1

    # If only an agent hit exists, mark as temporary agent code -2.
    if only_agent.any():
        first_idx[only_agent] = idx_agent[only_agent]
        first_kind[only_agent] = -2  # temp code for agent

    # -------------------------------------------------------------------------
    # Resolve agent hits into the correct 2..5 class codes
    # -------------------------------------------------------------------------
    agent_mask = (first_kind == -2)

    # Only do the extra gather work if there is at least one ray whose first hit
    # is an agent.
    if agent_mask.any():
        # gather_idx: shape (N, 32, 1)
        # We need to gather the corresponding (x, y) coordinates at the first hit.
        #
        # first_idx is 0-based along the S dimension; gather expects indices
        # within valid range. clamp_min(0) prevents invalid negative indices
        # from crashing gather; these will be masked out later anyway.
        gather_idx = first_idx.clamp_min(0).unsqueeze(-1)

        # Gather y and x at the first-hit step for each (agent, ray).
        #
        # y, x have shape (N, 32, S). We gather along dim=2 (the step dimension).
        gather_y = torch.gather(y, 2, gather_idx).squeeze(-1)
        gather_x = torch.gather(x, 2, gather_idx).squeeze(-1)

        # t: occupancy marker at the hit cell (from channel 0).
        # u: unit subtype at the hit cell (from unit_map).
        #
        # They are cast to int32 explicitly, presumably to ensure consistent
        # comparison behavior and avoid dtype mismatches.
        t = grid[0][gather_y, gather_x].to(torch.int32)
        u = unit_map[gather_y, gather_x].to(torch.int32)

        # code: placeholder tensor for resolved class codes (int64).
        code = torch.zeros_like(t, dtype=torch.int64)

        # Class resolution logic:
        #
        # t == 2 means "red team occupancy marker"
        # t == 3 means "blue team occupancy marker"
        #
        # u == 1 means subtype-1 (documented as soldier in comment)
        # u == 2 means subtype-2 (documented as archer in comment)
        #
        # Thus:
        #   (t==2, u==1) => class 2: red-soldier
        #   (t==2, u==2) => class 3: red-archer
        #   (t==3, u==1) => class 4: blue-soldier
        #   (t==3, u==2) => class 5: blue-archer
        code[(t == 2) & (u == 1)] = 2
        code[(t == 2) & (u == 2)] = 3
        code[(t == 3) & (u == 1)] = 4
        code[(t == 3) & (u == 2)] = 5

        # Overwrite the temporary -2 agent markers with the resolved codes.
        first_kind[agent_mask] = code[agent_mask]

    # -------------------------------------------------------------------------
    # Compute normalized distance feature
    # -------------------------------------------------------------------------
    # den: per-agent normalization denominator = max_steps_each, clamped to >= 1.
    #
    # Why clamp_min(1)?
    #   If an agent has max_steps_each = 0, dividing by 0 would be invalid.
    #   In that case, active mask would be all False (no steps allowed), and
    #   valid hits should be 0 anyway. Using den>=1 avoids NaNs/infs.
    den = max_steps_each.clamp_min(1).to(torch.float32).view(N, 1)

    # dist_idx: the distance "in steps" to the hit, 1-based rather than 0-based.
    #
    # first_idx is 0-based index into the step dimension:
    #   first_idx = 0 means hit at step 1
    #   first_idx = 1 means hit at step 2
    # so we add 1.0 to convert to human-meaningful step count.
    dist_idx = first_idx.to(torch.float32) + 1.0

    # valid: mask (as float 0.0/1.0) indicating whether a hit exists (first_idx>=0).
    valid = (first_idx >= 0).to(torch.float32)

    # dist_norm:
    #   dist_norm = (dist_idx / den) if valid else 0
    #
    # This yields a value in approximately [0, 1], where:
    #   • 1/den corresponds to a hit at the first step,
    #   • 1 corresponds to a hit at exactly max_steps_each,
    #   • 0 corresponds to no hit.
    dist_norm = (dist_idx / den) * valid

    # -------------------------------------------------------------------------
    # Compute normalized HP feature at the first-hit location
    # -------------------------------------------------------------------------
    # hp_first: gather HP at first-hit step index.
    #
    # Note:
    #   We gather even when first_idx is -1, but clamp_min(0) ensures index is
    #   valid. Multiplying by valid then zeros out those entries.
    hp_first = torch.gather(
        hp,
        2,
        first_idx.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1) * valid

    # -------------------------------------------------------------------------
    # One-hot encode the first-hit class
    # -------------------------------------------------------------------------
    # onehot: shape (N, 32, 6)
    # Initialized to zeros.
    onehot = torch.zeros((N, 32, _TYPE_CLASSES), dtype=dtype, device=device)

    # idx_valid: ensure class indices fall into [0, 5].
    #
    # This is defensive: if first_kind had unexpected negative values, or values
    # outside the class range, scatter_ would error. Here:
    #   • negative values become 0
    #   • too-large values become 5
    #
    # In normal operation, first_kind should already be in {0..5}.
    idx_valid = first_kind.clamp(min=0, max=_TYPE_CLASSES - 1)

    # scatter_ places a 1.0 into the appropriate class slot for each (N, 32).
    #   dimension 2 is the class dimension.
    onehot.scatter_(2, idx_valid.unsqueeze(-1), 1.0)

    # -------------------------------------------------------------------------
    # Normalize HP by MAX_HP
    # -------------------------------------------------------------------------
    # max_hp is read from config. The expression "or 1.0" ensures it is never 0.
    # This avoids division-by-zero if MAX_HP is missing or accidentally set to 0.
    max_hp = float(getattr(config, "MAX_HP", 1.0)) or 1.0

    # hp_norm:
    #   normalized hp in [0, 1] (clamped) and cast to the desired dtype.
    hp_norm = (hp_first / max_hp).clamp(0.0, 1.0).to(dtype)

    # Ensure dist_norm uses the same dtype for consistent feature tensor.
    dist_norm = dist_norm.to(dtype)

    # -------------------------------------------------------------------------
    # Concatenate per-ray feature components into final per-agent vector
    # -------------------------------------------------------------------------
    # feat: shape (N, 32, 8)
    # ordering is exactly:
    #   [onehot6, dist_norm, hp_norm]
    feat = torch.cat(
        [onehot, dist_norm.unsqueeze(-1), hp_norm.unsqueeze(-1)],
        dim=-1
    )

    # reshape to (N, 256) = (N, 32*8) for downstream consumption (e.g., MLP).
    return feat.reshape(N, 32 * 8)