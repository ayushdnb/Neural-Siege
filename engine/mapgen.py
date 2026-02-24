from __future__ import annotations
# ──────────────────────────────────────────────────────────────────────────────
# Forward-annotation semantics
# ──────────────────────────────────────────────────────────────────────────────
# This directive causes all type annotations to be stored as strings rather than
# evaluated immediately. In a multi-module simulation codebase (often featuring
# circular imports between engine, registry, and environment utilities), this is
# a pragmatic choice to reduce import-order fragility.
#
# It has no impact on tensor operations or runtime behavior beyond annotation
# evaluation semantics.
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Standard library and typing imports
# ──────────────────────────────────────────────────────────────────────────────
# dataclass provides a clean, declarative way to define data containers.
from dataclasses import dataclass
#
# List and Tuple are used for explicit typing of lists of masks and rectangle
# bounds.
from typing import List, Tuple
#
# random is Python's standard pseudo-random generator. It is used here for
# sampling starting points, directions, and rectangle placements.
import random

# ──────────────────────────────────────────────────────────────────────────────
# PyTorch import
# ──────────────────────────────────────────────────────────────────────────────
# torch is used for:
#   • grid manipulation
#   • boolean masks representing zones
#   • device placement (CPU/GPU)
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Project configuration
# ──────────────────────────────────────────────────────────────────────────────
# This module is expected to define knobs such as:
#   • TORCH_DEVICE
#   • RANDOM_WALLS, WALL_SEG_MIN, WALL_SEG_MAX, WALL_AVOID_MARGIN
#   • HEAL_ZONE_COUNT, HEAL_ZONE_SIZE_RATIO
#   • CP_COUNT, CP_SIZE_RATIO
#
# Those knobs determine the number, size, and placement of procedural features.
import config

# ──────────────────────────────────────────────────────────────────────────────
# Grid channels documentation
# ──────────────────────────────────────────────────────────────────────────────
# The code operates on a grid tensor of shape (3, H, W) with these semantics:
#
#   grid[0] occupancy (tile kind / team marker)
#       0 empty
#       1 wall
#       2 red occupancy marker
#       3 blue occupancy marker
#
#   grid[1] hp
#       typically 0..MAX_HP; for walls/empty tiles, it is usually 0 or unused.
#
#   grid[2] agent_id
#       -1 indicates no agent is present
#       >=0 indicates an agent exists in that cell (registry slot/id)
#
# The zone masks below are deliberately kept as separate boolean tensors rather
# than being embedded into occupancy channels. This is a design decision that
# reduces churn in renderer/engine logic: special zones can be added/removed
# without changing core grid encoding or touching other subsystems.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Zones:
    """
    Immutable masks for special tiles (kept off-grid to avoid renderer/engine churn).

    Conceptual intent:
    ------------------
    Many simulations need additional “semantic layers” beyond occupancy/hp/id,
    such as:
      • healing fields
      • capture points
      • hazards
      • terrain modifiers

    One approach is to encode these into extra grid channels. However, adding
    channels often forces changes across:
      • rendering pipelines (new colors/sprites)
      • physics/engine loops (new rules)
      • serialization formats (more tensors)
      • observation feature builders (new inputs)

    This structure chooses an alternative: store zones as separate boolean masks.

    Data invariants:
    ---------------
    • Each mask has shape (H, W) and dtype torch.bool.
    • Masks should live on config.TORCH_DEVICE (or a specified device), ensuring
      they can be used in GPU-side computations without unnecessary transfers.

    Fields:
    -------
    heal_mask:
        True where tiles heal hit points (HP). Semantics are defined elsewhere.

    cp_masks:
        A list of boolean masks, each describing a distinct capture zone patch.

    Note on "immutability":
    -----------------------
    Python cannot enforce strict immutability of tensors; the term “immutable”
    here should be understood as a *convention*: after construction, these masks
    are intended not to be mutated in-place by downstream systems.
    """
    heal_mask: torch.Tensor                  # True where tiles heal HP
    cp_masks: List[torch.Tensor]             # list of boolean masks for capture patches

    @property
    def cp_count(self) -> int:
        """
        Convenience property: number of capture-point patches.

        Returns:
        --------
        The length of cp_masks, i.e. the number of distinct capture zone masks.
        """
        return len(self.cp_masks)


# ------------------------------------------------------------------------------
# Random thin gray walls (1-cell thick, meandering segments)
# ------------------------------------------------------------------------------
@torch.no_grad()
def add_random_walls(
    grid: torch.Tensor,
    n_segments: int = config.RANDOM_WALLS,
    seg_min: int = config.WALL_SEG_MIN,
    seg_max: int = config.WALL_SEG_MAX,
    avoid_margin: int = config.WALL_AVOID_MARGIN,
    allow_over_agents: bool = False,
) -> None:
    """
    Procedurally carve “thin” (1-cell thick) wall traces into a grid.

    Functional objective:
    ---------------------
    This function modifies grid *in-place* by writing walls into the occupancy
    channel (grid[0]) using the wall code 1.0. It constructs walls as a set of
    random meandering polyline-like segments on the grid.

    Intended call timing:
    ---------------------
    It is designed to be called *before* spawning agents. If called after agent
    spawn, and allow_over_agents is False, it avoids overwriting agent cells.

    Inputs:
    -------
    grid:
        Tensor of shape (3, H, W). The function asserts this minimal structure.

    n_segments:
        Number of independent wall traces to draw. Each trace is a random walk
        of length L (see seg_min/seg_max).

    seg_min, seg_max:
        Minimum/maximum segment length, in steps. Each segment draws L steps.

    avoid_margin:
        Prevents starting positions too close to boundaries. This helps avoid
        interfering with existing “outer border walls” often pre-generated by
        the grid maker. Additionally, the function clamps motion within the
        interior [1, W-2] × [1, H-2].

    allow_over_agents:
        If False, the function will not place walls on cells whose occupancy is
        currently 2.0 or 3.0 (agent/team markers). If True, it can overwrite.

    Side effects (in-place mutations):
    ---------------------------------
    For each wall cell placed at (x, y), the function sets:
      • grid[0, y, x] = 1.0   (occupancy becomes wall)
      • grid[1, y, x] = 0.0   (hp cleared)
      • grid[2, y, x] = -1.0  (agent_id cleared)

    This enforces a strong invariant:
      “A wall cell cannot simultaneously contain an agent or HP state.”
    """

    # Validate grid shape: must be at least 3 channels.
    assert grid.ndim == 3 and grid.size(0) >= 3, "grid must be (3,H,W)"

    # occ is a view into the occupancy channel for concise access.
    occ = grid[0]

    # Height and width are derived from the occupancy channel shape (H, W).
    H, W = int(occ.size(0)), int(occ.size(1))

    # --------------------------------------------------------------------------
    # Direction set: 8-connected neighborhood (dx, dy)
    # --------------------------------------------------------------------------
    # The random walk uses the standard 8 directions (including diagonals),
    # enabling “meandering” walls that can move diagonally as well as orthogonally.
    #
    # Storing this as a tensor on occ.device ensures:
    #   • no device mismatch if grid is on GPU
    #   • direction lookup is cheap and device-resident
    dirs8 = torch.tensor(
        [[ 0, -1],[ 1, -1],[ 1,  0],[ 1,  1],[ 0,  1],[-1,  1],[-1,  0],[-1, -1]],
        dtype=torch.long, device=occ.device
    )

    # --------------------------------------------------------------------------
    # Local helper: place a single wall cell with optional agent avoidance
    # --------------------------------------------------------------------------
    def _place_wall_cell(x: int, y: int) -> None:
        """
        Write a wall cell into the grid at (x, y), subject to bounds and policy.

        Bounds:
        -------
        The function guards against out-of-range coordinates.

        Agent-overwrite policy:
        -----------------------
        If allow_over_agents is False, and the current occupancy at (x, y)
        indicates an agent cell (2.0 or 3.0), the placement is skipped.

        Invariant enforcement:
        ----------------------
        When a wall is placed, the function clears hp and agent_id at that cell,
        ensuring internal consistency between grid channels.
        """
        if 0 <= x < W and 0 <= y < H:
            if not allow_over_agents:
                v = float(occ[y, x].item())
                if v in (2.0, 3.0):  # skip unit cells
                    return
            occ[y, x] = 1.0
            grid[1, y, x] = 0.0
            grid[2, y, x] = -1.0

    # --------------------------------------------------------------------------
    # Compute allowable start region for wall traces
    # --------------------------------------------------------------------------
    # The code respects an avoid_margin while also keeping at least a 1-cell
    # border clear. This is important because many grid generators pre-wall the
    # boundary to prevent agents from leaving the world.
    #
    # x0_min/x0_max and y0_min/y0_max define a valid inclusive range for random
    # starting points. Note that random.randint(a, b) includes both endpoints.
    x0_min, x0_max = max(1, avoid_margin), W - max(1, avoid_margin) - 1
    y0_min, y0_max = max(1, avoid_margin), H - max(1, avoid_margin) - 1

    # If the grid is too small for the margins, or no segments requested, exit.
    if x0_min >= x0_max or y0_min >= y0_max or n_segments <= 0:
        return

    # --------------------------------------------------------------------------
    # Draw each segment as a biased random walk
    # --------------------------------------------------------------------------
    # For each segment:
    #   • sample a random start (x, y)
    #   • sample a random length L
    #   • place the initial cell
    #   • then for L steps:
    #       - choose direction with turn bias
    #       - step and clamp to interior
    #       - place cell
    #       - occasionally introduce a gap
    #
    # The bias is important: without it, walls can jitter chaotically and
    # self-intersect aggressively; with bias, they appear as smoother "traces".
    for _ in range(max(0, int(n_segments))):
        x = random.randint(x0_min, x0_max)
        y = random.randint(y0_min, y0_max)

        # Segment length L is sampled uniformly in [seg_min, seg_max], but both
        # are clamped to at least 1 to avoid degenerate segments.
        L = random.randint(max(1, int(seg_min)), max(1, int(seg_max)))

        # Place the starting cell of the segment.
        _place_wall_cell(x, y)

        # last_dir is a direction index in [0, 7].
        last_dir = random.randrange(8)

        for _step in range(L):
            # ------------------------------------------------------------------
            # Direction choice with “small turn bias”
            # ------------------------------------------------------------------
            # With probability 0.70, keep the same direction (continue forward).
            # Otherwise, turn slightly by ±1 or ±2 steps in the direction index.
            #
            # This creates gentle bends rather than noisy zig-zags.
            if random.random() < 0.70:
                d = last_dir
            else:
                d = (last_dir + random.choice([-2, -1, 1, 2])) % 8
            last_dir = d

            # Extract dx, dy from dirs8 tensor. .item() returns a Python scalar.
            dx, dy = int(dirs8[d, 0].item()), int(dirs8[d, 1].item())

            # ------------------------------------------------------------------
            # Step and clamp inside interior
            # ------------------------------------------------------------------
            # The wall is constrained to the interior grid to avoid painting over
            # boundary walls:
            #   x ∈ [1, W-2], y ∈ [1, H-2]
            #
            # This also prevents index errors.
            x = max(1, min(W - 2, x + dx))
            y = max(1, min(H - 2, y + dy))

            # Place wall cell at the new position.
            _place_wall_cell(x, y)

            # ------------------------------------------------------------------
            # Occasional deliberate gaps
            # ------------------------------------------------------------------
            # With probability 0.05, we do nothing (leave a gap).
            #
            # Design rationale:
            #   Continuous walls can partition the map into disconnected regions,
            #   potentially creating unreachable areas or degenerately trapping
            #   agents. Introducing random gaps reduces the risk of total
            #   partition while preserving the qualitative “maze-like” structure.
            if random.random() < 0.05:
                # leave a deliberate gap (do nothing)
                pass



# ------------------------------------------------------------------------------
# Heal & Capture zones (rectangular patches, scaled to grid)
# ------------------------------------------------------------------------------
@torch.no_grad()
def make_zones(
    H: int,
    W: int,
    *,
    heal_count: int = config.HEAL_ZONE_COUNT,
    heal_ratio: float = config.HEAL_ZONE_SIZE_RATIO,
    cp_count: int = config.CP_COUNT,
    cp_ratio: float = config.CP_SIZE_RATIO,
    device: torch.device | None = None,
) -> Zones:
    """
    Create boolean masks for heal zones and capture zones.

    Conceptual overview:
    --------------------
    The simulation includes special areas:
      • Heal zones: cells that provide healing effects.
      • Capture zones: rectangular “patches” that can be captured/contested.

    Rather than encoding these into the occupancy channel, this function produces
    separate boolean masks, which can be consumed by:
      • the tick engine (to apply healing or capture rules)
      • the renderer (to visualize special tiles)
      • the observation builder (to expose zone info to agents)

    Parameters:
    -----------
    H, W:
        Grid height and width.

    heal_count:
        Number of heal rectangles to place.

    heal_ratio:
        Side-length ratio for heal rectangles relative to grid dimensions.
        The actual rectangle side lengths are computed as:
            h_side = round(heal_ratio * H)
            w_side = round(heal_ratio * W)
        then clamped to at least 1.

    cp_count:
        Number of capture rectangles to place.

    cp_ratio:
        Side-length ratio for capture rectangles (same computation strategy).

    device:
        Optional torch.device controlling where masks live. If None, defaults to
        config.TORCH_DEVICE.

    Output:
    -------
    Zones object containing:
      • heal_mask: shape (H, W), dtype bool
      • cp_masks: list of cp_count boolean masks, each shape (H, W)

    Overlap semantics:
    ------------------
    The docstring states that masks are "non-overlapping where possible", but
    the implementation does not enforce non-overlap constraints; it simply
    samples rectangles independently. Therefore, overlap may occur:
      • heal zones may overlap with capture zones
      • capture zones may overlap each other
    and the comment clarifies that overlap is acceptable (semantics additive).
    """

    # Default device: if none provided, use config.TORCH_DEVICE.
    device = device or config.TORCH_DEVICE

    # heal_mask is a single boolean mask accumulating all heal rectangles.
    heal_mask = torch.zeros((H, W), dtype=torch.bool, device=device)

    # cp_masks is a list of distinct capture zone masks.
    cp_masks: List[torch.Tensor] = []

    # --------------------------------------------------------------------------
    # Helper: sample a rectangle fully contained within the interior
    # --------------------------------------------------------------------------
    def _sample_rect(h_side: int, w_side: int) -> Tuple[int, int, int, int]:
        """
        Sample an axis-aligned rectangle within the interior region of the grid.

        Returns:
        --------
        (y0, y1, x0, x1) defining a slice:
            rows y0:y1 and cols x0:x1

        Boundary policy:
        ----------------
        The sampler keeps a 1-cell border clear, consistent with the assumption
        that outer walls exist around the perimeter.

        Implementation detail:
        ----------------------
        random.randint(a, b) is inclusive at both ends. The max(...) ensures the
        upper bound is at least 1 to prevent ValueError for small grids.
        """
        # keep 1-cell border clear (outer walls)
        x0 = random.randint(1, max(1, W - w_side - 2))
        y0 = random.randint(1, max(1, H - h_side - 2))
        return y0, y0 + h_side, x0, x0 + w_side  # (y0, y1, x0, x1)

    # --------------------------------------------------------------------------
    # Heal zones: write True into heal_mask for each sampled rectangle
    # --------------------------------------------------------------------------
    if heal_count > 0 and heal_ratio > 0.0:
        # Compute rectangle side lengths proportional to grid dimensions.
        # round(...) allows proportional sizes to scale naturally with grid size.
        h_side = max(1, int(round(heal_ratio * H)))
        w_side = max(1, int(round(heal_ratio * W)))

        for _ in range(int(heal_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)

            # Boolean slicing assignment sets the region to True.
            # Because heal_mask accumulates all heal zones, repeated assignments
            # naturally union together (logical OR).
            heal_mask[y0:y1, x0:x1] = True

    # --------------------------------------------------------------------------
    # Capture zones: create a separate mask per rectangle and append to list
    # --------------------------------------------------------------------------
    if cp_count > 0 and cp_ratio > 0.0:
        h_side = max(1, int(round(cp_ratio * H)))
        w_side = max(1, int(round(cp_ratio * W)))

        for _ in range(int(cp_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)

            # Each capture zone is a distinct mask so the engine can track
            # ownership/contest state per zone independently (implied by design).
            m = torch.zeros((H, W), dtype=torch.bool, device=device)
            m[y0:y1, x0:x1] = True
            cp_masks.append(m)

    # Package into the Zones dataclass and return.
    return Zones(heal_mask=heal_mask, cp_masks=cp_masks)