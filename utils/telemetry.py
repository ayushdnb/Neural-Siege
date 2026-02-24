# Infinite_War_Simulation/utils/telemetry.py
"""
Telemetry module for recording agent life events, lineage, and damage.

This module is deliberately "boring" in a good way:
- It is **append-friendly** (doesn't destroy previous data when resuming).
- It is **analysis-friendly** (CSV snapshots + JSONL event logs are easy to parse).
- It is **safe in long runs** (atomic writes to avoid partial/corrupt files).
- It is **config-driven** (no new config system; uses existing config knobs).

What this module tracks (high-level)
------------------------------------
1) AgentLife snapshot (CSV):
   - One row per agent_id, including birth tick, death tick, totals like kills/damage.
   - Written periodically as a snapshot (overwrite) using atomic replace.

2) LineageEdges (CSV append):
   - Parent-child relationships (parent_id -> child_id) with tick recorded.

3) Events (JSONL chunked, append-ish):
   - Birth/death/damage/kill/resume events.
   - Buffered in memory and flushed into chunk files: events_000001.jsonl, etc.

4) Optional "sidecar" files (config gated):
   - run_meta.json: run metadata useful for analysis / reproducibility
   - agent_static.csv: static per-agent attributes (max_hp, atk, vision, brain_type, param_count)
   - tick_summary.csv: low-frequency time series of alive counts, mean HP, kills/deaths/dmg totals

Scientific / engineering rationale
----------------------------------
Telemetry for simulations and RL is not optional if you want reproducibility.
Long runs fail in subtle ways (NaNs, silent corruption, partially-written files).
This module is designed to:
- Detect anomalies early (validate())
- Avoid partial reads (atomic writes)
- Keep data volume manageable (event chunking)
- Keep data interoperable (CSV + JSONL)

Important constraints (project-specific)
----------------------------------------
- The simulation uses an "agent registry" where `agent_data` is stored as float dtype.
  That means categorical fields (agent_id, team_id, unit_id, alive flag) are **floats**,
  so telemetry must carefully convert to integers for stable logging.

- Team encoding is assumed to use IDs {2,3} for Red/Blue (based on other code).
  This file mostly records what it sees; it does not enforce game rules.

- This module must never mutate simulation state (pure observation).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # Tuple imported but may be used elsewhere.
import config


# =============================================================================
# Utility helpers (pure functions)
# =============================================================================

def _to_int(x: Any) -> int:
    """
    Convert an arbitrary value into an int in a robust way.

    Why this exists
    ---------------
    In this project, the agent registry stores agent attributes in a float tensor
    (likely for GPU/vectorization convenience).
    That means identifiers that are conceptually integers (agent_id, team_id, etc.)
    may appear as:
        2.0, 3.0, 117.0
    Additionally, when reading from CSV/JSON, values might be strings.

    Strategy
    --------
    1) First try `int(x)` directly.
    2) If that fails, try `int(float(x))`.
       - Handles strings like "12", "12.0", and numeric-like values.

    Note:
    -----
    `int(float("12.9"))` becomes 12 (floor toward zero). That is fine here because
    IDs should be integral already; if they are not, it is a sign of corruption
    upstream, and validation/anomaly paths should catch it.
    """
    try:
        return int(x)
    except Exception:
        # Fallback: convert via float to handle strings/other numeric types.
        return int(float(x))


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write text to `path`.

    Atomic write: what it means
    ---------------------------
    "Atomic" here means: readers will see either
    - the old full file, OR
    - the new full file,
    but never a partially-written file.

    How it's achieved
    -----------------
    1) Write content into a temporary file in the same directory.
    2) Use `os.replace(tmp, path)` which is an atomic rename on most OS/filesystems.

    Why it matters
    --------------
    During long runs you may:
    - tail files
    - run analysis scripts concurrently
    - resume from checkpoints
    A non-atomic write can leave half a CSV/JSON file which breaks parsers.

    Constraints
    -----------
    - The temp file uses a ".tmp" suffix appended to the original suffix.
    - We ensure the parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _parse_validate_level(v: Any, default: int = 2) -> int:
    """
    Parse and validate TELEMETRY_VALIDATE_LEVEL.

    Intended config behavior
    ------------------------
    TELEMETRY_VALIDATE_LEVEL can be specified as:
    - number: 0, 1, 2
    - string aliases:
        "off" or "0"     -> 0 (no validation)
        "basic" or "1"   -> 1
        "strict" or "2"  -> 2 (default)

    Returns
    -------
    An integer in {0,1,2}. If parsing fails, returns `default`.

    Why this is useful
    ------------------
    Config values may arrive as strings from env vars or config files.
    This makes the system tolerant and reduces "config gotcha" bugs.
    """
    # If it's already a number, try to convert to int.
    if isinstance(v, (int, float)):
        try:
            return int(v)
        except Exception:
            return int(default)

    # If it's a string, handle common aliases.
    if isinstance(v, str):
        s = v.strip().lower()
        m = {"off": 0, "0": 0, "basic": 1, "1": 1, "strict": 2, "2": 2}
        if s in m:
            return int(m[s])
        try:
            return int(s)
        except Exception:
            return int(default)

    # Fallback to default.
    return int(default)


# =============================================================================
# Main session class
# =============================================================================

class TelemetrySession:
    """
    TelemetrySession is a stateful recorder object used across simulation ticks.

    It stores:
    - In-memory state (`_life`, `_offspring_count`, `_events_buf`)
    - Output file paths (CSV/JSONL)
    - Config-driven behavior flags

    It provides a small "public API" that the engine calls:
    - attach_context(...)
    - write_run_meta(...)
    - record_resume(...)
    - bootstrap_from_registry(...)
    - ingest_spawn_meta(...)
    - record_birth(...)
    - record_damage_*(...)
    - record_kills(...)
    - record_deaths(...)
    - on_tick_end(...)
    - close()

    Design philosophy
    -----------------
    - The simulation engine does not need to know telemetry internals.
    - Telemetry does not control the simulation; it only observes and logs.
    - Output formats are simple and robust.
    """

    def __init__(self, run_dir: Path) -> None:
        """
        Initialize telemetry and prepare output structure.

        Parameters
        ----------
        run_dir:
            The root directory for this simulation run (e.g., results/sim_YYYY-MM-DD_...).
            Telemetry outputs go under: run_dir/telemetry/
        """
        # ---------------------------------------------------------------------
        # Core enable switch.
        # If not enabled, most methods become no-ops (return early).
        # ---------------------------------------------------------------------
        self.enabled: bool = bool(getattr(config, "TELEMETRY_ENABLED", False))
        self.run_dir = Path(run_dir)

        # ---------------------------------------------------------------------
        # Schema versioning (important for downstream analysis compatibility)
        # ---------------------------------------------------------------------
        # NOTE: The original code sets schema_version as int here...
        self.schema_version: int = int(getattr(config, "TELEMETRY_SCHEMA_VERSION", 2))

        # ---------------------------------------------------------------------
        # Reuse existing knobs (do not invent a second config system).
        # ---------------------------------------------------------------------
        self.tag: str = str(getattr(config, "TELEMETRY_TAG", ""))
        self.event_chunk_size: int = int(getattr(config, "TELEMETRY_EVENT_CHUNK_SIZE", 50_000))
        self.flush_every: int = int(getattr(config, "TELEMETRY_PERIODIC_FLUSH_EVERY", 250))

        # Events format: this implementation supports JSONL only.
        self.events_format: str = str(getattr(config, "TELEMETRY_EVENTS_FORMAT", "jsonl"))
        self.events_gzip: bool = bool(getattr(config, "TELEMETRY_EVENTS_GZIP", False))  # reserved for future; no deps

        # Which event types to emit
        self.log_births: bool = bool(getattr(config, "TELEMETRY_LOG_BIRTHS", True))
        self.log_deaths: bool = bool(getattr(config, "TELEMETRY_LOG_DEATHS", True))
        self.log_damage: bool = bool(getattr(config, "TELEMETRY_LOG_DAMAGE", True))
        self.log_kills: bool = bool(getattr(config, "TELEMETRY_LOG_KILLS", True))
        self.damage_mode: str = str(getattr(config, "TELEMETRY_DAMAGE_MODE", "victim_sum"))

        # ---------------------------------------------------------------------
        # Validation knobs (detect internal inconsistencies)
        # ---------------------------------------------------------------------
        self.validate_level: int = _parse_validate_level(
            getattr(config, "TELEMETRY_VALIDATE_LEVEL", 2),
            default=2,
        )
        self.validate_every: int = int(getattr(config, "TELEMETRY_VALIDATE_EVERY", 1000))
        self.abort_on_anomaly: bool = bool(getattr(config, "TELEMETRY_ABORT_ON_ANOMALY", False))

        # ---------------------------------------------------------------------
        # Output layout
        # ---------------------------------------------------------------------
        self.telemetry_dir = self.run_dir / "telemetry"
        self.events_dir = self.telemetry_dir / "events"
        self.agent_life_path = self.telemetry_dir / "agent_life.csv"
        self.lineage_edges_path = self.telemetry_dir / "lineage_edges.csv"

        # Sidecar files added by patch (config gated)
        self.run_meta_path = self.telemetry_dir / "run_meta.json"
        self.agent_static_path = self.telemetry_dir / "agent_static.csv"
        self.tick_summary_path = self.telemetry_dir / "tick_summary.csv"

        # Create directories (safe even if they already exist).
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # Additive sidecars (config gated) – note: potential type conflict below.
        # ---------------------------------------------------------------------
        # IMPORTANT OBSERVATION (engineering note):
        # The code below overwrites `self.schema_version` (int) with a string like "v2".
        # That is a semantic type change.
        #
        # It appears intentional due to a "patch" comment, but it causes subtle behavior:
        # - Some event emitters cast schema_version to int with int(getattr(...,2)),
        #   which will FAIL for "v2".
        #
        # However: YOU asked "do not change any code". So we document, not modify.
        self.schema_version: str = str(getattr(config, "TELEMETRY_SCHEMA_VERSION", "v2"))

        # FIX comment: Renamed attribute to avoid conflict with write_run_meta method
        # - The method is write_run_meta(...)
        # - The flag is do_write_run_meta (boolean).
        self.do_write_run_meta: bool = bool(getattr(config, "TELEMETRY_WRITE_RUN_META", True))

        self.write_agent_static: bool = bool(getattr(config, "TELEMETRY_WRITE_AGENT_STATIC", False))
        self.tick_summary_every: int = int(getattr(config, "TELEMETRY_TICK_SUMMARY_EVERY", 0))

        # ---------------------------------------------------------------------
        # Context references (never mutates; just pointers for reading data)
        # ---------------------------------------------------------------------
        self._registry: Any = None
        self._stats: Any = None

        # ---------------------------------------------------------------------
        # AgentStatic deduplication: track which agent_ids already written
        # ---------------------------------------------------------------------
        self._static_written: set[int] = set()
        if self.write_agent_static and self.agent_static_path.exists():
            # If agent_static.csv exists (e.g., resume), rehydrate `_static_written`
            # to avoid duplicating static rows.
            try:
                with self.agent_static_path.open("r", newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if "agent_id" in row and row["agent_id"] not in (None, ""):
                            # agent_id might be stored as "123" or "123.0".
                            self._static_written.add(int(float(row["agent_id"])))
            except Exception:
                # Fail-safe: if parsing fails, disable dedup state.
                # Worst-case: may re-emit some static rows; still usable for analysis.
                self._static_written = set()

        # ---------------------------------------------------------------------
        # Main AgentLife state
        # ---------------------------------------------------------------------
        # `_life` stores per-agent evolving stats and timestamps.
        # Key: agent_id (int)
        # Value: dict of fields (born_tick, death_tick, totals, etc.)
        self._life: Dict[int, Dict[str, Any]] = {}

        # Offspring count stored separately to keep `_life` simpler.
        self._offspring_count: Dict[int, int] = {}

        # ---------------------------------------------------------------------
        # Event buffering and chunking
        # ---------------------------------------------------------------------
        self._events_buf: List[Dict[str, Any]] = []
        self._chunk_idx: int = self._discover_next_chunk_idx()

        # ---------------------------------------------------------------------
        # Resume support: rehydrate agent life snapshot
        # ---------------------------------------------------------------------
        if self.enabled:
            self._rehydrate_agent_life()

        # Track last tick seen for graceful shutdown final summary.
        self._last_tick_seen: Optional[int] = None

        # Ensure lineage edges header exists if file is new.
        if not self.lineage_edges_path.exists():
            self._append_csv_rows(
                self.lineage_edges_path,
                fieldnames=["tick", "parent_id", "child_id"],
                rows=[],
            )

        # Create headers for agent_static and tick_summary if enabled and new.
        if self.write_agent_static and (not self.agent_static_path.exists()):
            self._append_csv_rows(
                self.agent_static_path,
                fieldnames=[
                    "agent_id", "team_id", "unit_type", "brain_type", "param_count",
                    "max_hp", "base_atk", "vision_range",
                    "spawn_tick", "parent_id", "spawn_reason",
                ],
                rows=[],
            )

        if self.tick_summary_every > 0 and (not self.tick_summary_path.exists()):
            self._append_csv_rows(
                self.tick_summary_path,
                fieldnames=[
                    "tick", "elapsed_s",
                    "red_alive", "blue_alive", "mean_hp_red", "mean_hp_blue",
                    "red_kills", "blue_kills", "red_deaths", "blue_deaths",
                    "red_dmg_dealt", "blue_dmg_dealt", "red_dmg_taken", "blue_dmg_taken",
                ],
                rows=[],
            )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _discover_next_chunk_idx(self) -> int:
        """
        Determine the next events chunk index by scanning existing chunk files.

        Why this is needed
        ------------------
        If you resume a run and keep logging into the same run directory,
        you must not overwrite previous events_* files. Instead, you continue with
        the next index.

        Implementation
        --------------
        - Search for files matching: events_*.jsonl
        - Parse the numeric suffix and return max+1
        - If parsing fails for a file, ignore it.
        """
        max_idx = -1
        for p in self.events_dir.glob("events_*.jsonl"):
            try:
                stem = p.stem  # e.g., "events_000001"
                idx = int(stem.split("_", 1)[1])
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        return max_idx + 1

    def _rehydrate_agent_life(self) -> None:
        """
        Load agent_life.csv into memory if it exists.

        Purpose
        -------
        If you resume from a checkpoint or restart the process and reuse the same run_dir,
        existing agents might already have recorded births/deaths/totals.
        This method repopulates `_life` and `_offspring_count`.

        Parsing strategy
        ---------------
        - Uses csv.DictReader (header-driven).
        - Attempts to parse known fields into appropriate types.
        - Gracefully ignores parse errors (fail-safe telemetry).
        """
        if not self.agent_life_path.exists():
            return

        try:
            with self.agent_life_path.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not row:
                        continue

                    aid_s = (row.get("agent_id") or "").strip()
                    if aid_s == "":
                        continue

                    aid = int(aid_s)
                    rec: Dict[str, Any] = {}

                    # Basic integer/string fields.
                    # - slot_id/team/unit_type are numeric but might appear as "12.0"
                    # - notes is free-form text
                    for k in ("slot_id", "team", "unit_type", "notes"):
                        v = row.get(k, "")
                        if v != "":
                            try:
                                rec[k] = int(float(v)) if k != "notes" else str(v)
                            except Exception:
                                rec[k] = str(v)

                    # Tick-like integer fields.
                    for k in ("born_tick", "death_tick", "parent_id", "kills_total"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = int(float(v))
                            except Exception:
                                # If parse fails, skip silently (telemetry should not kill the sim).
                                pass

                    # Float totals for damage fields.
                    for k in ("damage_dealt_total", "damage_taken_total"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = float(v)
                            except Exception:
                                pass

                    self._life[aid] = rec

                    # Offspring count stored separately.
                    oc = (row.get("offspring_count") or "").strip()
                    if oc != "":
                        try:
                            self._offspring_count[aid] = int(float(oc))
                        except Exception:
                            pass
        except Exception as e:
            self._anomaly(f"rehydrate agent_life failed: {e}")

    def _anomaly(self, msg: str) -> None:
        """
        Handle a detected anomaly.

        Policy
        ------
        - If `abort_on_anomaly` is True: raise AssertionError immediately.
          This is useful for debugging; it makes the simulation fail fast.
        - Otherwise: suppress output (silent) and continue.
          This is useful for production runs where you prefer not to spam logs.

        Note
        ----
        Silent anomaly handling can hide issues. A common compromise is:
        - keep silent during run
        - write anomaly events into telemetry (not implemented here)
        """
        if self.abort_on_anomaly:
            raise AssertionError(msg)

        # FIX: Silence console output as requested.
        # print(f"[telemetry] ANOMALY: {msg}")
        pass

    def _require_birth(self, agent_id: int, context: str) -> None:
        """
        Ensure that `agent_id` exists in `_life`.

        This is a consistency check: damage/kill/death events should only happen
        for agents that were born (record_birth) at some point.

        If missing, we treat it as an anomaly.
        """
        if agent_id not in self._life:
            self._anomaly(f"{context}: missing birth for agent_id={agent_id}")

    def _append_csv_rows(self, path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        """
        Append rows to a CSV file, writing the header if the file does not exist.

        Why append?
        -----------
        - For lineage_edges.csv and tick_summary.csv, we want a growing log.
        - For agent_static.csv, we want to append new unique agent rows.

        Technical notes
        ---------------
        - newline="" is recommended for csv module to avoid blank lines on Windows.
        - DictWriter enforces consistent column order via fieldnames.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()

        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)

    def _emit_event(self, ev: Dict[str, Any]) -> None:
        """
        Buffer a single event dict and flush if the buffer reaches chunk size.

        Event buffering rationale
        -------------------------
        Writing to disk on every event can be expensive (especially on Windows),
        and it can degrade simulation performance.
        Buffering batches events into memory and flushes them as a chunk.

        Format policy
        -------------
        - JSONL only in this implementation.
        - If config requests something else, we degrade safely to JSONL and annotate notes.
        """
        if not self.enabled:
            return

        if self.events_format.lower() != "jsonl":
            # Degrade safely to jsonl; do not invent other writers.
            ev = dict(ev)
            ev["notes"] = (ev.get("notes", "") + " events_format_forced_jsonl").strip()

        self._events_buf.append(ev)
        if len(self._events_buf) >= self.event_chunk_size:
            self._flush_event_chunk()

    def _flush_event_chunk(self) -> None:
        """
        Flush buffered events into a new events_XXXXXX.jsonl file (atomic overwrite).

        Chunking scheme
        --------------
        Each chunk is its own file:
            telemetry/events/events_000000.jsonl
            telemetry/events/events_000001.jsonl
            ...

        Pros:
        - Avoid huge single-file logs.
        - Easy parallel processing later.
        - Resume-safe if chunk index is discovered from disk.

        Cons:
        - Many small files (still generally fine).
        """
        if not self._events_buf:
            return

        out = self.events_dir / f"events_{self._chunk_idx:06d}.jsonl"

        # JSON Lines: one JSON object per line.
        # This is robust because a parser can read line-by-line.
        lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in self._events_buf) + "\n"
        _atomic_write_text(out, lines)

        self._events_buf.clear()
        self._chunk_idx += 1

    def _flush_agent_life_snapshot(self) -> None:
        """
        Write a full snapshot of `_life` to agent_life.csv (overwrite atomically).

        Why snapshot overwrite instead of append?
        -----------------------------------------
        AgentLife is a "current state" table: we want the latest known totals per agent.
        The simplest way is to write the whole table periodically.

        Atomic overwrite avoids partial snapshots.

        Field design
        ------------
        - Includes core life history (born_tick, death_tick, lifespan_ticks)
        - Includes totals (kills, damage)
        - Includes several "future extension" columns (moves_attempted, reward_total, etc.)
          which may remain blank today.
        """
        fieldnames = [
            "agent_id", "slot_id", "team", "unit_type",
            "born_tick", "death_tick", "lifespan_ticks", "parent_id", "offspring_count",
            "kills_total", "damage_dealt_total", "damage_taken_total", "notes",
            # Additional fields reserved for future extensions
            "moves_attempted", "moves_success", "moves_blocked_wall", "moves_blocked_occupied",
            "cells_walked_total_l1", "reward_total", "death_cause", "last_seen_tick",
        ]

        rows: List[Dict[str, Any]] = []
        for aid, rec in sorted(self._life.items(), key=lambda kv: kv[0]):
            r = dict(rec)
            r["agent_id"] = aid
            r["offspring_count"] = int(self._offspring_count.get(aid, 0))

            # lifespan_ticks is only computable if both born and death ticks exist.
            bt = r.get("born_tick")
            dt = r.get("death_tick")
            if bt is not None and dt is not None:
                r["lifespan_ticks"] = int(dt) - int(bt)
            else:
                r["lifespan_ticks"] = ""

            # Emit exactly the defined fields; missing fields become "" for CSV cleanliness.
            rows.append({k: r.get(k, "") for k in fieldnames})

        # CSV serialize using StringIO so we can atomic-write the complete content.
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

        _atomic_write_text(self.agent_life_path, buf.getvalue())

    def validate(self) -> None:
        """
        Run consistency checks on `_life` according to validation level.

        Levels
        ------
        0: off
        1: basic
        2: strict

        Current implementation note
        ---------------------------
        The method currently runs the same checks regardless of 1 vs 2,
        but the config architecture supports differentiating in the future.

        What we check
        -------------
        - death_tick >= born_tick
        - kills_total is not negative
        - damage totals are not negative

        Why negative totals matter
        --------------------------
        Totals represent cumulative sums of non-negative quantities.
        A negative value suggests overflow, wrong sign, or memory corruption.
        """
        if self.validate_level <= 0:
            return

        for aid, r in self._life.items():
            bt = r.get("born_tick")
            dt = r.get("death_tick")

            if bt is not None and dt is not None and int(dt) < int(bt):
                self._anomaly(f"agent_id={aid}: death_tick < born_tick ({dt} < {bt})")

            if r.get("kills_total", 0) < 0:
                self._anomaly(f"agent_id={aid}: negative kills_total")

            if r.get("damage_dealt_total", 0.0) < 0:
                self._anomaly(f"agent_id={aid}: negative damage_dealt_total")

            if r.get("damage_taken_total", 0.0) < 0:
                self._anomaly(f"agent_id={aid}: negative damage_taken_total")

    # =============================================================================
    # Public API called from main/tick
    # =============================================================================

    def attach_context(self, registry: Any, stats: Any) -> None:
        """
        Attach references to simulation objects that telemetry can read.

        Why this method exists
        ----------------------
        Telemetry sometimes needs access to:
        - registry.agent_data tensor
        - registry.brains list (for brain_type and parameter counts)
        - stats (for tick summary metrics)

        By attaching these references, we avoid deep refactors of the engine:
        telemetry can be "plugged in" with minimal integration points.

        Important:
        ----------
        - Telemetry must never mutate these objects.
        - These are observational references only.
        """
        self._registry = registry
        self._stats = stats

    def write_run_meta(self, meta: Dict[str, Any]) -> None:
        """
        Write run-level metadata to run_meta.json (atomic overwrite).

        Examples of meta
        ----------------
        - configuration summary
        - command line args
        - environment info (GPU name, torch version)
        - seeds
        - experiment name

        This data is crucial for reproducibility.

        Implementation details
        ----------------------
        - Schema version is inserted if missing.
        - Git commit is optionally added from env var GIT_COMMIT.
        """
        if not self.enabled or not self.do_write_run_meta:
            return

        out = dict(meta or {})
        out.setdefault("schema_version", self.schema_version)
        out.setdefault("git_commit", os.getenv("GIT_COMMIT", None))

        _atomic_write_text(self.run_meta_path, json.dumps(out, indent=2, sort_keys=True))

    def record_resume(self, tick: int, checkpoint_path: str) -> None:
        """
        Emit a 'resume' event when loading from a checkpoint.

        Why record resumes?
        -------------------
        In long runs, you may stop/restart/resume many times.
        Knowing exactly when and from where the run resumed is vital for analysis.
        """
        if not self.enabled:
            return
        self._emit_event({
            "tick": int(tick),
            "type": "resume",
            "checkpoint_path": str(checkpoint_path),
        })

    def bootstrap_from_registry(self, registry: Any, tick: int, note: str = "bootstrap") -> None:
        """
        Seed births for agents that already exist at startup.

        Use-case
        --------
        If the simulation begins with pre-spawned agents (e.g., initial population),
        telemetry needs a birth record for them to maintain consistency.

        Implementation detail
        ---------------------
        The registry is expected to have:
        - registry.agent_data tensor
        - optionally registry.agent_uids mapping slots->unique ids

        We find alive agents:
          alive_mask = data[:, COL_ALIVE] > 0.5
        and for each alive slot:
          - determine agent_id
          - record birth (allow_existing=True)
          - maybe write agent static attributes
        """
        if not self.enabled:
            return

        from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_UNIT, COL_AGENT_ID
        data = registry.agent_data
        alive_mask = (data[:, COL_ALIVE] > 0.5)

        # nonzero(...) returns indices where alive_mask is True.
        alive_slots = alive_mask.nonzero(as_tuple=False).view(-1).tolist()

        for slot in alive_slots:
            # Two possible sources of agent id:
            # - registry.agent_uids (preferred stable mapping)
            # - agent_data column COL_AGENT_ID (fallback)
            if hasattr(registry, "agent_uids"):
                aid = int(registry.agent_uids[slot].item())
            else:
                aid = _to_int(data[slot, COL_AGENT_ID].item())

            team = _to_int(data[slot, COL_TEAM].item())
            unit = _to_int(data[slot, COL_UNIT].item())

            self.record_birth(
                tick=tick,
                agent_id=aid,
                slot_id=int(slot),
                team=team,
                unit_type=unit,
                parent_id=None,
                notes=note,
                allow_existing=True,
            )

            # Also write static attributes if enabled.
            self._maybe_write_agent_static(
                tick=int(tick),
                slot_id=int(slot),
                agent_id=int(aid),
                team_id=int(team),
                unit_type=int(unit),
                parent_id=None,
                spawn_reason=str(note),
            )

    def record_birth(
        self,
        tick: int,
        agent_id: int,
        slot_id: int,
        team: int,
        unit_type: int,
        parent_id: Optional[int],
        notes: str = "",
        allow_existing: bool = False,
    ) -> None:
        """
        Record an agent birth (creation).

        Parameters
        ----------
        tick:
            Current simulation tick when the agent appears.
        agent_id:
            Global unique id of the agent.
        slot_id:
            The registry slot index where this agent currently resides.
        team:
            Team identifier (commonly 2=red, 3=blue).
        unit_type:
            Unit type identifier (project-specific).
        parent_id:
            Parent agent id if this agent was spawned from another agent (lineage).
        notes:
            Free-form string for provenance (e.g., "respawn", "bootstrap").
        allow_existing:
            If True, update existing record rather than treating as a duplicate.
            Used during bootstrap/resume scenarios.

        Consistency rules
        -----------------
        - If agent_id already exists in `_life` and allow_existing is False -> anomaly.
        - When allow_existing is True:
            - preserve existing born_tick if already present
            - otherwise set born_tick to current tick
        """
        if not self.enabled:
            return

        if (agent_id in self._life) and not allow_existing:
            self._anomaly(f"birth: duplicate agent_id={agent_id}")
            return

        rec = self._life.get(agent_id, {})
        rec.update({
            "slot_id": int(slot_id),
            "team": int(team),
            "unit_type": int(unit_type),

            # born_tick handling:
            # - bootstrap should not overwrite old born_tick if present
            "born_tick": (rec.get("born_tick", int(tick)) if allow_existing else int(tick)),

            # keep any previous death_tick if rehydrated
            "death_tick": rec.get("death_tick", None),

            "parent_id": (int(parent_id) if parent_id is not None else None),

            # totals should persist if record exists
            "kills_total": int(rec.get("kills_total", 0)),
            "damage_dealt_total": float(rec.get("damage_dealt_total", 0.0)),
            "damage_taken_total": float(rec.get("damage_taken_total", 0.0)),

            # notes: if notes provided, store it; else keep old
            "notes": str(notes or rec.get("notes", "")),
        })
        self._life[agent_id] = rec

        # If lineage info exists, record parent->child edge and update offspring counts.
        if parent_id is not None:
            self._offspring_count[int(parent_id)] = int(self._offspring_count.get(int(parent_id), 0)) + 1
            self._append_csv_rows(
                self.lineage_edges_path,
                fieldnames=["tick", "parent_id", "child_id"],
                rows=[{"tick": int(tick), "parent_id": int(parent_id), "child_id": int(agent_id)}],
            )

        # Optionally emit a birth event.
        if self.log_births:
            self._emit_event({
                "schema_version": int(getattr(self, "schema_version", 2)),
                "tick": int(tick),
                "type": "birth",
                "agent_id": int(agent_id),
                "slot_id": int(slot_id),
                "team": int(team),
                "unit_type": int(unit_type),
                "parent_id": (int(parent_id) if parent_id is not None else None),
                "notes": str(notes),
            })

    def ingest_spawn_meta(self, meta: List[Dict[str, Any]]) -> None:
        """
        Ingest spawn metadata from RespawnController.

        Expected meta schema
        --------------------
        The RespawnController likely emits dicts with keys like:
        - tick
        - slot
        - agent_id
        - team_id
        - unit_id
        - parent_agent_id (optional)

        For each spawn record:
        - record_birth(..., notes="respawn")
        - optionally write agent_static row
        """
        if not self.enabled or not meta:
            return

        for m in meta:
            tick = _to_int(m.get("tick"))
            slot = _to_int(m.get("slot"))
            aid = _to_int(m.get("agent_id"))
            team = _to_int(m.get("team_id"))
            unit = _to_int(m.get("unit_id"))
            parent = m.get("parent_agent_id", None)
            parent_id = (_to_int(parent) if parent is not None else None)

            self.record_birth(
                tick=tick,
                agent_id=aid,
                slot_id=slot,
                team=team,
                unit_type=unit,
                parent_id=parent_id,
                notes="respawn",
                allow_existing=False,
            )

            self._maybe_write_agent_static(
                tick=int(tick),
                slot_id=int(slot),
                agent_id=int(aid),
                team_id=int(team),
                unit_type=int(unit),
                parent_id=parent_id,
                spawn_reason="respawn",
            )

    def _maybe_write_agent_static(
        self,
        tick: int,
        slot_id: int,
        agent_id: int,
        team_id: int,
        unit_type: int,
        parent_id: Optional[int],
        spawn_reason: str,
    ) -> None:
        """
        Write a row to agent_static.csv for an agent (deduplicated).

        What "static" means here
        ------------------------
        Values that usually do not change over an agent's lifetime:
        - max_hp
        - base_atk
        - vision_range
        - brain type and parameter count

        Preconditions
        -------------
        - telemetry enabled
        - write_agent_static enabled
        - registry attached
        - agent_id not already written

        How brain info is retrieved
        ---------------------------
        registry.brains is expected to be a list aligned by slot.
        If present:
        - brain_type = class name
        - param_count = sum(p.numel()) across parameters

        Why param_count matters
        -----------------------
        In ML experiments, model size strongly influences compute cost and learning dynamics.
        Logging it enables later correlation analysis.
        """
        if (not self.enabled) or (not self.write_agent_static) or (self._registry is None):
            return

        if agent_id in self._static_written:
            return

        try:
            from engine.agent_registry import COL_HP_MAX, COL_VISION, COL_ATK
            data = self._registry.agent_data

            max_hp = float(data[slot_id, COL_HP_MAX].item())
            vision = float(data[slot_id, COL_VISION].item())
            atk = float(data[slot_id, COL_ATK].item())

            brain_type = None
            param_count = None
            brains = getattr(self._registry, "brains", None)

            if isinstance(brains, list) and 0 <= slot_id < len(brains) and brains[slot_id] is not None:
                b = brains[slot_id]
                brain_type = b.__class__.__name__
                try:
                    param_count = int(sum(int(p.numel()) for p in b.parameters()))
                except Exception:
                    param_count = None

            self._append_csv_rows(
                self.agent_static_path,
                fieldnames=[
                    "agent_id", "team_id", "unit_type", "brain_type", "param_count",
                    "max_hp", "base_atk", "vision_range",
                    "spawn_tick", "parent_id", "spawn_reason",
                ],
                rows=[{
                    "agent_id": int(agent_id),
                    "team_id": int(team_id),
                    "unit_type": int(unit_type),
                    "brain_type": brain_type,
                    "param_count": param_count,
                    "max_hp": max_hp,
                    "base_atk": atk,
                    "vision_range": vision,
                    "spawn_tick": int(tick),
                    "parent_id": (int(parent_id) if parent_id is not None else None),
                    "spawn_reason": str(spawn_reason),
                }],
            )

            self._static_written.add(int(agent_id))

        except Exception as e:
            self._anomaly(f"agent_static write failed: {e}")

    # =============================================================================
    # Damage, kills, deaths (core event types)
    # =============================================================================

    def record_damage_victim_sum(
        self,
        tick: int,
        victim_ids: List[int],
        victim_team: List[int],
        victim_unit: List[int],
        damage: List[float],
        hp_before: Optional[List[float]] = None,
        hp_after: Optional[List[float]] = None,
    ) -> None:
        """
        Record damage in 'victim_sum' mode.

        Definition
        ----------
        victim_sum means:
          For each victim, we record a single damage number = total damage received this tick.
          (Even if multiple attackers hit them.)

        Why this mode exists
        --------------------
        It is a compression strategy.
        Per-hit logging can explode event volume. victim_sum scales with number of victims,
        not number of hits.

        Effects
        -------
        - Updates victim's `damage_taken_total`.
        - Emits a 'damage' event if log_damage is enabled.
        - Optionally logs hp_before/hp_after if both provided.

        Notes on hp logging
        -------------------
        - hp_before/after are optional because not every engine stage has them available.
        - If only one is provided, this code logs neither (requires both).
        """
        if not self.enabled:
            return

        for i, vid in enumerate(victim_ids):
            self._require_birth(vid, "damage(victim_sum)")
            rec = self._life[vid]
            rec["damage_taken_total"] = float(rec.get("damage_taken_total", 0.0)) + float(damage[i])

            if self.log_damage:
                ev = {
                    "schema_version": int(getattr(self, "schema_version", 2)),
                    "tick": int(tick),
                    "type": "damage",
                    "mode": "victim_sum",
                    "victim_id": int(vid),
                    "victim_team": int(victim_team[i]) if i < len(victim_team) else None,
                    "victim_unit": int(victim_unit[i]) if i < len(victim_unit) else None,
                    "damage": float(damage[i]),
                }
                if hp_before is not None and hp_after is not None:
                    ev["hp_before"] = float(hp_before[i])
                    ev["hp_after"] = float(hp_after[i])

                self._emit_event(ev)

    def record_damage_attacker_sum(
        self,
        tick: int,
        attacker_ids: List[int],
        damage_dealt: List[float],
    ) -> None:
        """
        Record damage in 'attacker_sum' mode.

        Definition
        ----------
        attacker_sum means:
          For each attacker, we record a single number = total damage dealt this tick.

        Why it exists
        -------------
        This updates per-agent totals for analysis (e.g., "top damage dealers") while
        keeping event logs small.

        Important: no event is emitted here.
        -----------------------------------
        This is deliberate:
        - If you already log victim_sum or per_hit events, emitting attacker_sum events
          doubles event volume without adding much information.

        Effects
        -------
        - Updates attacker `damage_dealt_total`.
        """
        if not self.enabled:
            return

        for i, aid in enumerate(attacker_ids):
            self._require_birth(aid, "damage(attacker_sum)")
            rec = self._life[aid]
            rec["damage_dealt_total"] = float(rec.get("damage_dealt_total", 0.0)) + float(damage_dealt[i])

        # No event emission here; event volume is controlled elsewhere.

    def record_damage_per_hit(
        self,
        tick: int,
        attacker_ids: List[int],
        victim_ids: List[int],
        damage: List[float],
    ) -> None:
        """
        Record damage in 'per_hit' mode.

        Definition
        ----------
        per_hit mode logs *every* hit as its own event:
          (attacker_id, victim_id, damage)

        Pros
        ----
        - Maximum fidelity: can reconstruct micro-dynamics.
        - Enables causal graphs (who hit whom, how often).

        Cons
        ----
        - High volume: event count scales with number of hits.

        Important:
        ----------
        This method does NOT update totals. Totals are expected to be updated by
        victim_sum/attacker_sum methods in the engine, to avoid double-counting.
        """
        if not self.enabled or not self.log_damage:
            return

        for i in range(len(damage)):
            aid = int(attacker_ids[i])
            vid = int(victim_ids[i])
            self._require_birth(aid, "damage(per_hit attacker)")
            self._require_birth(vid, "damage(per_hit victim)")

            self._emit_event({
                "schema_version": int(getattr(self, "schema_version", 2)),
                "tick": int(tick),
                "type": "damage",
                "mode": "per_hit",
                "attacker_id": aid,
                "victim_id": vid,
                "damage": float(damage[i]),
            })

    def record_kills(self, tick: int, killer_ids: List[int], victim_ids: List[int]) -> None:
        """
        Record kills.

        Behavior
        --------
        - For each (killer_id, victim_id) pair:
            - ensure both have birth records
            - increment killer's kills_total
            - emit 'kill' event if enabled

        Note on correctness
        -------------------
        The engine must ensure killer_ids and victim_ids are aligned lists.
        This method uses `min(len(killer_ids), len(victim_ids))` to be defensive.
        """
        if not self.enabled:
            return

        n = min(len(killer_ids), len(victim_ids))
        for i in range(n):
            kid = int(killer_ids[i])
            vid = int(victim_ids[i])
            self._require_birth(kid, "kill(killer)")
            self._require_birth(vid, "kill(victim)")

            rec = self._life[kid]
            rec["kills_total"] = int(rec.get("kills_total", 0)) + 1

            if self.log_kills:
                self._emit_event({
                    "schema_version": int(getattr(self, "schema_version", 2)),
                    "tick": int(tick),
                    "type": "kill",
                    "killer_id": kid,
                    "victim_id": vid,
                })

    def record_deaths(
        self,
        tick: int,
        dead_ids: List[int],
        dead_team: List[int],
        dead_unit: List[int],
        dead_slots: List[int],
        notes: str = "",
    ) -> None:
        """
        Record deaths.

        Behavior
        --------
        - For each dead agent:
            - ensure birth exists
            - set death_tick (only if not already set)
            - emit 'death' event if enabled

        Defensive programming
        ---------------------
        If death_tick is already set, we treat it as an anomaly and skip.
        Duplicate death indicates:
        - engine double-counting
        - reuse of agent_id incorrectly
        - or telemetry lifecycle bug
        """
        if not self.enabled:
            return

        for i, did in enumerate(dead_ids):
            self._require_birth(did, "death")
            rec = self._life[did]

            if rec.get("death_tick", None) is not None:
                self._anomaly(f"death: duplicate death for agent_id={did}")
                continue

            rec["death_tick"] = int(tick)

            if self.log_deaths:
                self._emit_event({
                    "schema_version": int(getattr(self, "schema_version", 2)),
                    "tick": int(tick),
                    "type": "death",
                    "agent_id": int(did),
                    "slot_id": int(dead_slots[i]) if i < len(dead_slots) else None,
                    "team": int(dead_team[i]) if i < len(dead_team) else None,
                    "unit_type": int(dead_unit[i]) if i < len(dead_unit) else None,
                    "notes": str(notes),
                })

    # =============================================================================
    # Tick lifecycle hooks
    # =============================================================================

    def on_tick_end(self, tick: int) -> None:
        """
        Called once at the end of each simulation tick.

        Responsibilities
        ----------------
        1) Store last tick seen (for close()).
        2) Periodically validate telemetry state (detect anomalies early).
        3) Periodically flush:
           - agent_life snapshot
           - event chunk buffer
        4) Optionally write tick summary row.

        Performance considerations
        --------------------------
        Validation + flushing are expensive relative to pure simulation logic.
        That's why they run on schedules: every `validate_every` ticks, every `flush_every` ticks.
        """
        if not self.enabled:
            return

        self._last_tick_seen = int(tick)

        if self.validate_every > 0 and (int(tick) % int(self.validate_every) == 0):
            self.validate()

        if self.flush_every > 0 and (int(tick) % int(self.flush_every) == 0):
            self._flush_agent_life_snapshot()
            self._flush_event_chunk()

        if self.tick_summary_every > 0 and (tick % int(self.tick_summary_every)) == 0:
            self._write_tick_summary(tick=int(tick))

    def _write_tick_summary(self, tick: int) -> None:
        """
        Compute and append a summary row to tick_summary.csv.

        Preconditions
        -------------
        - telemetry enabled
        - registry and stats attached (attach_context must be called)

        What is computed
        ----------------
        - Alive counts for red/blue
        - Mean HP for red/blue
        - Some accumulated stats from `self._stats` (kills, deaths, dmg dealt/taken)

        Why this is useful
        ------------------
        This gives you a low-frequency time series that is cheap to plot:
        - population dynamics
        - survivability trends
        - macro balance between teams
        """
        if (not self.enabled) or (self._registry is None) or (self._stats is None):
            return

        try:
            import torch  # local import; torch is core dep already used by project
            from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_HP

            data = self._registry.agent_data

            alive_mask = (data[:, COL_ALIVE] > 0.5)
            red_mask = alive_mask & (data[:, COL_TEAM] == 2.0)
            blue_mask = alive_mask & (data[:, COL_TEAM] == 3.0)

            red_n = int(red_mask.sum().item())
            blue_n = int(blue_mask.sum().item())

            mean_red = 0.0
            if red_n > 0:
                mean_red = float(data[red_mask, COL_HP].mean().item())

            mean_blue = 0.0
            if blue_n > 0:
                mean_blue = float(data[blue_mask, COL_HP].mean().item())

            s = self._stats
            row = {
                "tick": int(tick),
                "elapsed_s": float(getattr(s, "elapsed_seconds", 0.0)),
                "red_alive": red_n,
                "blue_alive": blue_n,
                "mean_hp_red": mean_red,
                "mean_hp_blue": mean_blue,
                "red_kills": float(getattr(getattr(s, "red", None), "kills", 0.0)),
                "blue_kills": float(getattr(getattr(s, "blue", None), "kills", 0.0)),
                "red_deaths": float(getattr(getattr(s, "red", None), "deaths", 0.0)),
                "blue_deaths": float(getattr(getattr(s, "blue", None), "deaths", 0.0)),
                "red_dmg_dealt": float(getattr(getattr(s, "red", None), "dmg_dealt", 0.0)),
                "blue_dmg_dealt": float(getattr(getattr(s, "blue", None), "dmg_dealt", 0.0)),
                "red_dmg_taken": float(getattr(getattr(s, "red", None), "dmg_taken", 0.0)),
                "blue_dmg_taken": float(getattr(getattr(s, "blue", None), "dmg_taken", 0.0)),
            }

            # Write a single-row append with fieldnames derived from row keys.
            # This assumes the CSV header matches these keys (created in __init__).
            self._append_csv_rows(self.tick_summary_path, fieldnames=list(row.keys()), rows=[row])

        except Exception as e:
            self._anomaly(f"tick_summary write failed: {e}")

    # =============================================================================
    # Shutdown
    # =============================================================================

    def close(self) -> None:
        """
        Clean shutdown: flush buffers and write final snapshots.

        What happens on close
        ---------------------
        1) Flush any remaining events buffer to a chunk file.
        2) Write the final agent_life snapshot.
        3) Optionally write one last tick summary row (if enabled).

        Reliability note
        ----------------
        This method is typically called in a `finally:` block or at program end.
        It should not crash the program. Hence the try/except.

        Current behavior
        ----------------
        If an exception occurs here, it prints a message (not silent).
        That may be intentional because shutdown failures are important.
        """
        if not self.enabled:
            return
        try:
            self._flush_event_chunk()
            self._flush_agent_life_snapshot()
            if self._last_tick_seen is not None and self.tick_summary_every > 0:
                self._write_tick_summary(tick=self._last_tick_seen)
        except Exception as e:
            print(f"[telemetry] Close failed: {e}")