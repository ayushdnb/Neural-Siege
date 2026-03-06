"""
Micro-benchmark: measures simulation throughput (ticks/sec) before and after patches.

Usage:
    python benchmark_ticks.py [--ticks 500] [--warmup 50]

The script runs the simulation in headless mode and reports:
    - mean ticks/sec
    - 95th-percentile tick duration
    - total wall time

Run BEFORE applying patches to get a baseline, then run AFTER.
"""

import time
import argparse
import statistics
import torch

# ── project imports ─────────────────────────────────────────────────────────
import config
from engine.agent_registry import AgentsRegistry
from engine.tick import TickEngine
from engine.mapgen import generate_map, Zones
from engine.spawn import spawn_agents
from simulation.stats import SimulationStats


def run_benchmark(n_ticks: int = 500, warmup: int = 50) -> None:
    device = torch.device(config.TORCH_DEVICE)

    # Build grid + registry
    grid, zones = generate_map(config.GRID_H, config.GRID_W, device=device)
    registry = AgentsRegistry(grid)
    stats = SimulationStats()

    # Spawn agents (uses config.MAX_AGENTS, team ratios, etc.)
    spawn_agents(registry, grid, stats)

    engine = TickEngine(registry=registry, grid=grid, stats=stats, zones=zones)

    # Warmup pass (fills caches, JIT, etc.)
    print(f"Warming up ({warmup} ticks)…")
    for _ in range(warmup):
        engine.run_tick()

    # Synchronize GPU before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Benchmarking ({n_ticks} ticks)…")
    tick_times = []
    t_total_start = time.perf_counter()

    for _ in range(n_ticks):
        t0 = time.perf_counter()
        engine.run_tick()
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure GPU work is complete before timing
        t1 = time.perf_counter()
        tick_times.append(t1 - t0)

    t_total = time.perf_counter() - t_total_start

    mean_ms   = statistics.mean(tick_times) * 1000
    p50_ms    = statistics.median(tick_times) * 1000
    p95_ms    = sorted(tick_times)[int(0.95 * len(tick_times))] * 1000
    ticks_sec = n_ticks / t_total

    print(f"\n{'─'*50}")
    print(f"  Ticks measured : {n_ticks}")
    print(f"  Total wall time: {t_total:.3f}s")
    print(f"  Ticks / sec    : {ticks_sec:,.1f}")
    print(f"  Mean tick (ms) : {mean_ms:.2f}")
    print(f"  P50 tick  (ms) : {p50_ms:.2f}")
    print(f"  P95 tick  (ms) : {p95_ms:.2f}")
    print(f"{'─'*50}\n")

    # Optional: per-component profiling using PyTorch profiler
    if torch.cuda.is_available():
        print("GPU memory summary:")
        print(torch.cuda.memory_summary(abbreviated=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks",  type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(n_ticks=args.ticks, warmup=args.warmup)