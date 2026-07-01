#!/usr/bin/env python
"""Stage-level performance benchmark for tmap.

Times the three hot stages of the pipeline independently so we can see where
wall-clock goes and track improvements as the code is optimised:

    1. alignment      -> calculate_distance_matrix (pairwise DTW/OT)
    2. probability    -> calculate_high_dimensional_probability_matrix
    3. embedding      -> the JAX gradient-descent optimisation loop

Results are printed as a table and optionally written to CSV. This is the
Phase 0 baseline harness for the performance work: nothing should land on a
hot path without a before/after number from this script.

Usage
-----
    python scripts/benchmark.py                         # default sweep
    python scripts/benchmark.py --sequences 10 20 40    # custom K sweep
    python scripts/benchmark.py --length 100 --features 3 --iterations 200
    python scripts/benchmark.py --output baseline.csv
    python scripts/benchmark.py --real-data
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import Optional

import numpy as np

from tmap import temporal
from tmap.alignment import DTWAlignment
from tmap.layout import SpectralLayout


def generate_sequences(
    n_sequences: int, length: int, n_features: int, *, seed: int = 42
) -> list[np.ndarray]:
    """Deterministic synthetic sequences (smooth random walks)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_sequences):
        steps = rng.standard_normal((length, n_features)) * 0.1
        seqs.append(np.cumsum(steps, axis=0).astype(np.float64))
    return seqs


def load_real_data(n_sequences: Optional[int] = None) -> Optional[list[np.ndarray]]:
    from pathlib import Path

    data_path = Path("examples/data/trimmed_tracks.npy")
    if not data_path.exists():
        return None
    raw = np.load(data_path)
    tracks = [raw[raw[:, 0] == idx, 1:] for idx in np.unique(raw[:, 0])]
    return tracks[:n_sequences] if n_sequences else tracks


def _time_embedding(prob: np.ndarray, n_components: int, iterations: int) -> float:
    """Time only the optimisation loop, excluding JIT compilation.

    Uses the production ``optimize_embedding`` path so the benchmark tracks
    whatever the library actually does.
    """
    a, b = temporal.find_hyperparameters(0.01)
    rng = np.random.default_rng(0)
    y = rng.standard_normal((prob.shape[0], n_components))

    # warm-up / compile (not timed)
    temporal.optimize_embedding(
        prob, y, a, b, learning_rate=0.1, n_iterations=1, progress=False
    )

    start = time.perf_counter()
    temporal.optimize_embedding(
        prob, y, a, b, learning_rate=0.1, n_iterations=iterations, progress=False
    )
    return time.perf_counter() - start


def benchmark_one(
    sequences: list[np.ndarray],
    *,
    n_neighbors: int,
    n_components: int,
    iterations: int,
    window: Optional[int],
    n_jobs: Optional[int] = None,
) -> dict:
    n_nodes = sum(s.shape[0] for s in sequences)
    aligner = DTWAlignment(window=window)

    t0 = time.perf_counter()
    dist = temporal.calculate_distance_matrix(sequences, aligner, mask=True, n_jobs=n_jobs)
    t_align = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob = temporal.calculate_high_dimensional_probability_matrix(dist, n_neighbors)
    t_prob = time.perf_counter() - t0

    t_embed = _time_embedding(prob, n_components, iterations)

    dense_bytes = n_nodes * n_nodes * 8
    return {
        "n_sequences": len(sequences),
        "n_nodes": n_nodes,
        "n_features": sequences[0].shape[1],
        "t_alignment_s": round(t_align, 4),
        "t_probability_s": round(t_prob, 4),
        "t_embedding_s": round(t_embed, 4),
        "t_total_s": round(t_align + t_prob + t_embed, 4),
        "dense_matrix_mb": round(dense_bytes / 1e6, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", type=int, nargs="+", default=[10, 20, 40])
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--features", type=int, default=3)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--components", type=int, default=2)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--jobs", type=int, default=None, help="threads for pairwise DTW; -1 for all cores")
    p.add_argument("--real-data", action="store_true")
    p.add_argument("--output", type=str, default=None, help="CSV path to append results to")
    args = p.parse_args()

    rows = []
    for k in args.sequences:
        if args.real_data:
            seqs = load_real_data(k)
            if seqs is None:
                print("Real data not found; falling back to synthetic.")
                seqs = generate_sequences(k, args.length, args.features)
        else:
            seqs = generate_sequences(k, args.length, args.features)

        print(f"\n=== K={len(seqs)} sequences, N={sum(s.shape[0] for s in seqs)} nodes ===")
        row = benchmark_one(
            seqs,
            n_neighbors=args.neighbors,
            n_components=args.components,
            iterations=args.iterations,
            window=args.window,
            n_jobs=args.jobs,
        )
        rows.append(row)
        print(
            f"  alignment  {row['t_alignment_s']:8.3f}s | "
            f"probability {row['t_probability_s']:8.3f}s | "
            f"embedding {row['t_embedding_s']:8.3f}s | "
            f"total {row['t_total_s']:8.3f}s | "
            f"dense P {row['dense_matrix_mb']:.0f} MB"
        )

    if args.output:
        fieldnames = list(rows[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
