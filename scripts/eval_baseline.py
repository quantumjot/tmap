#!/usr/bin/env python
"""Baseline evaluation battery for tmap.

Runs the full pipeline on labelled synthetic trajectories and reports every
metric in :mod:`tmap.evaluate`, so that a change to alignment, graph
construction or the optimiser can be scored against a common reference.

The trajectories carry ground-truth group labels, so the supervised metrics
(``knn_purity``) mean something — unlike trustworthiness, which only compares
the embedding to the high-dimensional input.

The default substrate is ``branching``: trajectories share a trunk then diverge
onto distinct routes, at their own uneven traversal speeds. Scores on it track
the amount of real structure, so a change can be read against a scale. The
``oscillator`` substrates are retained for continuity but every trajectory
there traverses the same 1-D line, so their numbers have no reference point.

Usage
-----
    python scripts/eval_baseline.py                      # branching (default)
    python scripts/eval_baseline.py --branch-time 0.25   # easier task
    python scripts/eval_baseline.py --substrate oscillator
    python scripts/eval_baseline.py --substrate unbalanced
    python scripts/eval_baseline.py --aligner ot
    python scripts/eval_baseline.py --noise 0.5          # robustness probe
    python scripts/eval_baseline.py --output baseline.json

Compare two runs by diffing their ``--output`` files. Record the baseline once
on ``main``, then re-run on a branch and diff.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

from tmap import evaluate, simulate, temporal
from tmap.alignment import DTWAlignment, OTAlignment


def build_aligner(name: str):
    if name == "dtw":
        return DTWAlignment()
    if name == "ot":
        return OTAlignment()
    raise SystemExit(f"unknown aligner: {name!r} (expected 'dtw' or 'ot')")


def run(
    *,
    n: int,
    length: int,
    features: int,
    noise: float | None,
    substrate: str,
    branch_time: float,
    aligner_name: str,
    n_neighbors: int,
    n_components: int,
    iterations: int,
    k: int,
    seed: int,
    n_triples: int,
) -> dict:
    # each substrate has its own sensible noise level; --noise overrides it.
    # the oscillator keeps 0.0 for continuity with its historical behaviour.
    if noise is None:
        noise = 0.05 if substrate == "branching" else 0.0

    if substrate == "branching":
        result = simulate.simulate_branching_trajectories(
            n=n, length=length, n_components=features, branch_time=branch_time,
            noise=noise, seed=seed, return_labels=True,
        )
    elif substrate in ("oscillator", "unbalanced"):
        generator = (
            simulate.simulate_unbalanced_trajectories
            if substrate == "unbalanced"
            else simulate.simulate_trajectories
        )
        result = generator(
            t=np.linspace(0, 10, length), n=n, n_components=features,
            noise=noise, seed=seed, return_labels=True,
        )
    else:
        raise SystemExit(f"unknown substrate: {substrate!r}")
    sequences, labels = result[0], result[1]

    aligner = build_aligner(aligner_name)
    seq_lengths = [s.shape[0] for s in sequences]
    x = np.concatenate(sequences, axis=0)

    dist = temporal.calculate_distance_matrix(sequences, aligner, mask=True)
    prob = temporal.calculate_high_dimensional_probability_matrix(dist, n_neighbors)
    a, b = temporal.find_hyperparameters(0.01)

    y0 = np.random.default_rng(seed).standard_normal((x.shape[0], n_components))
    y = temporal.optimize_embedding(
        prob, y0, a, b, n_iterations=iterations, optimizer="sampled",
        random_state=seed, progress=False,
    )

    metrics = evaluate.summarize(
        y, seq_lengths=seq_lengths, labels=labels, P=prob, x=x, k=k
    )
    # alignment-level: independent of the embedding
    metrics["transitivity_violation"] = evaluate.transitivity_violation(
        sequences, aligner, n_triples=n_triples, seed=seed
    )

    return {
        "config": {
            "n_per_regime": n,
            "n_trajectories": len(sequences),
            "n_nodes": int(x.shape[0]),
            "length": length,
            "features": features,
            "noise": noise,
            "substrate": substrate,
            "branch_time": branch_time if substrate == "branching" else None,
            "aligner": aligner_name,
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "iterations": iterations,
            "k": k,
            "seed": seed,
        },
        "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


# Higher is better for everything except transitivity_violation.
LOWER_IS_BETTER = {"transitivity_violation"}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sequences", type=int, default=8, help="trajectories per regime (x3 total)")
    p.add_argument("--length", type=int, default=60)
    p.add_argument("--features", type=int, default=3)
    p.add_argument(
        "--noise", type=float, default=None,
        help="per-component observation noise; default is per-substrate "
             "(0.05 branching, 0.0 oscillator)",
    )
    p.add_argument(
        "--substrate", type=str, default="branching",
        choices=("branching", "oscillator", "unbalanced"),
        help="trajectory generator; 'branching' (default) is the only one whose "
             "scores track the amount of real structure",
    )
    p.add_argument(
        "--branch-time", type=float, default=0.5,
        help="branching only: fraction traversed before branches separate; "
             "1.0 identical paths, 0.0 independent routes",
    )
    p.add_argument("--aligner", type=str, default="dtw", choices=("dtw", "ot"))
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--components", type=int, default=2)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--k", type=int, default=15, help="neighbourhood size for the metrics")
    p.add_argument("--triples", type=int, default=20, help="triples sampled for transitivity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="write the report as JSON")
    args = p.parse_args()

    report = run(
        n=args.sequences,
        length=args.length,
        features=args.features,
        noise=args.noise,
        substrate=args.substrate,
        branch_time=args.branch_time,
        aligner_name=args.aligner,
        n_neighbors=args.neighbors,
        n_components=args.components,
        iterations=args.iterations,
        k=args.k,
        seed=args.seed,
        n_triples=args.triples,
    )

    cfg = report["config"]
    described = cfg["substrate"]
    if cfg["branch_time"] is not None:
        described += f"(branch_time={cfg['branch_time']})"
    print(
        f"{cfg['n_trajectories']} trajectories / {cfg['n_nodes']} nodes, "
        f"{described}, aligner={cfg['aligner']}, noise={cfg['noise']}, k={cfg['k']}"
    )
    print("-" * 58)
    for key, value in report["metrics"].items():
        arrow = "lower better" if key in LOWER_IS_BETTER else ""
        print(f"{key:32s} {value:>8}  {arrow}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
