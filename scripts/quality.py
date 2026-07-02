#!/usr/bin/env python
"""Embedding quality comparison: sampled vs dense optimiser.

Computes trustworthiness (sklearn) and k-NN neighbourhood preservation of the
low-dimensional embedding with respect to the concatenated high-dimensional
input, for both optimisers on the same graph and initial layout. Used to
verify that the O(nnz) sampled optimiser does not degrade embedding quality
relative to the original O(N^2) dense path.

Usage
-----
    python scripts/quality.py                     # synthetic benchmark data
    python scripts/quality.py --sequences 20 --length 50
    python scripts/quality.py --real-data
"""
from __future__ import annotations

import argparse

import numpy as np

from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from tmap import temporal
from tmap.alignment import DTWAlignment

from benchmark import generate_sequences, load_real_data


def knn_preservation(x: np.ndarray, y: np.ndarray, k: int = 15) -> float:
    """Mean fraction of each point's high-D k-NN preserved in the embedding."""
    nn_x = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(return_distance=False)
    nn_y = NearestNeighbors(n_neighbors=k + 1).fit(y).kneighbors(return_distance=False)
    overlap = [
        len(set(a[1:]) & set(b[1:])) / k for a, b in zip(nn_x, nn_y)
    ]
    return float(np.mean(overlap))


def compare(
    sequences: list[np.ndarray],
    *,
    n_neighbors: int = 15,
    n_components: int = 2,
    iterations: int = 200,
    k_quality: int = 15,
) -> dict:
    x = np.concatenate(sequences, axis=0)
    dist = temporal.calculate_distance_matrix(sequences, DTWAlignment(), mask=True)
    prob = temporal.calculate_high_dimensional_probability_matrix(dist, n_neighbors)
    a, b = temporal.find_hyperparameters(0.01)

    rng = np.random.default_rng(0)
    y0 = rng.standard_normal((x.shape[0], n_components))

    # each optimiser runs at its own default learning rate (0.1 constant for
    # dense, 1.0 with linear decay for sampled) — the production defaults
    y_dense = temporal.optimize_embedding(
        prob, y0, a, b, n_iterations=iterations, optimizer="dense", progress=False
    )
    y_sampled = temporal.optimize_embedding(
        prob,
        y0,
        a,
        b,
        n_iterations=iterations,
        optimizer="sampled",
        random_state=0,
        progress=False,
    )

    return {
        "n_nodes": x.shape[0],
        "trustworthiness_dense": round(trustworthiness(x, y_dense, n_neighbors=k_quality), 4),
        "trustworthiness_sampled": round(trustworthiness(x, y_sampled, n_neighbors=k_quality), 4),
        "knn_preservation_dense": round(knn_preservation(x, y_dense, k=k_quality), 4),
        "knn_preservation_sampled": round(knn_preservation(x, y_sampled, k=k_quality), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", type=int, default=10)
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--features", type=int, default=3)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--real-data", action="store_true")
    args = p.parse_args()

    if args.real_data:
        seqs = load_real_data(args.sequences)
        if seqs is None:
            raise SystemExit("Real data not found at examples/data/trimmed_tracks.npy")
    else:
        seqs = generate_sequences(args.sequences, args.length, args.features)

    result = compare(seqs, n_neighbors=args.neighbors, iterations=args.iterations)
    for key, value in result.items():
        print(f"{key:28s} {value}")


if __name__ == "__main__":
    main()
