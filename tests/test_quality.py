"""Embedding quality guardrail: the sampled optimiser must not degrade
embedding quality relative to the dense optimiser.

Trustworthiness measures how well the low-dimensional neighbourhoods reflect
the high-dimensional data; the sampled optimiser changes the objective (true
per-pair UMAP kernel, stochastic repulsion) so embeddings differ, but quality
must stay within tolerance of the dense reference.
"""
import numpy as np

from sklearn.manifold import trustworthiness

from tmap import temporal
from tmap.alignment import DTWAlignment

TOLERANCE = 0.05


def test_sampled_quality_within_tolerance_of_dense():
    rng = np.random.default_rng(7)
    seqs = [
        np.cumsum(rng.standard_normal((30, 3)) * 0.1, axis=0) for _ in range(8)
    ]
    x = np.concatenate(seqs, axis=0)

    dist = temporal.calculate_distance_matrix(seqs, DTWAlignment(), mask=True)
    prob = temporal.calculate_high_dimensional_probability_matrix(dist, 5)
    a, b = temporal.find_hyperparameters(0.01)

    y0 = np.random.default_rng(0).standard_normal((x.shape[0], 2))

    # each optimiser at its own default learning rate (the production defaults)
    y_dense = temporal.optimize_embedding(
        prob, y0, a, b, n_iterations=200, optimizer="dense", progress=False
    )
    y_sampled = temporal.optimize_embedding(
        prob, y0, a, b, n_iterations=200, optimizer="sampled", random_state=0, progress=False
    )

    t_dense = trustworthiness(x, y_dense, n_neighbors=10)
    t_sampled = trustworthiness(x, y_sampled, n_neighbors=10)

    assert t_sampled >= t_dense - TOLERANCE, (
        f"sampled trustworthiness {t_sampled:.4f} more than {TOLERANCE} below "
        f"dense reference {t_dense:.4f}"
    )
