"""Guardrails for the sampled (sparse edge + negative sampling) optimiser.

The sampled optimiser intentionally produces *different* embeddings from the
dense full-batch path (it optimises the standard UMAP per-pair objective over
the graph edges), so it is guarded by determinism, loss-decrease and shape
invariants rather than the golden fixtures.
"""
import numpy as np
import pytest

from tmap import TemporalMAP, temporal
from tmap.alignment import DTWAlignment
from tmap.embedding import (
    edges_from_matrix,
    optimize_embedding_sampled,
    sampled_embedding_loss,
)

K, LENGTH, FEATURES, N_NEIGHBORS, SEED = 6, 20, 3, 5, 1234


def _make_sequences() -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [
        np.cumsum(rng.standard_normal((LENGTH, FEATURES)) * 0.1, axis=0)
        for _ in range(K)
    ]


@pytest.fixture(scope="module")
def graph():
    seqs = _make_sequences()
    dist = temporal.calculate_distance_matrix(seqs, DTWAlignment(), mask=True)
    prob = temporal.calculate_high_dimensional_probability_matrix(dist, N_NEIGHBORS)
    a, b = temporal.find_hyperparameters(0.01)
    return prob, a, b


def test_edges_from_matrix_excludes_diagonal_and_zeros(graph):
    prob, _, _ = graph
    heads, tails, weights = edges_from_matrix(prob)
    assert np.all(heads != tails)
    assert np.all(weights > 0)
    # both directions of every undirected edge are present (P is symmetric)
    fwd = set(zip(heads.tolist(), tails.tolist()))
    assert all((j, i) in fwd for i, j in fwd)


def _run_sampled(graph, seed, n_iterations=100):
    prob, a, b = graph
    edges = edges_from_matrix(prob)
    rng = np.random.default_rng(0)
    y0 = rng.standard_normal((prob.shape[0], 2))
    y = optimize_embedding_sampled(
        edges,
        y0,
        a,
        b,
        learning_rate=1.0,
        n_iterations=n_iterations,
        random_state=seed,
        progress=False,
    )
    return y0, y, edges


def test_sampled_deterministic(graph):
    _, y1, _ = _run_sampled(graph, seed=42)
    _, y2, _ = _run_sampled(graph, seed=42)
    np.testing.assert_array_equal(y1, y2)


def test_sampled_seed_changes_result(graph):
    _, y1, _ = _run_sampled(graph, seed=1)
    _, y2, _ = _run_sampled(graph, seed=2)
    assert not np.allclose(y1, y2)


def test_sampled_loss_decreases(graph):
    prob, a, b = graph
    y0, y, edges = _run_sampled(graph, seed=0)
    loss_before = sampled_embedding_loss(edges, y0, a, b)
    loss_after = sampled_embedding_loss(edges, y, a, b)
    assert np.isfinite(loss_after)
    assert loss_after < loss_before


def test_sampled_output_shape_and_finite(graph):
    prob, _, _ = graph
    _, y, _ = _run_sampled(graph, seed=0)
    assert y.shape == (prob.shape[0], 2)
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("optimizer", ["sampled", "dense"])
def test_fit_with_both_optimizers(optimizer):
    seqs = _make_sequences()
    tm = TemporalMAP(
        n_neighbors=N_NEIGHBORS,
        layout="RANDOM",
        optimizer=optimizer,
        random_state=0,
    )
    y = tm.fit(seqs, max_iterations=20)
    assert y.shape == (sum(s.shape[0] for s in seqs), 2)
    assert np.all(np.isfinite(y))


def test_unknown_optimizer_raises(graph):
    prob, a, b = graph
    rng = np.random.default_rng(0)
    y0 = rng.standard_normal((prob.shape[0], 2))
    with pytest.raises(ValueError, match="Unknown optimizer"):
        temporal.optimize_embedding(
            prob, y0, a, b, learning_rate=0.1, n_iterations=1, optimizer="bogus"
        )
