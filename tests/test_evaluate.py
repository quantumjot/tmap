"""Known-answer tests for the evaluation metrics.

Each metric is checked against a construction whose correct value is known by
hand, so the metrics can be trusted when they are used to judge method changes.
"""
import numpy as np
import pytest

from scipy import sparse

from tmap import evaluate
from tmap.alignment import DTWAlignment


# --- helpers ---------------------------------------------------------------


def test_node_labels_expands_per_trajectory():
    out = evaluate.node_labels([3, 2], np.array([7, 9]))
    assert out.tolist() == [7, 7, 7, 9, 9]


def test_node_labels_rejects_length_mismatch():
    with pytest.raises(ValueError, match="2 sequences but 3 labels"):
        evaluate.node_labels([3, 2], np.array([0, 1, 2]))


# --- supervised ------------------------------------------------------------


def test_knn_purity_is_one_for_separated_groups():
    # three tight clusters, far apart; every close neighbour shares the label
    y = np.concatenate([
        np.zeros((10, 2)) + [0, 0],
        np.zeros((10, 2)) + [100, 0],
        np.zeros((10, 2)) + [0, 100],
    ]) + np.random.default_rng(0).standard_normal((30, 2)) * 0.01
    labels = np.repeat([0, 1, 2], 10)
    assert evaluate.knn_purity(y, labels, k=5) == pytest.approx(1.0)


def test_knn_purity_near_chance_for_random_labels():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((300, 2))
    labels = rng.integers(0, 3, size=300)  # unrelated to position
    # three roughly equal groups => chance level is ~1/3
    assert evaluate.knn_purity(y, labels, k=10) == pytest.approx(1 / 3, abs=0.08)


def test_silhouette_higher_for_separated_than_overlapping():
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 50)
    separated = np.concatenate([rng.standard_normal((50, 2)), rng.standard_normal((50, 2)) + 50])
    overlapping = rng.standard_normal((100, 2))
    assert evaluate.silhouette(separated, labels) > 0.9
    assert evaluate.silhouette(overlapping, labels) < 0.2


def test_silhouette_nan_for_single_group():
    assert np.isnan(evaluate.silhouette(np.random.default_rng(3).standard_normal((10, 2)), np.zeros(10)))


# --- temporal --------------------------------------------------------------


def test_temporal_coherence_is_one_when_time_is_the_layout():
    # one trajectory laid out along a line: the next timepoint is always nearest
    y = np.stack([np.arange(20.0), np.zeros(20)], axis=-1)
    assert evaluate.temporal_coherence([20], y, k=2) == pytest.approx(1.0)


def test_temporal_coherence_drops_when_time_order_is_destroyed():
    rng = np.random.default_rng(4)
    y = rng.standard_normal((60, 2)) * 10
    ordered = np.stack([np.arange(60.0), np.zeros(60)], axis=-1)
    assert evaluate.temporal_coherence([60], y, k=2) < evaluate.temporal_coherence([60], ordered, k=2)


def test_velocity_coherence_is_one_for_co_flowing_trajectories():
    # five parallel trajectories all advancing in +x
    trajs = [np.stack([np.arange(10.0), np.full(10, float(j))], axis=-1) for j in range(5)]
    y = np.concatenate(trajs)
    assert evaluate.velocity_coherence([10] * 5, y, k=3) == pytest.approx(1.0)


def test_velocity_coherence_penalises_counter_flow():
    # two interleaved trajectories occupying the same space, flowing oppositely
    forward = np.stack([np.arange(10.0), np.zeros(10)], axis=-1)
    backward = np.stack([np.arange(9.0, -1.0, -1.0), np.full(10, 0.01)], axis=-1)
    mixed = np.concatenate([forward, backward])
    co = np.concatenate([forward, forward + [0, 0.01]])
    assert evaluate.velocity_coherence([10, 10], mixed, k=3) < evaluate.velocity_coherence([10, 10], co, k=3)


def test_velocity_coherence_excludes_static_nodes():
    # a completely stationary trajectory has no velocity anywhere -> nan
    y = np.zeros((10, 2))
    assert np.isnan(evaluate.velocity_coherence([10], y, k=2))


# --- graph-referenced and high-D reference --------------------------------


def test_graph_neighbor_preservation_is_one_when_embedding_matches_graph():
    # chain graph; embed on a line so graph neighbours are embedded neighbours
    n = 20
    rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    P = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    y = np.stack([np.arange(float(n)), np.zeros(n)], axis=-1)
    assert evaluate.graph_neighbor_preservation(P, y, k=2) == pytest.approx(1.0)


def test_knn_preservation_is_one_for_identity_embedding():
    x = np.random.default_rng(5).standard_normal((50, 2))
    assert evaluate.knn_preservation(x, x.copy(), k=5) == pytest.approx(1.0)


# --- alignment-level -------------------------------------------------------


def test_correspondence_map_is_identity_for_identical_sequences():
    seq = np.cumsum(np.random.default_rng(6).standard_normal((15, 3)), axis=0)
    m = evaluate.correspondence_map(DTWAlignment(), seq, seq.copy())
    assert m.tolist() == list(range(15))


def test_transitivity_violation_is_zero_for_identical_trajectories():
    seq = np.cumsum(np.random.default_rng(7).standard_normal((12, 3)), axis=0)
    seqs = [seq.copy() for _ in range(4)]
    assert evaluate.transitivity_violation(seqs, DTWAlignment(), n_triples=4, seed=0) == pytest.approx(0.0)


def test_transitivity_violation_in_unit_range_for_real_trajectories():
    rng = np.random.default_rng(8)
    seqs = [np.cumsum(rng.standard_normal((15, 3)) * 0.1, axis=0) for _ in range(5)]
    v = evaluate.transitivity_violation(seqs, DTWAlignment(), n_triples=6, seed=0)
    assert 0.0 <= v <= 1.0


def test_transitivity_violation_needs_three_sequences():
    seqs = [np.zeros((5, 2)), np.zeros((5, 2))]
    with pytest.raises(ValueError, match="at least 3"):
        evaluate.transitivity_violation(seqs, DTWAlignment())


def test_edge_recall_counts_retained_structure():
    n = 6
    rows = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 5])
    cols = np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 0])
    full = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    keep = 7
    screened = sparse.csr_matrix(
        (np.ones(keep), (rows[:keep], cols[:keep])), shape=(n, n)
    )
    assert evaluate.edge_recall(full, screened) == pytest.approx(keep / rows.size)
    assert evaluate.edge_recall(full, full) == pytest.approx(1.0)


def test_edge_recall_ignores_edge_values():
    rows, cols = np.array([0, 1, 2]), np.array([1, 2, 0])
    a = sparse.csr_matrix((np.ones(3), (rows, cols)), shape=(3, 3))
    b = sparse.csr_matrix((np.full(3, 99.0), (rows, cols)), shape=(3, 3))
    assert evaluate.edge_recall(a, b) == pytest.approx(1.0)


def test_edge_recall_rejects_shape_mismatch():
    a = sparse.csr_matrix(np.ones((3, 3)))
    b = sparse.csr_matrix(np.ones((4, 4)))
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate.edge_recall(a, b)


# --- battery ---------------------------------------------------------------


def test_summarize_skips_absent_inputs_and_includes_present_ones():
    y = np.stack([np.arange(20.0), np.zeros(20)], axis=-1)
    bare = evaluate.summarize(y, seq_lengths=[10, 10], k=3)
    assert set(bare) == {"temporal_coherence", "velocity_coherence"}

    full = evaluate.summarize(
        y, seq_lengths=[10, 10], labels=np.array([0, 1]), x=y.copy(), k=3
    )
    assert {"knn_purity", "silhouette", "knn_preservation", "trustworthiness"} <= set(full)
