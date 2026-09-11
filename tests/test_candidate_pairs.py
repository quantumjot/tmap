"""Candidate-pair screening: align only plausible neighbours, not all K^2 pairs.

Screening is approximate by design, so these tests pin the structural guarantees
(coverage, ordering, no fragmentation) and the fallbacks, while recall against
the all-pairs graph is measured separately with evaluate.edge_recall.
"""
import numpy as np
import pytest

from tmap import evaluate, temporal
from tmap.alignment import DTWAlignment
from tmap.simulate import simulate_branching_trajectories


@pytest.fixture(scope="module")
def sequences():
    # 4 per branch x 3 branches: branches give the descriptor real structure
    return simulate_branching_trajectories(n=4, length=30, noise=0.05, seed=0)


# --- descriptors -----------------------------------------------------------


def test_descriptors_have_fixed_width_for_ragged_input():
    seqs = [np.random.default_rng(0).standard_normal((n, 3)) for n in (10, 25, 40)]
    d = temporal.trajectory_descriptors(seqs, n_resample=8)
    assert d.shape == (3, 8 * 3)


def test_descriptors_preserve_the_endpoints():
    seq = np.stack([np.arange(10.0), np.arange(10.0) * 2], axis=-1)
    d = temporal.trajectory_descriptors([seq], n_resample=5).reshape(5, 2)
    np.testing.assert_allclose(d[0], seq[0])
    np.testing.assert_allclose(d[-1], seq[-1])


def test_descriptors_are_length_invariant_for_the_same_route():
    """A route sampled at two rates should give nearly the same descriptor."""
    fine = np.stack([np.linspace(0, 1, 100), np.linspace(0, 2, 100)], axis=-1)
    coarse = np.stack([np.linspace(0, 1, 20), np.linspace(0, 2, 20)], axis=-1)
    d = temporal.trajectory_descriptors([fine, coarse], n_resample=16)
    np.testing.assert_allclose(d[0], d[1], atol=1e-12)


def test_descriptors_handle_a_single_timepoint():
    d = temporal.trajectory_descriptors([np.ones((1, 2))], n_resample=4)
    assert d.shape == (1, 8) and np.all(d == 1.0)


def test_descriptors_reject_degenerate_resampling():
    with pytest.raises(ValueError, match="n_resample must be >= 2"):
        temporal.trajectory_descriptors([np.zeros((5, 2))], n_resample=1)


# --- pair selection --------------------------------------------------------


def test_none_returns_every_pair(sequences):
    K = len(sequences)
    pairs = temporal.candidate_pairs(sequences, n_candidates=None)
    assert pairs == [(i, j) for i in range(K) for j in range(i + 1, K)]
    assert len(pairs) == K * (K - 1) // 2


def test_screening_reduces_the_pair_count(sequences):
    K = len(sequences)
    screened = temporal.candidate_pairs(sequences, n_candidates=3)
    assert len(screened) < K * (K - 1) // 2


def test_every_trajectory_keeps_at_least_one_partner(sequences):
    """No trajectory may be dropped from the inter-trajectory graph entirely."""
    pairs = temporal.candidate_pairs(sequences, n_candidates=2)
    covered = {i for pair in pairs for i in pair}
    assert covered == set(range(len(sequences)))


def test_pairs_are_sorted_upper_triangular_and_unique(sequences):
    pairs = temporal.candidate_pairs(sequences, n_candidates=3)
    assert all(i < j for i, j in pairs)
    assert pairs == sorted(set(pairs))


def test_saturating_n_candidates_falls_back_to_all_pairs(sequences):
    K = len(sequences)
    everything = temporal.candidate_pairs(sequences, n_candidates=None)
    assert temporal.candidate_pairs(sequences, n_candidates=K - 1) == everything
    assert temporal.candidate_pairs(sequences, n_candidates=K + 50) == everything


def test_invalid_n_candidates_raises(sequences):
    with pytest.raises(ValueError, match="n_candidates must be >= 1"):
        temporal.candidate_pairs(sequences, n_candidates=0)


def test_screening_prefers_same_branch_partners():
    """The descriptor should recover the branch structure it is screening on."""
    seqs, labels, _ = simulate_branching_trajectories(
        n=5, length=30, branch_time=0.3, noise=0.05, seed=0, return_labels=True
    )
    pairs = temporal.candidate_pairs(seqs, n_candidates=3)
    same = sum(labels[i] == labels[j] for i, j in pairs)
    assert same / len(pairs) > 0.7


# --- integration -----------------------------------------------------------


def test_screened_graph_is_a_subgraph_with_measurable_recall(sequences):
    full = temporal.calculate_distance_matrix(sequences, DTWAlignment(), mask=True)
    screened = temporal.calculate_distance_matrix(
        sequences, DTWAlignment(), mask=True, n_candidates=3
    )
    assert screened.nnz < full.nnz
    recall = evaluate.edge_recall(full, screened)
    assert 0.0 < recall < 1.0
    # screening only removes edges; it must never invent one
    assert evaluate.edge_recall(screened, full) == pytest.approx(1.0)


def test_screened_graph_still_yields_a_valid_probability_matrix(sequences):
    dist = temporal.calculate_distance_matrix(
        sequences, DTWAlignment(), mask=True, n_candidates=2
    )
    prob = temporal.calculate_high_dimensional_probability_matrix(dist, 8).toarray()
    np.testing.assert_allclose(np.diag(prob), 1.0)
    np.testing.assert_allclose(prob, prob.T, rtol=1e-6, atol=1e-9)
    assert prob.min() >= 0.0 and prob.max() <= 1.0 + 1e-12


def test_temporalmap_default_is_all_pairs_and_option_reaches_the_graph(sequences):
    assert temporal.TemporalMAP().n_candidates is None
    a = temporal.TemporalMAP(n_components=2, random_state=0)
    b = temporal.TemporalMAP(n_components=2, random_state=0, n_candidates=2)
    for mapper in (a, b):
        mapper.fit(sequences, max_iterations=1)
    assert b.distance_matrix.nnz < a.distance_matrix.nnz
