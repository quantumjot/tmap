"""UMAP-faithfulness options: symmetrisation and the sigma bandwidth target.

Two documented departures from UMAP's construction are now selectable. Both
default to the historical behaviour, so these tests pin down what each option
actually changes as well as the equivalence of the dense and sparse paths under
every combination.
"""
import itertools

import numpy as np
import pytest

from scipy import sparse

from tmap import temporal
from tmap.alignment import DTWAlignment
from tmap.simulate import simulate_branching_trajectories

N_NEIGHBORS = 15


@pytest.fixture(scope="module")
def sequences():
    return simulate_branching_trajectories(n=3, length=25, noise=0.05, seed=0)


@pytest.fixture(scope="module")
def graphs(sequences):
    """The same distance graph in sparse and dense form."""
    return (
        temporal.calculate_distance_matrix(sequences, DTWAlignment(), mask=True),
        temporal._calculate_distance_matrix_dense(sequences, DTWAlignment(), mask=True),
    )


# --- symmetrisation --------------------------------------------------------


def test_fuzzy_union_is_the_probabilistic_t_conorm():
    v = np.array([[1.0, 0.5], [0.25, 1.0]])
    expected = v + v.T - v * v.T
    np.testing.assert_allclose(temporal.symmetrize_fuzzy_union(v), expected)


def test_mean_is_the_arithmetic_average():
    v = np.array([[1.0, 0.5], [0.25, 1.0]])
    np.testing.assert_allclose(temporal.symmetrize_mean(v), (v + v.T) / 2)


def test_both_symmetrisations_preserve_a_unit_diagonal():
    v = np.array([[1.0, 0.5], [0.25, 1.0]])
    for fn in (temporal.symmetrize_mean, temporal.symmetrize_fuzzy_union):
        np.testing.assert_allclose(np.diag(fn(v)), 1.0)


def test_fuzzy_union_stays_within_the_unit_interval():
    rng = np.random.default_rng(0)
    v = rng.random((20, 20))
    out = temporal.symmetrize_fuzzy_union(v)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_historical_aliases_preserve_their_old_behaviour():
    """The legacy names are inverted vs the literature; keep them behavioural."""
    v = np.array([[1.0, 0.5], [0.25, 1.0]])
    # ..._tsne historically computed the fuzzy union (which is UMAP's choice)
    np.testing.assert_allclose(
        temporal.symmetrize_probability_matrix_tsne(v), temporal.symmetrize_fuzzy_union(v)
    )
    # ..._umap historically computed the mean (which is the t-SNE choice)
    np.testing.assert_allclose(
        temporal.symmetrize_probability_matrix_umap(v), temporal.symmetrize_mean(v)
    )


def test_unknown_symmetrize_raises(graphs):
    dist_sparse, _ = graphs
    with pytest.raises(ValueError, match="unknown symmetrize"):
        temporal.calculate_high_dimensional_probability_matrix(
            dist_sparse, N_NEIGHBORS, symmetrize="nope"
        )


def test_union_and_mean_give_different_graphs(graphs):
    dist_sparse, _ = graphs
    mean = temporal.calculate_high_dimensional_probability_matrix(
        dist_sparse, N_NEIGHBORS, symmetrize="mean"
    ).toarray()
    union = temporal.calculate_high_dimensional_probability_matrix(
        dist_sparse, N_NEIGHBORS, symmetrize="union"
    ).toarray()
    assert not np.allclose(mean, union)
    # the union is a t-conorm, so it never falls below the mean
    assert np.all(union >= mean - 1e-12)


# --- the sigma bandwidth target -------------------------------------------


def _neighbour_membership_sum(dist_dense, n_neighbors, count_self):
    """Off-diagonal membership sum per row, i.e. what UMAP targets."""
    rho = np.partition(dist_dense, 1, axis=1)[:, 1]
    d = dist_dense - rho[:, None]
    sigma = temporal.estimate_sigma_vectorized(d, n_neighbors, count_self=count_self)
    v = np.exp(-np.clip(d, 0.0, np.inf) / sigma[:, None])
    return (v.sum(axis=1) - np.diag(v)).mean()


@pytest.mark.parametrize("n_neighbors", [5, 15, 30])
def test_count_self_false_targets_log2_k_over_neighbours(graphs, n_neighbors):
    """UMAP's actual target: the neighbours alone sum to log2(k)."""
    _, dist_dense = graphs
    got = _neighbour_membership_sum(dist_dense, n_neighbors, count_self=False)
    assert got == pytest.approx(np.log2(n_neighbors), abs=0.02)


@pytest.mark.parametrize("n_neighbors", [5, 15, 30])
def test_count_self_true_halves_the_effective_neighbourhood(graphs, n_neighbors):
    """The historical behaviour lands on log2(k) - 1 == log2(k / 2)."""
    _, dist_dense = graphs
    got = _neighbour_membership_sum(dist_dense, n_neighbors, count_self=True)
    assert got == pytest.approx(np.log2(n_neighbors) - 1.0, abs=0.02)
    # stated as an effective neighbour count, that is exactly half
    assert 2.0**got == pytest.approx(n_neighbors / 2, rel=0.02)


def test_count_self_changes_sigma(graphs):
    _, dist_dense = graphs
    rho = np.partition(dist_dense, 1, axis=1)[:, 1]
    d = dist_dense - rho[:, None]
    a = temporal.estimate_sigma_vectorized(d, N_NEIGHBORS, count_self=True)
    b = temporal.estimate_sigma_vectorized(d, N_NEIGHBORS, count_self=False)
    assert not np.allclose(a, b)
    assert np.all(b > a)  # excluding self needs a wider bandwidth


def test_scalar_estimate_sigma_agrees_with_vectorised(graphs):
    """estimate_sigma is unused in the pipeline but documented as equivalent."""
    _, dist_dense = graphs
    rho = np.partition(dist_dense, 1, axis=1)[:, 1]
    d = dist_dense - rho[:, None]
    for count_self in (True, False):
        vec = temporal.estimate_sigma_vectorized(d, N_NEIGHBORS, count_self=count_self)
        for row in (0, 5, 11):
            scalar = temporal.estimate_sigma(
                d[row], N_NEIGHBORS, count_self=count_self
            )
            assert scalar == pytest.approx(vec[row], rel=1e-6)


# --- dense / sparse equivalence under every combination -------------------


@pytest.mark.parametrize(
    "symmetrize,count_self", list(itertools.product(("mean", "union"), (True, False)))
)
def test_sparse_matches_dense_for_every_combination(graphs, symmetrize, count_self):
    dist_sparse, dist_dense = graphs
    kwargs = {"symmetrize": symmetrize, "count_self": count_self}
    prob_sparse = temporal.calculate_high_dimensional_probability_matrix(
        dist_sparse, N_NEIGHBORS, **kwargs
    )
    prob_dense = temporal.calculate_high_dimensional_probability_matrix(
        dist_dense, N_NEIGHBORS, **kwargs
    )
    assert sparse.issparse(prob_sparse)
    np.testing.assert_allclose(
        prob_sparse.toarray(), prob_dense, rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize(
    "symmetrize,count_self", list(itertools.product(("mean", "union"), (True, False)))
)
def test_graph_invariants_hold_for_every_combination(graphs, symmetrize, count_self):
    dist_sparse, _ = graphs
    prob = temporal.calculate_high_dimensional_probability_matrix(
        dist_sparse, N_NEIGHBORS, symmetrize=symmetrize, count_self=count_self
    ).toarray()
    np.testing.assert_allclose(np.diag(prob), 1.0)
    np.testing.assert_allclose(prob, prob.T, rtol=1e-6, atol=1e-9)
    assert prob.min() >= 0.0 and prob.max() <= 1.0 + 1e-12


# --- defaults and plumbing ------------------------------------------------


def test_defaults_are_the_historical_behaviour(graphs):
    dist_sparse, _ = graphs
    default = temporal.calculate_high_dimensional_probability_matrix(
        dist_sparse, N_NEIGHBORS
    ).toarray()
    explicit = temporal.calculate_high_dimensional_probability_matrix(
        dist_sparse, N_NEIGHBORS, symmetrize="mean", count_self=True
    ).toarray()
    np.testing.assert_array_equal(default, explicit)


def test_temporalmap_exposes_and_validates_the_options():
    assert temporal.TemporalMAP().symmetrize == "mean"
    assert temporal.TemporalMAP().count_self is True
    mapper = temporal.TemporalMAP(symmetrize="union", count_self=False)
    assert mapper.symmetrize == "union" and mapper.count_self is False
    with pytest.raises(ValueError, match="unknown symmetrize"):
        temporal.TemporalMAP(symmetrize="nope")


def test_temporalmap_options_reach_the_graph(sequences):
    a = temporal.TemporalMAP(n_components=2, random_state=0)
    b = temporal.TemporalMAP(
        n_components=2, random_state=0, symmetrize="union", count_self=False
    )
    for mapper in (a, b):
        mapper.fit(sequences, max_iterations=1)
    assert not np.allclose(a._P.toarray(), b._P.toarray())
