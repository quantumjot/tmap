"""Phase 0 performance guardrails.

These tests lock in the *current* numerical behaviour of the hot pipeline
stages so that performance refactors (parallel DTW, vectorised sigma, sparse /
on-device embedding) can be proven to preserve results rather than silently
changing them.

The golden reference arrays in ``tests/fixtures/perf_golden.npz`` were
generated from the pre-optimisation implementation. When an optimisation
*intentionally* changes semantics (e.g. switching hard DTW for soft-DTW), the
fixtures must be regenerated deliberately:

    TMAP_UPDATE_GOLDEN=1 pytest tests/test_perf_equivalence.py

and the diff reviewed. An *unintended* change will fail these tests.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from scipy import sparse

from tmap import temporal
from tmap.alignment import DTWAlignment

FIXTURE = Path(__file__).parent / "fixtures" / "perf_golden.npz"

# Must match the parameters used to generate the fixture (stored in `meta`).
K, LENGTH, FEATURES, N_NEIGHBORS, SEED = 6, 20, 3, 5, 1234


def _make_sequences() -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [
        np.cumsum(rng.standard_normal((LENGTH, FEATURES)) * 0.1, axis=0)
        for _ in range(K)
    ]


def _compute_current():
    """Run the current pipeline and return *dense* dist/prob for comparison.

    The pipeline is sparse end-to-end since Phase 5; densifying recovers the
    historical representation (non-edges ``inf`` in dist, zero in prob) so the
    golden fixtures and invariants below are unchanged.
    """
    seqs = _make_sequences()
    dist = temporal.calculate_distance_matrix(seqs, DTWAlignment(), mask=True)
    prob = temporal.calculate_high_dimensional_probability_matrix(dist, N_NEIGHBORS)
    return temporal.densify_distance_matrix(dist), prob.toarray()


@pytest.fixture(scope="module")
def current():
    return _compute_current()


@pytest.fixture(scope="module")
def golden():
    if os.environ.get("TMAP_UPDATE_GOLDEN") == "1":
        dist, prob = _compute_current()
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            FIXTURE,
            dist=dist.astype(np.float64),
            prob=prob.astype(np.float64),
            meta=np.array([K, LENGTH, FEATURES, N_NEIGHBORS, SEED]),
        )
    if not FIXTURE.exists():
        pytest.skip("golden fixture missing; run with TMAP_UPDATE_GOLDEN=1")
    return np.load(FIXTURE)


# --- equivalence against the frozen reference ------------------------------


def test_distance_matrix_matches_golden(current, golden):
    dist, _ = current
    np.testing.assert_allclose(dist, golden["dist"], rtol=1e-5, atol=1e-6)


def test_probability_matrix_matches_golden(current, golden):
    _, prob = current
    np.testing.assert_allclose(prob, golden["prob"], rtol=1e-5, atol=1e-6)


def test_fixture_metadata_matches(golden):
    assert list(golden["meta"]) == [K, LENGTH, FEATURES, N_NEIGHBORS, SEED]


# --- invariants that any implementation must uphold ------------------------


def test_distance_matrix_symmetric(current):
    dist, _ = current
    finite = np.where(np.isinf(dist), 0.0, dist)
    np.testing.assert_allclose(finite, finite.T, rtol=1e-6, atol=1e-6)


def test_distance_matrix_zero_diagonal(current):
    dist, _ = current
    assert np.allclose(np.diag(dist), 0.0)


def test_probability_matrix_symmetric(current):
    _, prob = current
    np.testing.assert_allclose(prob, prob.T, rtol=1e-6, atol=1e-6)


def test_probability_matrix_in_unit_interval(current):
    _, prob = current
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0 + 1e-9


def test_pipeline_deterministic():
    """Two runs with identical inputs must produce identical outputs."""
    d1, p1 = _compute_current()
    d2, p2 = _compute_current()
    np.testing.assert_array_equal(np.where(np.isinf(d1), 0, d1), np.where(np.isinf(d2), 0, d2))
    np.testing.assert_array_equal(p1, p2)


# --- dense vs sparse equivalence --------------------------------------------


def test_sparse_dist_matches_dense_reference():
    """The sparse assembly must reproduce the dense semantics exactly."""
    seqs = _make_sequences()
    dist_sparse = temporal.calculate_distance_matrix(seqs, DTWAlignment(), mask=True)
    dist_dense = temporal._calculate_distance_matrix_dense(seqs, DTWAlignment(), mask=True)
    np.testing.assert_allclose(
        temporal.densify_distance_matrix(dist_sparse), dist_dense, rtol=1e-6, atol=0
    )


def test_sparse_probability_matches_dense_reference():
    """Sparse rho/sigma/probability must match the dense computation."""
    seqs = _make_sequences()
    dist_sparse = temporal.calculate_distance_matrix(seqs, DTWAlignment(), mask=True)
    dist_dense = temporal._calculate_distance_matrix_dense(seqs, DTWAlignment(), mask=True)

    prob_sparse = temporal.calculate_high_dimensional_probability_matrix(
        dist_sparse, N_NEIGHBORS
    )
    prob_dense = temporal.calculate_high_dimensional_probability_matrix(
        dist_dense, N_NEIGHBORS
    )

    assert sparse.issparse(prob_sparse)
    np.testing.assert_allclose(prob_sparse.toarray(), prob_dense, rtol=1e-5, atol=1e-8)


# --- embedding optimisation guardrail --------------------------------------


def test_embedding_loss_decreases_and_is_deterministic():
    """The optimiser must reduce the loss and be reproducible for a fixed seed.

    This guards the planned move to an on-device loop / Adam / sparse edges:
    those may change the loss trajectory, but the loss must still fall and the
    result must stay deterministic for a fixed initialisation.
    """
    import jax

    _, prob = _compute_current()
    a, b = temporal.find_hyperparameters(0.01)
    grad_fn = jax.value_and_grad(temporal.jax_cross_entropy_gradient_2, argnums=1)

    def run():
        rng = np.random.default_rng(0)
        y = rng.standard_normal((prob.shape[0], 2))
        losses = []
        for _ in range(50):
            loss, grad = grad_fn(prob, y, a, b)
            losses.append(float(loss))
            y = y - 0.1 * grad
        return np.asarray(y), losses

    y1, losses1 = run()
    y2, losses2 = run()

    assert losses1[-1] < losses1[0]  # loss decreased
    np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)  # deterministic


# --- aligner input-dtype robustness ----------------------------------------


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_dtw_alignment_accepts_non_float64_input(dtype):
    """The compiled DTW fast path only accepts float64; the aligner must coerce
    other dtypes (float16/float32) that the pure-Python path used to tolerate,
    and return the same result regardless of input dtype.
    """
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    b = np.cumsum(rng.standard_normal((30, 3)), axis=0)

    ref = DTWAlignment()(a.astype(np.float64), b.astype(np.float64))
    out = DTWAlignment()(a.astype(dtype), b.astype(dtype))

    assert out.shape == ref.shape
    if dtype is np.float64:
        np.testing.assert_array_equal(out, ref)
