"""Tests for the synthetic trajectory generators.

The generators are the substrate for every supervised evaluation, so their
reproducibility and their ground-truth labelling are themselves under test.
"""
import numpy as np
import pytest

from tmap import simulate


def test_default_return_type_is_a_plain_list():
    seqs = simulate.simulate_trajectories(n=2, seed=0)
    assert isinstance(seqs, list)
    assert len(seqs) == 6  # 3 regimes x n
    assert all(s.shape == (100, 3) for s in seqs)


def test_labels_identify_the_three_regimes():
    seqs, labels = simulate.simulate_trajectories(n=4, seed=0, return_labels=True)
    assert len(seqs) == len(labels) == 12
    assert labels.tolist() == [0] * 4 + [1] * 4 + [2] * 4


def test_seed_makes_generation_reproducible():
    a = simulate.simulate_trajectories(n=3, seed=42)
    b = simulate.simulate_trajectories(n=3, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_different_seeds_differ():
    a = simulate.simulate_trajectories(n=3, seed=1)
    b = simulate.simulate_trajectories(n=3, seed=2)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b))


def test_generation_does_not_disturb_the_global_rng():
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    simulate.simulate_trajectories(n=2)
    assert np.random.random() == before


def test_noise_breaks_rank_one_degeneracy():
    # without noise every feature column is the same signal (rank 1)
    clean = simulate.simulate_trajectories(n=1, seed=0)[0]
    assert np.linalg.matrix_rank(clean) == 1
    noisy = simulate.simulate_trajectories(n=1, seed=0, noise=0.5)[0]
    assert np.linalg.matrix_rank(noisy) == 3


def test_regimes_are_distinguishable_by_construction():
    seqs, labels = simulate.simulate_trajectories(n=5, seed=0, return_labels=True)
    amplitude = np.array([np.ptp(s[:, 0]) for s in seqs])
    undamped, damped, low = (amplitude[labels == g] for g in (0, 1, 2))
    assert undamped.mean() > damped.mean()  # damping shrinks the envelope
    assert low.mean() < damped.mean()       # low-amplitude is smallest


# --- unbalanced ------------------------------------------------------------


def test_unbalanced_trajectories_have_varying_lengths():
    seqs = simulate.simulate_unbalanced_trajectories(n=5, seed=0)
    lengths = {s.shape[0] for s in seqs}
    assert len(lengths) > 1


def test_unbalanced_lifetimes_match_sequence_lengths():
    seqs, labels, lifetimes = simulate.simulate_unbalanced_trajectories(
        n=4, seed=0, return_labels=True
    )
    assert len(seqs) == len(labels) == len(lifetimes) == 12
    for s, (start, stop) in zip(seqs, lifetimes):
        assert s.shape[0] == stop - start


def test_unbalanced_population_varies_over_time():
    # the whole point: mass is created and destroyed across the window
    _, _, lifetimes = simulate.simulate_unbalanced_trajectories(
        n=6, seed=0, return_labels=True
    )
    alive = [
        int(((lifetimes[:, 0] <= t) & (t < lifetimes[:, 1])).sum())
        for t in range(100)
    ]
    assert min(alive) < max(alive)


def test_unbalanced_respects_min_fraction():
    seqs = simulate.simulate_unbalanced_trajectories(
        n=5, seed=0, t=np.linspace(0, 10, 100), min_fraction=0.5
    )
    assert min(s.shape[0] for s in seqs) >= 50


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_unbalanced_rejects_invalid_min_fraction(bad):
    with pytest.raises(ValueError, match="min_fraction"):
        simulate.simulate_unbalanced_trajectories(min_fraction=bad)


def test_unbalanced_uses_absolute_time():
    # a trajectory born late continues the oscillation rather than restarting,
    # so its first sample generally differs from a trajectory born at t=0
    seqs, _, lifetimes = simulate.simulate_unbalanced_trajectories(
        n=8, seed=3, return_labels=True
    )
    late = [s for s, (start, _) in zip(seqs, lifetimes) if start > 0]
    assert late, "expected at least one late-born trajectory"
    assert not all(np.allclose(s[0], seqs[0][0]) for s in late)


# --- perturbations ---------------------------------------------------------


def test_add_observation_noise_is_reproducible_and_shape_preserving():
    seqs = simulate.simulate_trajectories(n=2, seed=0)
    a = simulate.add_observation_noise(seqs, scale=0.2, seed=1)
    b = simulate.add_observation_noise(seqs, scale=0.2, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert all(x.shape == s.shape for x, s in zip(a, seqs))
    assert not np.allclose(a[0], seqs[0])


def test_mis_warp_preserves_length_and_stalls_at_index():
    seq = np.arange(20.0)[:, None] * np.ones((1, 2))
    out = simulate.mis_warp(seq, index=5, repeat=3)
    assert out.shape == seq.shape
    # the stalled value is held for `repeat` frames
    assert np.allclose(out[5], seq[5]) and np.allclose(out[7], seq[5])
    # the tail resumes in time-alignment; the skipped frames are dropped
    assert np.allclose(out[8], seq[8])
    assert not np.allclose(out[6], seq[6])


@pytest.mark.parametrize("kwargs", [{"index": -1}, {"index": 99}, {"index": 2, "repeat": 0}])
def test_mis_warp_validates_arguments(kwargs):
    seq = np.zeros((10, 2))
    with pytest.raises(ValueError):
        simulate.mis_warp(seq, **kwargs)


# --- branching --------------------------------------------------------------


def test_branching_shapes_and_labels():
    seqs, labels, progress = simulate.simulate_branching_trajectories(
        n=3, n_branches=3, length=40, seed=0, return_labels=True
    )
    assert len(seqs) == 9
    assert all(s.shape == (40, 3) for s in seqs)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert progress.shape == (9, 40)


def test_branching_default_return_type_is_a_plain_list():
    seqs = simulate.simulate_branching_trajectories(n=2, seed=0)
    assert isinstance(seqs, list) and len(seqs) == 6


def test_branch_time_controls_path_dissimilarity():
    """The defining property: one knob spans identical -> fully dissimilar."""
    def across_branch_spread(branch_time):
        seqs, labels, _ = simulate.simulate_branching_trajectories(
            n=4, n_branches=2, branch_time=branch_time, noise=0.0, seed=0,
            return_labels=True,
        )
        ends = np.array([s[-1] for s in seqs])
        a, b = ends[labels == 0], ends[labels == 1]
        return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))

    identical, mixed, independent = (across_branch_spread(t) for t in (1.0, 0.5, 0.0))
    assert identical < mixed < independent
    assert identical == pytest.approx(0.0, abs=1e-9)  # one shared path


def test_branching_data_is_full_rank():
    # unlike the oscillator, branches occupy distinct directions in feature space
    seqs = simulate.simulate_branching_trajectories(n=3, noise=0.0, seed=0)
    assert np.linalg.matrix_rank(np.concatenate(seqs)) == 3


def test_progress_is_monotone_and_normalised():
    _, _, progress = simulate.simulate_branching_trajectories(
        n=4, seed=0, return_labels=True
    )
    assert np.all(np.diff(progress, axis=1) > 0)
    assert np.allclose(progress[:, 0], 0.0)
    assert np.allclose(progress[:, -1], 1.0)


def test_speed_jitter_creates_the_warping_problem():
    # jitter > 0: same geometry, different pacing -> DTW has real work to do
    _, _, jittered = simulate.simulate_branching_trajectories(
        n=2, speed_jitter=0.3, seed=0, return_labels=True
    )
    assert not np.allclose(jittered[0], jittered[1])

    # jitter == 0: every trajectory shares one time grid
    _, _, uniform = simulate.simulate_branching_trajectories(
        n=2, speed_jitter=0.0, seed=0, return_labels=True
    )
    assert np.allclose(uniform[0], uniform[1])


def test_progress_is_ground_truth_correspondence():
    """Equal progress on a shared path must mean an equal position."""
    seqs, _, progress = simulate.simulate_branching_trajectories(
        n=2, n_branches=2, branch_time=1.0, noise=0.0, speed_jitter=0.3,
        seed=0, return_labels=True,
    )
    # trajectory 0's midpoint, and wherever trajectory 1 reaches the same progress
    p = progress[0, 20]
    q = int(np.argmin(np.abs(progress[1] - p)))
    assert np.allclose(seqs[0][20], seqs[1][q], atol=1e-2)


def test_branching_is_reproducible():
    a = simulate.simulate_branching_trajectories(n=3, seed=11)
    b = simulate.simulate_branching_trajectories(n=3, seed=11)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_branching_does_not_disturb_the_global_rng():
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    simulate.simulate_branching_trajectories(n=2)
    assert np.random.random() == before


@pytest.mark.parametrize("kwargs,match", [
    ({"branch_time": 1.5}, "branch_time"),
    ({"branch_time": -0.1}, "branch_time"),
    ({"n_components": 1}, "n_components >= 2"),
    ({"n_branches": 3, "n_components": 2}, "fan out"),
    ({"speed_jitter": -0.1}, "speed_jitter"),
])
def test_branching_validates_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        simulate.simulate_branching_trajectories(n=2, **kwargs)
