"""Layout reproducibility and parameter-threading guardrails."""
import numpy as np
import pytest

import tmap  # noqa: F401  (importing the package must not touch global RNG state)
from tmap.layout import RandomLayout, SpectralLayout


@pytest.fixture
def sequences():
    rng = np.random.default_rng(11)
    return [rng.random((15, 3)), rng.random((10, 3))]


def test_import_does_not_seed_global_rng():
    """Importing tmap must not reset numpy's global random state."""
    state_a = np.random.get_state()[1]
    import importlib

    import tmap.temporal

    importlib.reload(tmap.temporal)
    state_b = np.random.get_state()[1]
    np.testing.assert_array_equal(state_a, state_b)


def test_random_layout_reproducible_with_seed(sequences):
    y1 = RandomLayout()(sequences, n_components=2, random_state=42)
    y2 = RandomLayout()(sequences, n_components=2, random_state=42)
    np.testing.assert_array_equal(y1, y2)


def test_random_layout_differs_without_seed(sequences):
    y1 = RandomLayout()(sequences, n_components=2)
    y2 = RandomLayout()(sequences, n_components=2)
    assert not np.allclose(y1, y2)


def test_spectral_layout_uses_sample_based_neighbors(sequences):
    """n_neighbors must be a neighbour count, not the feature dimensionality.

    The old implementation passed ``x_cat.shape[-1]`` (= 3 features) as
    ``n_neighbors``; the layout must now accept and respect a real neighbour
    count, clipped to the number of samples.
    """
    y = SpectralLayout()(
        sequences, n_components=2, n_neighbors=10, random_state=0
    )
    assert y.shape == (25, 2)
    # a neighbour count larger than the sample count must be clipped, not crash
    y = SpectralLayout()(
        sequences, n_components=2, n_neighbors=1000, random_state=0
    )
    assert y.shape == (25, 2)


def test_spectral_layout_reproducible_with_seed(sequences):
    y1 = SpectralLayout()(sequences, n_components=2, n_neighbors=10, random_state=0)
    y2 = SpectralLayout()(sequences, n_components=2, n_neighbors=10, random_state=0)
    np.testing.assert_allclose(y1, y2)


def test_fit_reproducible_with_random_state(sequences):
    from tmap import TemporalMAP

    def run():
        tm = TemporalMAP(
            n_neighbors=5, layout="RANDOM", optimizer="sampled", random_state=7
        )
        return tm.fit(sequences, max_iterations=20)

    np.testing.assert_array_equal(run(), run())
