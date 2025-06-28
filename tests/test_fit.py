import numpy as np
import pytest

from tmap import TemporalMAP
from tmap.layout import InitialLayout, SpectralLayout
from tmap.simulate import simulate_trajectories


@pytest.fixture
def sim_data():
    TEST_DATA = [
        np.random.random((15, 3)),
        np.random.random((10, 3)),
    ]
    return TEST_DATA


# @pytest.fixture
# def sim_data():
#     TEST_DATA = simulate_trajectories()
#     return TEST_DATA


def test_map(sim_data):
    t = TemporalMAP()
    y = t.fit(sim_data)
    assert y.ndim == 2


@pytest.mark.parametrize("layout", (layout for layout in InitialLayout))
def test_layout(sim_data, layout):
    t = TemporalMAP(layout=layout)
    y = t.fit(sim_data)
    assert y.ndim == 2


@pytest.mark.parametrize("layout", ("SPECTRAL", InitialLayout.SPECTRAL, SpectralLayout()))
def test_layout_type(sim_data, layout):
    t = TemporalMAP(layout=layout)
    y = t.fit(sim_data)
    assert y.ndim == 2


@pytest.mark.xfail
def test_layout_type_uninstantiated(sim_data):
    t = TemporalMAP(layout=SpectralLayout)
    y = t.fit(sim_data)
    assert y.ndim == 2


@pytest.mark.parametrize("ndim", (2, 3))
def test_ndim(sim_data, ndim):
    t = TemporalMAP(n_components=ndim)
    y = t.fit(sim_data)
    assert y.shape[-1] == ndim
