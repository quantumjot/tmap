import numpy as np
import pytest

from tmap import InitialLayout, TemporalMAP

TEST_DATA = [
    np.random.random((15, 3)),
    np.random.random((10, 3)),
]


def test_map():
    t = TemporalMAP()
    y = t.fit(TEST_DATA)
    assert y.ndim == 2

@pytest.mark.parametrize("layout", (l for l in InitialLayout))
def test_layout(layout):
    t = TemporalMAP(layout=layout)
    y = t.fit(TEST_DATA)
    assert y.ndim == 2
