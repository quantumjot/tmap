import numpy as np

from tmap import TemporalMAP 


TEST_DATA = [
    np.random.random((15, 3)),
    np.random.random((10, 3)),
]


def test_map():
    t = TemporalMAP()
    y = t.fit(TEST_DATA)
    assert y.ndim == 2