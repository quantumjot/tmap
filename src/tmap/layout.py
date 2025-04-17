import enum
from functools import partial

import numpy as np
from sklearn.manifold import SpectralEmbedding

from tmap.base import LayoutBase


class SpectralLayout(LayoutBase):
    def __init__(self):
        pass

    def fit_transform(self, x, *, n_components: int = 2):
        model = SpectralEmbedding(n_components=n_components) #, n_neighbors=x.shape[-1])
        return model.fit_transform(x)


class RandomLayout(LayoutBase):
    def __init__(self):
        pass

    def fit_transform(self, x, *, n_components: int = 2):
        return np.random.randn(x.shape[0], n_components)


class UMAPLayout(LayoutBase):
    def __init__(self):
        pass


class TemporalLayout(LayoutBase):
    def __init__(self):
        pass


class InitialLayout(enum.Enum):
    RANDOM = partial(RandomLayout())
    # UMAP = partial(UMAPLayout())
    SPECTRAL = partial(SpectralLayout())
    # TEMPORAL = partial(TemporalLayout())

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
