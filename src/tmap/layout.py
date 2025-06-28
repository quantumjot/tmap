import enum
import numpy as np
import numpy.typing as npt
import umap

from functools import partial
from sklearn.manifold import SpectralEmbedding
from tmap.base import LayoutBase


class SpectralLayout(LayoutBase):
    """Basic spectral embedding for initial layout of features"""
    def __init__(self):
        pass

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2):
        x_cat = self._concatenate_sequences(sequences)
        model = SpectralEmbedding(n_components=n_components, n_neighbors=x_cat.shape[-1])
        return model.fit_transform(x_cat)


class RandomLayout(LayoutBase):
    """A random initial layout of features"""
    def __init__(self):
        pass

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2):
        x_cat = self._concatenate_sequences(sequences)
        return np.random.randn(x_cat.shape[0], n_components)


class UMAPLayout(LayoutBase):
    """A UMAP embedding for initial layout of features"""
    def __init__(self, *, min_dist: float = 0.1, n_neighbors: int = 10):
        self._umap = umap.UMAP()
        self._min_dist = min_dist 
        self._n_neighbors = n_neighbors

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2, ):
        x_cat = self._concatenate_sequences(sequences)
        y = self._umap.fit_transform(
            x_cat,
            min_dist=self._min_dist,
            n_neighbors=self._n_neighbors,
            n_components=n_components,

        )
        return y


class TemporalLayout(LayoutBase):
    """A temporal alignment of sequences for initial layout of features"""
    def __init__(self):
        pass

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2):

        from dtaidistance import clustering, dtw_ndim
        from scipy.cluster.hierarchy import leaves_list

        y = []
        linkage = clustering.LinkageTree(dtw_ndim.distance_matrix, {})
        Z = linkage.fit(sequences)
        cluster_idx = leaves_list(Z)

        for idx in cluster_idx:
            seq = sequences[idx]
            n = len(seq)
            _y = np.zeros((n, n_components))
            _y[:, 0] = idx 
            _y[:n, 1] = np.arange(n)
            y.append(_y)
        y = np.concatenate(y, axis=0)
        return y


class InitialLayout(enum.Enum):
    RANDOM = partial(RandomLayout())
    UMAP = partial(UMAPLayout())
    SPECTRAL = partial(SpectralLayout())
    TEMPORAL = partial(TemporalLayout())

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
