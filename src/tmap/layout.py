"""Initial layout methods for temporal embeddings.

This module provides various strategies for computing initial positions of
embeddings before optimization.
"""
import enum
import numpy as np
import numpy.typing as npt
import umap

from sklearn.manifold import SpectralEmbedding
from tmap.base import LayoutBase


class SpectralLayout(LayoutBase):
    """Basic spectral embedding for initial layout of features."""

    def __init__(self):
        """Initialize spectral layout method."""
        pass

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2):
        """Compute spectral embedding of concatenated sequences.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.
        n_components : int
            Number of embedding dimensions.

        Returns
        -------
        npt.NDArray
            Initial layout using spectral embedding.
        """
        x_cat = self._concatenate_sequences(sequences)
        model = SpectralEmbedding(n_components=n_components, n_neighbors=x_cat.shape[-1])
        return model.fit_transform(x_cat)


class RandomLayout(LayoutBase):
    """A random initial layout of features."""

    def __init__(self, random_state=None):
        """Initialize random layout method.

        Parameters
        ----------
        random_state : int, np.random.RandomState, or None
            Random state for reproducibility.
        """
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState()

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2):
        """Generate random initial positions.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.
        n_components : int
            Number of embedding dimensions.

        Returns
        -------
        npt.NDArray
            Random initial positions from standard normal distribution.
        """
        x_cat = self._concatenate_sequences(sequences)
        return self.random_state.randn(x_cat.shape[0], n_components)


class UMAPLayout(LayoutBase):
    """A UMAP embedding for initial layout of features."""

    def __init__(self, *, min_dist: float = 0.1, n_neighbors: int = 10):
        """Initialize UMAP layout method.

        Parameters
        ----------
        min_dist : float
            UMAP minimum distance parameter.
        n_neighbors : int
            UMAP number of neighbors parameter.
        """
        self._umap = umap.UMAP()
        self._min_dist = min_dist
        self._n_neighbors = n_neighbors

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2, ):
        """Compute UMAP embedding of concatenated sequences.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.
        n_components : int
            Number of embedding dimensions.

        Returns
        -------
        npt.NDArray
            Initial layout using UMAP.
        """
        x_cat = self._concatenate_sequences(sequences)
        y = self._umap.fit_transform(
            x_cat,
            min_dist=self._min_dist,
            n_neighbors=self._n_neighbors,
            n_components=n_components,

        )
        return y


class TemporalLayout(LayoutBase):
    """A temporal alignment of sequences for initial layout of features."""

    def __init__(self):
        """Initialize temporal layout method."""
        pass

    def fit_transform(self, sequences: list[npt.NDArray], *, n_components: int = 2):
        """Compute layout based on temporal clustering.

        Uses hierarchical clustering with DTW distances to order sequences
        temporally, with time progressing along one dimension.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.
        n_components : int
            Number of embedding dimensions (minimum 2).

        Returns
        -------
        npt.NDArray
            Initial layout with sequences ordered by similarity, time along
            second dimension.
        """
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
    """Enumeration of available initial layout methods.

    Provides convenient access to layout method instances.
    """
    RANDOM = RandomLayout
    UMAP = UMAPLayout
    SPECTRAL = SpectralLayout
    TEMPORAL = TemporalLayout
