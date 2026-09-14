import abc

import numpy as np
import numpy.typing as npt

EPSILON_WEIGHT = np.inf
# Chosen to preserve tmap's historical *effective* neighbourhood while fixing
# what the parameter means. The old default (n_neighbors=15 with the
# self-membership counted toward the log2(k) target) gave an effective
# neighbourhood of 7.5; with count_self=False the target covers the neighbours
# alone, so 8 reproduces that operating point almost exactly and now carries
# UMAP's meaning. Note this is *not* UMAP's own default of 15 -- matching that
# would double the neighbourhood and cost measurable group purity, a change
# deferred until there is real-data evidence for it.
# Align each trajectory to its 10 nearest neighbours in descriptor space rather
# than to all K-1 others, making alignment O(K * c) instead of O(K ** 2). A
# no-op for K <= 11, since candidate_pairs falls back to all pairs once
# n_candidates >= K - 1, so this only engages where the quadratic cost bites.
# Measured on branching data at K=36: correspondence improves (should-match
# 0.195 -> 0.127) and K=120 runs 5.7x faster. Set None to align all pairs.
N_CANDIDATES = 10

N_NEIGHBORS = 8
N_COMPONENTS = 2
MIN_DIST = 0.01
LEARNING_RATE = 1e-1
LEARNING_RATE_SAMPLED = 1.0  # UMAP-conventional initial alpha, decays to zero
MAX_ITERATIONS = 200
N_NEGATIVE = 5
REPULSION_STRENGTH = 1.0
SIGMA_LOW_ESTIMATE = 0.0
SIGMA_HIGH_ESTIMATE = 1000.0


class MapperBase(abc.ABC):

    @abc.abstractmethod
    def fit(
        self,
        sequences: list[npt.NDArray],
        learning_rate: float = 0.00,
        max_iterations: int = 1,
    ) -> npt.NDArray:
        raise NotImplementedError

    @property
    def sequence_shapes(self) -> list[int]:
        """Shapes/Lengths of the sequences used.

        Returns
        -------
        """
        return [s.shape[0] for s in self._sequences]

    @property
    def trajectories(self) -> list[npt.NDArray]:
        """Trajectories in the low dimensional representation.

        Returns
        -------
        trajectories : list
            A list of numpy arrays of the low dimensional embeddings for each
            trajectory.
        """
        seq = self.sequence_shapes
        slice_seq = lambda idx: slice(sum(seq[:idx]), sum(seq[: idx + 1]), 1)
        return [self.embeddings[slice_seq(i), ...] for i in range(len(seq))]

    @property
    def distance_matrix(self):
        """The pairwise distance graph as ``scipy.sparse.csr_matrix``.

        Only edges are stored; absent entries denote unconnected pairs (the
        dense representation's ``inf``). Use :attr:`distance_matrix_dense`
        for the historical dense ``ndarray`` form.
        """
        return self._distance_matrix

    @property
    def distance_matrix_dense(self) -> npt.NDArray | None:
        """Dense ``ndarray`` form of :attr:`distance_matrix` (non-edges are
        ``inf``, diagonal zero)."""
        if self._distance_matrix is None:
            return None
        from tmap.temporal import densify_distance_matrix

        return densify_distance_matrix(self._distance_matrix)

    @property
    def embeddings(self) -> npt.NDArray | None:
        """Return the embeddings"""
        return self._embedding


class LayoutBase(abc.ABC):

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)
    
    def _concatenate_sequences(self, sequences: list[npt.NDArray]) -> npt.NDArray:
        return np.concatenate(sequences, axis=0)

    @abc.abstractmethod
    def fit_transform(
        self, sequences: list[npt.NDArray], *, n_components: int = N_COMPONENTS
    ) -> npt.NDArray:
        raise NotImplementedError
    

class AlignmentBase(abc.ABC):

    @abc.abstractmethod 
    def __call__(self, sequence_i: npt.NDArray, sequence_j: npt.NDArray, *, mask: bool = True) -> npt.NDArray:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError
