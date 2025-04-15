import abc

import numpy as np
import numpy.typing as npt

EPSILON_WEIGHT = np.inf
N_NEIGHBORS = 15
N_COMPONENTS = 2
MIN_DIST = 0.01
LEARNING_RATE = 1e-1
MAX_ITERATIONS = 200



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
    def distance_matrix(self) -> npt.NDArray | None:
        return self._distance_matrix

    @property
    def embeddings(self) -> npt.NDArray | None:
        """Return the embeddings"""
        return self._embedding


class LayoutBase(abc.ABC):

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abc.abstractmethod
    def fit_transform(self, x: npt.NDArray, *, n_components: int = N_COMPONENTS) -> npt.NDArray:
        raise NotImplementedError
