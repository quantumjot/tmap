import abc
import enum

import numpy as np
import numpy.typing as npt


class PreEmbedding(str, enum.Enum):
    RANDOM = "random"
    UMAP = "umap"
    SPECTRAL = "spectral"


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


class PreEmbeddingFunction(abc.ABC):
    ...