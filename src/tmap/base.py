"""Base classes and constants for tmap.

This module defines abstract base classes for mappers, layout methods, and
alignment methods, along with default constants used throughout the package.
"""
import abc

import numpy as np
import numpy.typing as npt

EPSILON_WEIGHT = np.inf
N_NEIGHBORS = 15
N_COMPONENTS = 2
MIN_DIST = 0.01
LEARNING_RATE = 1e-1
MAX_ITERATIONS = 200
SIGMA_LOW_ESTIMATE = 0.0
SIGMA_HIGH_ESTIMATE = 1000.0


class MapperBase(abc.ABC):
    """Abstract base class for embedding/mapping algorithms.

    Defines the interface for algorithms that create low-dimensional embeddings
    of temporal sequence data.
    """

    @abc.abstractmethod
    def fit(self, sequences: list[npt.NDArray], **kwargs) -> npt.NDArray:
        """Fit the mapper to sequences and return embeddings.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences to embed.
        **kwargs
            Additional parameters specific to the implementation.

        Returns
        -------
        npt.NDArray
            Low-dimensional embeddings.
        """
        raise NotImplementedError

    @property
    def sequence_shapes(self) -> list[int]:
        """Shapes/Lengths of the sequences used.

        Returns
        -------
        list of int
            Length of each input sequence.
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
        """Distance matrix between sequences.

        Returns
        -------
        npt.NDArray or None
            Computed distance matrix, or None if not yet computed.
        """
        return self._distance_matrix

    @property
    def embeddings(self) -> npt.NDArray | None:
        """Low-dimensional embeddings.

        Returns
        -------
        npt.NDArray or None
            Computed embeddings, or None if not yet fitted.
        """
        return self._embedding


class LayoutBase(abc.ABC):
    """Abstract base class for initial layout methods.

    Layout methods provide initial positions for embeddings before optimization.
    """

    def __call__(self, *args, **kwargs):
        """Make the layout callable, delegating to fit_transform."""
        return self.fit_transform(*args, **kwargs)

    def _concatenate_sequences(self, sequences: list[npt.NDArray]) -> npt.NDArray:
        """Concatenate list of sequences into single array.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Sequences to concatenate.

        Returns
        -------
        npt.NDArray
            Concatenated array of all sequences.
        """
        return np.concatenate(sequences, axis=0)

    @abc.abstractmethod
    def fit_transform(
        self, sequences: list[npt.NDArray], *, n_components: int = N_COMPONENTS
    ) -> npt.NDArray:
        """Compute initial layout for sequences.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.
        n_components : int
            Number of dimensions for the layout.

        Returns
        -------
        npt.NDArray
            Initial positions in n_components dimensions.
        """
        raise NotImplementedError


class AlignmentBase(abc.ABC):
    """Abstract base class for sequence alignment methods.

    Alignment methods compute correspondence between pairs of sequences.
    """

    @abc.abstractmethod
    def __call__(self, sequence_i: npt.NDArray, sequence_j: npt.NDArray, *, mask: bool = True) -> npt.NDArray:
        """Align two sequences and return alignment matrix.

        Parameters
        ----------
        sequence_i : npt.NDArray
            First sequence.
        sequence_j : npt.NDArray
            Second sequence.
        mask : bool
            Whether to mask the alignment to an optimal path.

        Returns
        -------
        npt.NDArray
            Alignment or distance matrix between sequences.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the alignment method.

        Returns
        -------
        str
            Human-readable name of the alignment method.
        """
        raise NotImplementedError
