"""Dynamic Time Warping alignment for sequence pairs.

This module provides DTW-based alignment for temporal sequences.
"""
import numpy as np
import numpy.typing as npt

from typing import Optional

from dtaidistance import dtw, dtw_ndim
from tmap import base
from tmap.alignment.masking import masked_path_from_DTW


class DTWAlignment(base.AlignmentBase):
    """Use DTW to perform the alignment between two sequences.

    Parameters
    ----------
    window : int, optional
        Sakoe-Chiba window size to constrain the warping path. If None, no
        constraint is applied.
    """

    def __init__(self, *, window: Optional[int] = None):
        """Initialize DTW alignment with optional window constraint.

        Parameters
        ----------
        window : int, optional
            Sakoe-Chiba window size.
        """
        self.window = window

    def __call__(self, sequence_i: npt.NDArray, sequence_j: npt.NDArray, *, mask: bool = True) -> npt.NDArray:
        """Align two sequences using DTW.

        Parameters
        ----------
        sequence_i : npt.NDArray
            First sequence of shape (n, features).
        sequence_j : npt.NDArray
            Second sequence of shape (m, features).
        mask : bool
            If True, return only the optimal path. If False, return full cost matrix.

        Returns
        -------
        npt.NDArray
            DTW alignment matrix. If mask=True, masked to optimal path. If mask=False,
            full warping paths cost matrix.
        """
        _, paths = dtw_ndim.warping_paths(sequence_i, sequence_j, window=self.window)

        if not mask:
            return paths[1:, 1:]

        best_path = dtw.best_path(paths)
        mask = masked_path_from_DTW(paths, best_path)
        return mask

    @property
    def name(self) -> str:
        """Name of the alignment method.

        Returns
        -------
        str
            'DTW'
        """
        return "DTW"