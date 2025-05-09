import numpy as np
import numpy.typing as npt

from typing import Optional

from dtaidistance import dtw, dtw_ndim
from tmap import base
from tmap.alignment.masking import masked_path_from_DTW


class DTWAlignment(base.AlignmentBase):
    """Use DTW to perform the alignment between two sequences."""

    def __init__(self, *, window: Optional[int] = None):
        self.window = window

    def __call__(self, sequence_i: npt.NDArray, sequence_j: npt.NDArray) -> npt.NDArray:
        _, paths = dtw_ndim.warping_paths(sequence_i, sequence_j, window=self.window)
        best_path = dtw.best_path(paths)
        mask = masked_path_from_DTW(paths, best_path)
        return mask
    
    @property
    def name(self) -> str:
        return "DTW"