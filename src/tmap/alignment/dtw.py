import numpy as np
import numpy.typing as npt

from typing import Optional

from dtaidistance import dtw, dtw_ndim
from tmap import base
from tmap.alignment.masking import (
    masked_path_from_DTW,
    masked_path_triplets_from_DTW,
)


# Whether the dtaidistance compiled C extension is available. When it is, the
# ``*_fast`` implementations run in C and are ~100x faster than the pure-Python
# fallback; they also release the GIL, so thread-based parallelism helps. When
# it is not (e.g. libomp missing), we fall back to pure Python and callers
# should prefer process-based parallelism.
HAS_DTW_C = dtw.try_import_c(verbose=False)


class DTWAlignment(base.AlignmentBase):
    """Use DTW to perform the alignment between two sequences."""

    def __init__(self, *, window: Optional[int] = None):
        self.window = window
        # exposed so calculate_distance_matrix can pick threads (C, GIL released)
        # vs processes (pure Python) for parallel alignment.
        self.releases_gil = HAS_DTW_C

    def __call__(
        self,
        sequence_i: npt.NDArray,
        sequence_j: npt.NDArray,
        *,
        mask: bool = True,
        sparse: bool = False,
    ) -> npt.NDArray:
        """Align two sequences.

        With ``sparse=True`` (and ``mask=True``) the warping path is returned
        as ``(rows, cols, values)`` triplets instead of a dense block, so the
        caller can assemble a sparse graph without materialising rectangular
        matrices. ``sparse`` is ignored when ``mask=False`` (the full cost
        matrix is inherently dense).
        """
        warping_paths = dtw_ndim.warping_paths_fast if HAS_DTW_C else dtw_ndim.warping_paths
        # The compiled DTW requires C-contiguous float64 input; it rejects
        # other dtypes (e.g. float16/float32) that the pure-Python path
        # tolerated. Coerce here so both paths behave the same.
        sequence_i = np.ascontiguousarray(sequence_i, dtype=np.float64)
        sequence_j = np.ascontiguousarray(sequence_j, dtype=np.float64)
        _, paths = warping_paths(sequence_i, sequence_j, window=self.window)

        if not mask:
            return paths[1:, 1:]

        best_path = dtw.best_path(paths)
        if sparse:
            return masked_path_triplets_from_DTW(paths, best_path)
        return masked_path_from_DTW(paths, best_path)

    @property
    def name(self) -> str:
        return "DTW"