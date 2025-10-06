"""Utilities for extracting optimal paths from alignment matrices.

This module provides functions for masking alignment matrices to optimal paths
using various strategies including DTW best path and Linear Assignment Problem.
"""
import numpy as np
import numpy.typing as npt

from scipy.optimize import linear_sum_assignment


def masked_path_from_DTW(paths, best_path) -> npt.NDArray:
    """Extract cost values along optimal DTW path.

    Parameters
    ----------
    paths : npt.NDArray
        DTW cost matrix.
    best_path : list of tuples
        Optimal warping path as list of (i, j) index pairs.

    Returns
    -------
    npt.NDArray
        Masked matrix with values only along the optimal path.
    """
    i, j = zip(*best_path)
    paths = paths[1:, 1:]
    masked = np.zeros(paths.shape)
    dpath = paths[i, j]  # - np.concatenate([[0,], paths[i, j]])[:-1]
    masked[i, j] = dpath
    return masked


def masked_path_from_transport_plan_LAP(transport_plan: npt.NDArray) -> npt.NDArray:
    """Extract optimal assignment from transport plan using Linear Assignment.

    Uses the Hungarian algorithm to solve the linear assignment problem, finding
    the optimal one-to-one correspondence that maximizes total transport mass.

    Parameters
    ----------
    transport_plan : npt.NDArray
        Optimal transport plan matrix.

    Returns
    -------
    npt.NDArray
        Masked matrix with values only at optimal assignment positions.
    """
    cost_matrix = -np.array(transport_plan)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    best_path = list(zip(row_ind, col_ind))
    mask = np.zeros_like(cost_matrix)
    mask[row_ind, col_ind] = -cost_matrix[row_ind, col_ind]
    return mask


def masked_path_from_transport_plan_argmax(transport_plan: npt.NDArray) -> npt.NDArray:
    """Extract greedy path from transport plan using argmax.

    For each row, selects the column with maximum transport mass. This is a
    simpler but potentially suboptimal alternative to LAP.

    Parameters
    ----------
    transport_plan : npt.NDArray
        Optimal transport plan matrix.

    Returns
    -------
    npt.NDArray
        Masked matrix with values only at argmax positions.
    """
    mask = np.zeros_like(transport_plan)
    for i in range(transport_plan.shape[0]):
        j = np.argmax(transport_plan[i, :])
        mask[i, j] = transport_plan[i, j]
    return mask
