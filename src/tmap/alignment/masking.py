import numpy as np 
import numpy.typing as npt

from scipy.optimize import linear_sum_assignment


def masked_path_from_DTW(paths, best_path) -> npt.NDArray:
    """Calculate the adjacency matrix in high dimensional space.

    Parameters
    ----------

    Returns
    -------
    """
    i, j, dpath = masked_path_triplets_from_DTW(paths, best_path)
    masked = np.zeros(paths[1:, 1:].shape)
    masked[i, j] = dpath
    return masked


def masked_path_triplets_from_DTW(
    paths, best_path
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Warping-path cells as ``(rows, cols, values)`` triplets.

    Same values as :func:`masked_path_from_DTW` but without materialising the
    dense rectangular block — only the O(path length) nonzero cells.
    """
    i, j = zip(*best_path)
    i = np.asarray(i)
    j = np.asarray(j)
    dpath = paths[1:, 1:][i, j]  # - np.concatenate([[0,], paths[i, j]])[:-1]
    return i, j, dpath


def masked_path_from_transport_plan_LAP(transport_plan: npt.NDArray) -> npt.NDArray:
    """Masked path from transport plan using LAP"""
    row_ind, col_ind, values = masked_path_triplets_from_transport_plan_LAP(
        transport_plan
    )
    mask = np.zeros_like(np.asarray(transport_plan))
    mask[row_ind, col_ind] = values
    return mask


def masked_path_triplets_from_transport_plan_LAP(
    transport_plan: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """LAP assignment cells as ``(rows, cols, values)`` triplets."""
    cost_matrix = -np.array(transport_plan)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, -cost_matrix[row_ind, col_ind]


def masked_path_from_transport_plan_argmax(transport_plan: npt.NDArray) -> npt.NDArray:
    """Masked path from transport plan using argmax"""
    mask = np.zeros_like(transport_plan)
    for i in range(transport_plan.shape[0]):
        j = np.argmax(transport_plan[i, :])
        mask[i, j] = transport_plan[i, j]
    return mask
