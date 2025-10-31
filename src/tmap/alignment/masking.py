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
    i, j = zip(*best_path)
    paths = paths[1:, 1:]
    masked = np.zeros(paths.shape)
    dpath = paths[i, j]  # - np.concatenate([[0,], paths[i, j]])[:-1]
    masked[i, j] = dpath
    return masked


def masked_path_from_transport_plan_LAP(transport_plan: npt.NDArray) -> npt.NDArray:
    """Masked path from transport plan using LAP"""
    cost_matrix = -np.array(transport_plan)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    best_path = list(zip(row_ind, col_ind))
    mask = np.zeros_like(cost_matrix)
    mask[row_ind, col_ind] = -cost_matrix[row_ind, col_ind]
    return mask


def masked_path_from_transport_plan_argmax(transport_plan: npt.NDArray) -> npt.NDArray:
    """Masked path from transport plan using argmax"""
    mask = np.zeros_like(transport_plan)
    for i in range(transport_plan.shape[0]):
        j = np.argmax(transport_plan[i, :])
        mask[i, j] = transport_plan[i, j]
    return mask
