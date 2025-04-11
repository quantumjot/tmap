import numpy as np
import numpy.typing as npt

from scipy.spatial import KDTree
from tqdm import tqdm


def _calculate_distance_matrix(x: npt.NDArray, xy: npt.NDArray) -> npt.NDArray:
    """Calculate a Euclidean distance matrix."""
    d_xy = xy - np.broadcast_to(x, xy.shape)
    d = np.linalg.norm(d_xy, axis=-1)
    return d


def _calculate_shepard_weights(
    x: npt.NDArray, xy: npt.NDArray, max_radius: float, power: int
) -> npt.NDArray:
    """Calculate Shepard weights."""
    d = _calculate_distance_matrix(x, xy)
    weights = np.power(
        np.clip(max_radius - d, 0, np.inf) / (max_radius * d), power
    )
    return weights


def shepard_interp(
    vectors: npt.NDArray,
    grid: npt.NDArray,
    *,
    max_radius: float = 100.0,
    power: int = 2,
) -> npt.NDArray:
    """Modified Shepard's method of interpolating vectors.

    Parameters
    ----------
    vectors : array
        An array of vectors (NxD), where D is an even number. Data are stored
        as [xyuv] for 2D data, or [xyzuvw] for 3D, and so on.
    grid : array
        An array of points (NxD), where D is the number of spatial dimenisons,
        at which to interpolate the vector field.
    max_radius : float
        The maximum radius over which to interpolate. In units of the data.
    power : int
        An exponent used to scale the Shepard weights.

    Returns
    -------
    interpolated : array
        An array of interpolated vectors.
    """

    if vectors.ndim != 2 or vectors.shape[-1] % 2 != 0:
        raise ValueError("``vectors`` must be a 2D array.")

    if grid.ndim != 2:
        raise ValueError("``grid`` must be a 2D array.")

    xy, uv = np.split(vectors, 2, axis=1)
    tree = KDTree(xy)
    queries = tree.query_ball_point(grid, max_radius)
    interpolated = np.zeros(grid.shape)
    nq = grid.shape[0]

    for idx, query in tqdm(enumerate(queries), desc="Interpolation", total=nq):
        if not query:
            continue

        # grid point to interpolate to
        x = grid[idx, ...]

        # nearest real locations
        x_xy = xy[query, ...]

        # vectors at the real locations
        x_uv = uv[query, ...]

        # calculate the weights and interpolate the vectors
        weights = _calculate_shepard_weights(x, x_xy, max_radius, power)
        sum_x = np.sum(weights)
        sum_u = np.sum(weights[:, np.newaxis] * x_uv, axis=0)
        interpolated[idx, ...] = sum_u / sum_x

    return interpolated
