"""Utility functions for visualization and plotting.

This module provides plotting functions for visualizing temporal embeddings,
including flow fields and trajectory visualization.
"""
from __future__ import annotations

import enum
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import KDTree
from tqdm import tqdm

from tmap.temporal import TemporalMAP


class PlotDimensionality(enum.IntEnum):
    """Enumeration of supported plot dimensionalities."""
    TWO = 2
    THREE = 3


def _calculate_distance_matrix(x: npt.NDArray, xy: npt.NDArray) -> npt.NDArray:
    """Calculate Euclidean distance from point x to all points in xy.

    Parameters
    ----------
    x : npt.NDArray
        Single point.
    xy : npt.NDArray
        Array of points.

    Returns
    -------
    npt.NDArray
        Distances from x to each point in xy.
    """
    d_xy = xy - np.broadcast_to(x, xy.shape)
    d = np.linalg.norm(d_xy, axis=-1)
    return d


def _calculate_shepard_weights(
    x: npt.NDArray, xy: npt.NDArray, max_radius: float, power: int
) -> npt.NDArray:
    """Calculate Shepard interpolation weights with distance decay.

    Parameters
    ----------
    x : npt.NDArray
        Query point.
    xy : npt.NDArray
        Data points.
    max_radius : float
        Maximum influence radius.
    power : int
        Power for distance weighting.

    Returns
    -------
    npt.NDArray
        Interpolation weights for each point in xy.
    """
    d = _calculate_distance_matrix(x, xy)
    weights = np.power(np.clip(max_radius - d, 0, np.inf) / (max_radius * d), power)
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


def plot_embeddings(
    mapper: TemporalMAP,
    *,
    fig: Figure | None = None,
    ax: Axes | Axes3D | None = None,
    show_flow: bool = True,
    show_labels: bool = False,
    show_markers: bool = True,
    title: str = "tmap",
    cmap: str = "RdBu",
) -> None:
    """Plot the embeddings.

    Parameters
    ----------
    mapper : TemporalMAP
        An instance of the TemporalMAP.
    fig : Figure
        [optional] An instance of a matplotlib figure
    ax : Axes
        [optional] An instance of a matplotlib axes
    show_flow : bool
        Whether to show the flow field.
    show_labels : bool
        Whether to show sequence ID markers.
    show_markers : bool
        Whether to show points representing the individual data points.
    title : str
        The title of the plot.
    cmap : str 
        A colormap for the trajectories.

    Returns
    -------
    None
    """

    # silently return if there are issues here
    if np.isnan(mapper.embeddings[0, 0]) or mapper.embeddings is None:
        raise ValueError("Cannot plot embeddings.")
    
    if mapper.n_components not in PlotDimensionality._value2member_map_:
        raise ValueError(f"Too many components to plot: {mapper.n_components}")

    # set the plot dimensionality specifics
    if mapper.n_components == PlotDimensionality.THREE:
        subplot_kwargs = {"projection": "3d"}
        quiver_kwargs = {"length": 0.5, "normalize": True}
        quiver_subdiv = 10
        lc_fn = Line3DCollection
    else:
        subplot_kwargs = {}
        quiver_kwargs = {"angles": "xy", "scale_units": "xy", "scale": 2}
        quiver_subdiv = 50
        lc_fn = LineCollection

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        ax = None

    if ax is None:
        ax = fig.add_subplot(111, **subplot_kwargs)

    # split the embeddings by component
    embeddings = np.hsplit(mapper.embeddings, mapper.n_components)

    if show_markers:
        ax.plot(*embeddings, "k.")

    assert mapper.embeddings.shape[-1] == mapper.n_components

    if show_flow:
        grid_axes = [np.linspace(np.min(p), np.max(p), quiver_subdiv) for p in embeddings]
        grids = np.meshgrid(*grid_axes, indexing="ij")
        grid = np.concatenate(
            [xyz.ravel().reshape(-1, 1) for xyz in grids], axis=-1
        )

        # note(arl): this is a hack to get an approximate scaling for the vectors
        dx = np.max([np.ptp(p) for p in embeddings])

        vectors = shepard_interp(
            vectors_from_tracks(mapper.trajectories),
            grid=grid,
            max_radius=50 * dx,
        )

        ax.quiver(
            *np.hsplit(grid, mapper.n_components),
            *np.hsplit(vectors, mapper.n_components),
            color="k",
            zorder=-1000,
            alpha=0.2,
            **quiver_kwargs,
        )

    for idx, traj in enumerate(mapper.trajectories):
        # get the points and segments
        points = traj[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # note(arl): this just scales the colormap to the trajectory
        # this should probably be adjusted to absolute time
        dydx = np.linspace(0, 1, traj.shape[0])
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = lc_fn(segments, cmap=cmap, norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        # show markers at the midpoint of the trajectory
        if show_labels:
            j = traj.shape[0] // 2
            centroid = tuple(traj[j, ...].tolist())
            ax.text(
                *centroid, 
                f"{idx}",
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=2),
                zorder=1000,
            )

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Time", rotation=270)
    ax.set_title(
        f"{title} | n_neighbors: {mapper.n_neighbors}, min_dist: {mapper.min_dist}"
    )


def vectors_from_tracks(
    trajectories: list[npt.NDArray],
) -> np.ndarray:
    """Make an array of vectors from a list of tracks.

    Parameters
    ----------
    trajectories : list
        A list of low-dimensional trajectories.

    Returns
    -------
    vectors : array
        An array of vectors (NxD), where D is an even number. Data are stored
        as [xyuv] for 2D data, or [xyzuvw] for 3D, and so on.
    """

    vectors = []
    for track_arr in trajectories:
        t = np.arange(0, track_arr.shape[0])[:, None]
        track_arr = np.concatenate(
            [t, track_arr],
            axis=-1,
        )
        d = np.diff(track_arr, n=1, axis=0)

        # scale the vector by dt
        d[:, 1:] = d[:, 1:] * (1.0 / d[:, 0:1])

        # make the vector as [x, y, u, v]
        vec = np.concatenate(
            [
                track_arr[:-1, 1:],
                d[:, 1:],
            ],
            axis=-1,
        )
        vectors.append(vec)
    return np.concatenate(vectors, axis=0)
