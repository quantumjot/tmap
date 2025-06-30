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

from tmap.flow import shepard_interp
from tmap.temporal import TemporalMAP


class PlotDimensionality(enum.IntEnum):
    TWO = 2 
    THREE = 3


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
