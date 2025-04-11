from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from tmap.flow import shepard_interp
from tmap.temporal import TemporalMAP


def plot_embeddings(
        mapper: TemporalMAP, 
        *, 
        fig: Figure | None = None, 
        ax: Axes | None = None,
        title: str = "",
    ) -> None:
    """Plot the embeddings.
    
    Parameters
    ----------
    mapper : TemporalMAP 
        An instance of the TemporalMAP.

    Returns
    -------
    None
    """
    if np.isnan(mapper.embeddings[0, 0]):
        return
    
    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(mapper.embeddings[:, 0], mapper.embeddings[:, 1], "k.")



    xx, yy = np.meshgrid(
        np.linspace(np.min(mapper.embeddings[:, 0]), np.max(mapper.embeddings[:, 0]), 50),
        np.linspace(np.min(mapper.embeddings[:, 1]), np.max(mapper.embeddings[:, 1]), 50),
        indexing="ij",
    )

    grid = np.concatenate(
        [xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=-1
    )

    # note(arl): this is a hack to get an approximate scaling for the vectors
    dx = np.max([np.ptp(mapper.embeddings[:, 0]), np.ptp(mapper.embeddings[:, 1])])

    vectors = shepard_interp(
        vectors_from_tracks(mapper.trajectories),
        grid=grid,
        max_radius=50*dx,
    )

    for traj in mapper.trajectories:
        x, y = traj[:, 0], traj[:, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # note(arl): this just scales the colormap to the trajectory
        # this should probably be adjusted to absolute time
        dydx = np.linspace(0, 1, len(x))
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='RdBu', norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        ax.quiver(
            grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1],
            angles="xy", scale_units="xy", scale=2, color="k", #zorder=1000,
        )

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Time", rotation=270)
    ax.set_title(f"{title} | n_neighbors: {mapper.n_neighbors}, min_dist: {mapper.min_dist}, window: {mapper.window}")


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
        # track_arr = np.stack(track_data, axis=-1)
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