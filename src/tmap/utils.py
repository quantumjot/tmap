from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
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
    
    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(mapper.embeddings[:, 0], mapper.embeddings[:, 1], "k.")

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

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Time", rotation=270)
    ax.set_title(f"{title} | n_neighbors: {mapper.n_neighbors}, min_dist: {mapper.min_dist}, window: {mapper.window}")