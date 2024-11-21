from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from tmap.temporal import TemporalMAP


def plot_embeddings(mapper: TemporalMAP) -> None:
    """Plot the embeddings.
    
    Parameters
    ----------
    mapper : TemporalMAP 
        An instance of the TemporalMAP.

    Returns
    -------
    None
    """

    _y = mapper.embeddings

    fig, axs = plt.subplots()
    axs.plot(_y[:, 0], _y[:, 1], "k.")

    for i, seq in enumerate(mapper.sequence_shapes):
        start_idx = sum(mapper.sequence_shapes[:max(0, i)])
        end_idx = start_idx + seq 
        s = slice(start_idx, end_idx, 1)
        x, y = _y[s, 0], _y[s, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # note(arl): this just scales the colormap to the trajectory
        # this should probably be adjusted to absolute time
        dydx = np.linspace(0, 1, len(x))
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='RdBu', norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs.add_collection(lc)


    cbar = fig.colorbar(line, ax=axs)
    cbar.set_label("Time", rotation=270)
    axs.set_title(f"n_neighbors: {mapper.n_neighbors}, min_dist: {mapper.min_dist}, window: {mapper.window}")

    plt.show()