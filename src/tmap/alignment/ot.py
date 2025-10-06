"""Optimal Transport alignment for sequence pairs.

This module provides OT-based alignment using entropic regularization and
Sinkhorn iterations.
"""
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from typing import Optional

from tmap import base
from tmap.alignment.masking import masked_path_from_transport_plan_LAP, masked_path_from_transport_plan_argmax


class OTAlignment(base.AlignmentBase):
    """Use Optimal Transport to perform alignment between two sequences.

    Computes optimal transport plan using entropic regularization and Sinkhorn
    algorithm, then extracts alignment path.

    Parameters
    ----------
    epsilon : float
        Entropic regularization parameter.
    sinkhorn_threshold : float
        Convergence threshold for Sinkhorn iterations.
    """

    def __init__(self, *, epsilon: float = 0.1, sinkhorn_threshold: float = 1e-8):
        """Initialize OT alignment with regularization parameters.

        Parameters
        ----------
        epsilon : float
            Entropic regularization parameter.
        sinkhorn_threshold : float
            Convergence threshold for Sinkhorn iterations.
        """
        self.epsilon = epsilon
        self.sinkhorn_threshold = sinkhorn_threshold

    def __call__(self, sequence_i: npt.NDArray, sequence_j: npt.NDArray, *, mask: bool = True):
        """Align two sequences using Optimal Transport.

        Parameters
        ----------
        sequence_i : npt.NDArray
            First sequence of shape (n, features).
        sequence_j : npt.NDArray
            Second sequence of shape (m, features).
        mask : bool
            If True, extract optimal path from transport plan. If False, return
            full transport plan matrix.

        Returns
        -------
        npt.NDArray
            OT alignment matrix. If mask=True, masked to optimal assignment path
            using Linear Assignment Problem. If mask=False, full transport plan.
        """
        x = jnp.array(sequence_i, dtype=jnp.float32)
        y = jnp.array(sequence_j, dtype=jnp.float32)
        geom = pointcloud.PointCloud(x, y, epsilon=self.epsilon)

        # Uniform marginal distributions for the two trajectories
        p_x = jnp.ones(x.shape[0], dtype=jnp.float32) / x.shape[0]
        p_y = jnp.ones(y.shape[0], dtype=jnp.float32) / y.shape[0]
        ot_problem = linear_problem.LinearProblem(geom, p_x, p_y)

        # Solve it using the Sinkhorn solver
        solver = sinkhorn.Sinkhorn(threshold=self.sinkhorn_threshold)
        ot_solution = solver(ot_problem)

        # Get the optimal transport plan
        transport_plan = np.asarray(ot_solution.matrix)

        if not mask:
            return transport_plan
        mask = masked_path_from_transport_plan_LAP(transport_plan)
        return mask

    @property
    def name(self) -> str:
        """Name of the alignment method.

        Returns
        -------
        str
            'OT'
        """
        return "OT"