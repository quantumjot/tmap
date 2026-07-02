import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from typing import Optional

from tmap import base
from tmap.alignment.masking import (
    masked_path_from_transport_plan_LAP,
    masked_path_from_transport_plan_argmax,
    masked_path_triplets_from_transport_plan_LAP,
)


class OTAlignment(base.AlignmentBase):
    """Use OT to perform the alignment between two sequences.

    Parameters
    ----------
    epsilon : float
        Entropic regularisation of the Sinkhorn problem.
    sinkhorn_threshold : float
        Convergence threshold of the Sinkhorn solver.
    distance : str
        How the transported mass on the optimal path is converted to a
        distance in :func:`tmap.temporal.calculate_distance_matrix`:
        ``"inverse"`` (default, the historical ``1 / (mass + 1e-9)``) or
        ``"neglog"`` (``-log(mass + 1e-9)``, a Boltzmann-style cost that is
        additive over independent correspondences and far less
        scale-explosive for small masses).
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        sinkhorn_threshold: float = 1e-8,
        distance: str = "inverse",
    ):
        if distance not in ("inverse", "neglog"):
            raise ValueError(f"Unknown distance conversion: {distance!r}")
        self.epsilon = epsilon
        self.sinkhorn_threshold = sinkhorn_threshold
        self.distance = distance
        # Build the solver once and jit the solve: JAX caches the compiled
        # kernel per input shape, so aligning many equal-length pairs reuses
        # one trace instead of re-tracing solver internals on every call.
        self._solver = sinkhorn.Sinkhorn(threshold=sinkhorn_threshold)
        self._solve = jax.jit(self._transport_matrix)

    def _transport_matrix(self, x, y, p_x, p_y):
        geom = pointcloud.PointCloud(x, y, epsilon=self.epsilon)
        ot_problem = linear_problem.LinearProblem(geom, p_x, p_y)
        return self._solver(ot_problem).matrix

    def transport_mass_to_distance(self, mass: npt.NDArray) -> npt.NDArray:
        """Convert transported mass (correspondence strength) to a distance."""
        if self.distance == "neglog":
            return np.maximum(-np.log(mass + 1e-9), 1e-9).astype(np.float32)
        return (1.0 / (mass + 1e-9)).astype(np.float32)

    def __call__(
        self,
        sequence_i: npt.NDArray,
        sequence_j: npt.NDArray,
        *,
        mask: bool = True,
        sparse: bool = False,
    ):
        x = jnp.array(sequence_i, dtype=jnp.float32)
        y = jnp.array(sequence_j, dtype=jnp.float32)

        # Uniform marginal distributions for the two trajectories
        p_x = jnp.ones(x.shape[0], dtype=jnp.float32) / x.shape[0]
        p_y = jnp.ones(y.shape[0], dtype=jnp.float32) / y.shape[0]

        transport_plan = np.asarray(self._solve(x, y, p_x, p_y))

        if not mask:
            return transport_plan
        if sparse:
            return masked_path_triplets_from_transport_plan_LAP(transport_plan)
        return masked_path_from_transport_plan_LAP(transport_plan)

    @property
    def name(self) -> str:
        return "OT"
