"""Simulation utilities for generating test trajectory data.

This module provides functions for simulating temporal trajectories using
physical models like damped oscillators.
"""
import numpy as np
import numpy.typing as npt


def damped_oscillator(
    t: npt.NDArray,
    *,
    x_0: float = 10.0,
    mu: float = 0.1,
    omega_d: float = 10.0,
    omega_0: float = 10.0,
    phi: float = 0.0,
) -> npt.NDArray:
    """Damped oscillator.

    x(t) is the displacement at time
    x0​ is the initial displacement,
    μ is the damping ratio,
    ω0 is the undamped angular frequency,
    ωd​ is the damped angular frequency, and
    ϕ is the phase angle.
    """

    return x_0 * np.exp(-mu * omega_0 * t) * np.cos(omega_d * t + phi)


def simulate_trajectories(
    *, t: npt.NDArray = np.linspace(0, 10, 100), n: int = 10, n_components: int = 3
) -> list[npt.NDArray]:
    """Generate simulated trajectories using damped oscillators.

    Creates three groups of trajectories with different dynamics:
    - Undamped oscillators
    - Damped oscillators
    - Low amplitude undamped oscillators

    Parameters
    ----------
    t : npt.NDArray
        Time points for simulation.
    n : int
        Number of trajectories per group.
    n_components : int
        Dimensionality of each trajectory.

    Returns
    -------
    list of npt.NDArray
        List of simulated trajectories, each of shape (len(t), n_components).
    """
    trajectories = []

    def high_d_trajectory(*args, **kwargs):
        """Create high-dimensional trajectory with random phase jitter."""
        kwargs.update({"phi": np.random.random() * 10.0})
        _traj = np.stack([damped_oscillator(*args, **kwargs)] * n_components, axis=-1)
        assert _traj.shape == (args[0].shape[0], n_components)
        return _traj

    # first some undamped oscillators
    for _ in range(n):
        trajectories += [high_d_trajectory(t, mu=0)]

    # now some damped oscillators
    for _ in range(n):
        trajectories += [high_d_trajectory(t, mu=0.1)]

    # now some low amplitude undamped oscillators
    for _ in range(n):
        trajectories += [high_d_trajectory(t, mu=0, x_0=0.1)]

    return trajectories
