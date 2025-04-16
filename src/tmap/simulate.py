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
    trajectories = []

    def high_d_trajectory(*args, **kwargs):
        # make a high dimensional trajectory with jittered phi
        kwargs.update({"phi": np.random.random() * 10.0})
        _traj = np.stack([damped_oscillator(*args, **kwargs)] * n_components, axis=-1)
        assert _traj.shape == (args[0].shape[0], n_components)
        return _traj

    # first some undamped oscillators
    for _ in range(10):
        trajectories += [high_d_trajectory(t, mu=0)]

    # now some damped oscillators
    for _ in range(10):
        trajectories += [high_d_trajectory(t, mu=0.1)]

    # now some low amplitude undamped oscillators
    for _ in range(10):
        trajectories += [high_d_trajectory(t, mu=0, x_0=0.1)]

    return trajectories
