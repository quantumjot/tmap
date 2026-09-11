"""Synthetic trajectory generators.

Two families:

``simulate_trajectories``
    Fixed-length trajectories in three known damping regimes. Every trajectory
    spans the same time grid, so the alignment problem is *balanced*.

``simulate_unbalanced_trajectories``
    Trajectories that enter and leave the observation window at different
    times, so at any instant a different number are present and any two have
    different lengths. This is the regime unbalanced OT exists to handle.

Both return group labels on request (``return_labels=True``) so evaluation can
be supervised; see :mod:`tmap.evaluate`. Both take a ``seed`` and draw from a
local ``Generator`` rather than the global NumPy RNG.
"""
import numpy as np
import numpy.typing as npt

# Group labels used by both generators: index into DAMPING_REGIMES.
DAMPING_REGIMES = (
    {"mu": 0.0, "x_0": 10.0},   # 0: undamped
    {"mu": 0.1, "x_0": 10.0},   # 1: damped
    {"mu": 0.0, "x_0": 0.1},    # 2: low-amplitude undamped
)


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


def _high_d_trajectory(
    t: npt.NDArray,
    *,
    n_components: int,
    rng: np.random.Generator,
    noise: float = 0.0,
    **kwargs,
) -> npt.NDArray:
    """One trajectory with jittered phase, broadcast across components.

    ``noise > 0`` adds independent per-component Gaussian noise. Without it the
    components are exact copies of a single 1-D signal, so the data is rank-1 —
    fine as a smoke-test input, misleading as an evaluation substrate.
    """
    kwargs.setdefault("phi", rng.random() * 10.0)
    signal = damped_oscillator(t, **kwargs)
    traj = np.stack([signal] * n_components, axis=-1)
    if noise:
        traj = traj + rng.standard_normal(traj.shape) * noise
    assert traj.shape == (t.shape[0], n_components)
    return traj


def simulate_trajectories(
    *,
    t: npt.NDArray = np.linspace(0, 10, 100),
    n: int = 10,
    n_components: int = 3,
    noise: float = 0.0,
    seed: int | None = None,
    return_labels: bool = False,
):
    """Simulate ``3 * n`` fixed-length trajectories in three damping regimes.

    Parameters
    ----------
    t : array
        The shared time grid.
    n : int
        Trajectories *per regime*; the total is ``3 * n``.
    n_components : int
        Feature dimensionality.
    noise : float
        Per-component Gaussian noise scale. ``0.0`` (default) reproduces the
        historical rank-1 behaviour.
    seed : int, optional
        Seeds a local generator. ``None`` is nondeterministic but still does not
        touch the global NumPy RNG.
    return_labels : bool
        When ``True`` return ``(trajectories, labels)`` where ``labels[i]`` is
        the index into :data:`DAMPING_REGIMES` for trajectory ``i``. Default
        ``False`` preserves the original return type.

    Returns
    -------
    trajectories : list of arrays
    labels : npt.NDArray
        Only when ``return_labels=True``.
    """
    rng = np.random.default_rng(seed)

    trajectories = []
    labels = []
    for group, params in enumerate(DAMPING_REGIMES):
        for _ in range(n):
            trajectories.append(
                _high_d_trajectory(
                    t, n_components=n_components, rng=rng, noise=noise, **params
                )
            )
            labels.append(group)

    if return_labels:
        return trajectories, np.asarray(labels)
    return trajectories


def simulate_unbalanced_trajectories(
    *,
    t: npt.NDArray = np.linspace(0, 10, 100),
    n: int = 10,
    n_components: int = 3,
    min_fraction: float = 0.3,
    noise: float = 0.0,
    seed: int | None = None,
    return_labels: bool = False,
):
    """Simulate trajectories with staggered birth and death.

    Each trajectory is observed over a contiguous sub-window of ``t`` whose
    length is drawn uniformly from ``[min_fraction, 1.0]`` of the full grid and
    whose start is drawn uniformly from the remaining room. Consequences:

    - trajectories have **different lengths**, so a pair's OT marginals are
      over different supports;
    - the number of trajectories alive at a given time **varies**, so mass is
      created and destroyed across the window.

    The oscillator is evaluated on absolute time, so a trajectory born late
    starts mid-phase rather than restarting — births and deaths are genuine
    entries into and exits from the observation window, not fresh trajectories.

    Parameters
    ----------
    min_fraction : float
        Shortest allowed lifetime as a fraction of the full grid, in ``(0, 1]``.

    Returns
    -------
    trajectories : list of arrays
        Variable length ``(n_i, n_components)``.
    labels : npt.NDArray
        Regime index per trajectory. Only when ``return_labels=True``.
    lifetimes : npt.NDArray
        ``(3 * n, 2)`` array of ``[start, stop)`` indices into ``t``. Only when
        ``return_labels=True``.
    """
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction}")

    rng = np.random.default_rng(seed)
    n_t = t.shape[0]
    min_length = max(2, int(round(min_fraction * n_t)))

    trajectories = []
    labels = []
    lifetimes = []
    for group, params in enumerate(DAMPING_REGIMES):
        for _ in range(n):
            length = int(rng.integers(min_length, n_t + 1))
            start = int(rng.integers(0, n_t - length + 1))
            window = t[start : start + length]
            trajectories.append(
                _high_d_trajectory(
                    window, n_components=n_components, rng=rng, noise=noise, **params
                )
            )
            labels.append(group)
            lifetimes.append((start, start + length))

    if return_labels:
        return trajectories, np.asarray(labels), np.asarray(lifetimes)
    return trajectories


def add_observation_noise(
    trajectories: list[npt.NDArray],
    *,
    scale: float,
    seed: int | None = None,
) -> list[npt.NDArray]:
    """Add i.i.d. Gaussian noise to every trajectory.

    For robustness evaluation: score a method on clean and noisy versions of
    the same trajectories and report the degradation.
    """
    rng = np.random.default_rng(seed)
    return [x + rng.standard_normal(x.shape) * scale for x in trajectories]


def mis_warp(
    trajectory: npt.NDArray,
    *,
    index: int,
    repeat: int = 3,
) -> npt.NDArray:
    """Stall a trajectory at one timepoint, then resume.

    Holds ``trajectory[index]`` for ``repeat`` frames, then resumes at
    ``index + repeat``, dropping the intervening frames. Length is preserved and
    the tail stays time-aligned, so the perturbation is *local*: an aligner must
    absorb a stall-and-skip rather than a global offset. This is the single
    mis-warped step that hard DTW handles badly and soft-DTW is claimed to
    absorb.
    """
    n = trajectory.shape[0]
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for length {n}")
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    held = np.repeat(trajectory[index : index + 1], repeat, axis=0)
    out = np.concatenate([trajectory[:index], held, trajectory[index + repeat :]], axis=0)
    return out[:n]
