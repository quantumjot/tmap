from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import umap

from scipy import optimize
from tqdm import tqdm

from tmap import base
from tmap.alignment import DTWAlignment, OTAlignment
from tmap.layout import InitialLayout


# set a default random seed
np.random.seed(123)


def intra_sequence_feature_dist(seq: npt.NDArray) -> list[float]:
    """Calculate the intra sequence feature distance.
    
    Assumes that the inter-node distance should be based on the equivalent 
    feature distance between the nodes. Calculated based on the distance 
    function used for DTW to compare sequences. The final node has an infinte
    distance to prevent linking between sequences.
    
    """
    norm_delta_sq = np.linalg.norm(np.diff(seq, axis=0), 2, axis=1) ** 2
    dist = norm_delta_sq.tolist() + [np.inf]
    assert len(dist) == seq.shape[0]
    return dist


def _align_pair_worker(args):
    """Module-level worker so pairs can be dispatched to a process pool."""
    aligner, i, j, seq_i, seq_j, mask = args
    return i, j, np.asarray(aligner(seq_i, seq_j, mask=mask), dtype=np.float32)


def calculate_distance_matrix(
    sequences: List[npt.NDArray],
    aligner: base.AlignmentBase,
    *,
    mask: bool = True,
    n_jobs: Optional[int] = None,
) -> npt.NDArray:
    """Calculate the distance matrix.

    Parameters
    ----------
    sequences : list
        The sequences to align and calculate the distance matrix.
    aligner : AlignmentBase
        An alignment method to construct the sparse distance graph.
    mask : bool
        Whether to mask the transport plan to the optimal path.
    n_jobs : int, optional
        Number of workers used to compute the (independent) pairwise
        alignments. ``None`` or ``1`` runs serially (default); ``-1`` uses all
        available cores. When the aligner releases the GIL (DTW with the C
        extension available) threads are used; otherwise a process pool is used
        so pure-Python alignment still parallelises.

    Returns
    -------
    distance_matrix : array
        An array representing the distance matrix between all pairs of sequences in the input.

    Notes
    -----
    This could be a sparse matrix in practice.
    """

    seq_shapes = [s.shape[0] for s in sequences]
    n = sum(seq_shapes)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    pairs = [(i, j) for i in range(len(sequences)) for j in range(i + 1, len(sequences))]
    desc = aligner.name + f" (masking={mask})"

    if n_jobs in (None, 1):
        results = [
            (i, j, aligner(sequences[i], sequences[j], mask=mask).astype(np.float32))
            for i, j in tqdm(pairs, desc=desc)
        ]
    else:
        max_workers = None if n_jobs == -1 else n_jobs
        # threads when the aligner releases the GIL (compiled DTW), else processes
        use_threads = getattr(aligner, "releases_gil", False)
        if use_threads:
            def _align(pair):
                i, j = pair
                return i, j, aligner(sequences[i], sequences[j], mask=mask).astype(np.float32)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_align, pairs), total=len(pairs), desc=desc))
        else:
            work = [(aligner, i, j, sequences[i], sequences[j], mask) for i, j in pairs]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    tqdm(executor.map(_align_pair_worker, work), total=len(pairs), desc=desc)
                )

    for i, j, alignment_output in results:
        sx = slice(sum(seq_shapes[:i]), sum(seq_shapes[: i + 1]), 1)
        sy = slice(sum(seq_shapes[:j]), sum(seq_shapes[: j + 1]), 1)

        if isinstance(aligner, OTAlignment):
            # for OT, output is a weighted transport plan (correspondence)
            distances = 1.0 / (alignment_output + 1e-9)
            distance_matrix[sx, sy] = distances
        elif isinstance(aligner, DTWAlignment):
            # for DTW, the output is the cost matrix, which is already a
            # distance matrix (low cost = low distance), so we can use it directly
            distance_matrix[sx, sy] = alignment_output
        else:
            distance_matrix[sx, sy] = alignment_output

    # consider the local connectivity of the trajectory too
    local = [intra_sequence_feature_dist(seq) for seq in sequences]
    distance_matrix[np.eye(n, k=1).astype(bool)] = np.concatenate(local)[:-1]

    # now make the matrix symmetric
    distance_matrix = distance_matrix + distance_matrix.T
    distance_matrix[distance_matrix == 0] = base.EPSILON_WEIGHT
    distance_matrix[np.eye(n).astype(bool)] = 0.0

    return distance_matrix


def high_dimensional_probability(d: npt.NDArray, sigma: float) -> npt.NDArray:
    d = np.clip(d, 0.0, np.inf)  # clamp to greater than zero
    # d = np.abs(d)
    assert sigma > 0.0
    return np.exp(-d / sigma)


def estimate_sigma(
    d: npt.NDArray, n_neighbors: int, iterations: int = 20, tolerance: float = 1e-5
):
    """Binary search to estimate a value of sigma.

    Parameters
    ----------

    Returns
    -------
    """

    def k_of_sigma(sigma):
        prob = high_dimensional_probability(d, sigma)
        return np.power(2, np.clip(np.sum(prob), 0, 31.0))

    sigma_lower_estimate = base.SIGMA_LOW_ESTIMATE
    sigma_upper_estimate = base.SIGMA_HIGH_ESTIMATE

    for _ in range(iterations):
        sigma_estimate = (sigma_lower_estimate + sigma_upper_estimate) / 2
        k_sigma_estimate = k_of_sigma(sigma_estimate)

        if k_sigma_estimate < n_neighbors:
            sigma_lower_estimate = sigma_estimate
        else:
            sigma_upper_estimate = sigma_estimate

        if np.abs(n_neighbors - k_sigma_estimate) <= tolerance:
            break

    return sigma_estimate


def estimate_sigma_vectorized(
    d: npt.NDArray,
    n_neighbors: int,
    iterations: int = 20,
) -> npt.NDArray:
    """Binary search for per-row sigma, for all rows at once.

    Vectorised equivalent of :func:`estimate_sigma` applied to every row of the
    (row-shifted, non-negative-clipped) distance matrix ``d``. Runs the same
    bisection on ``[SIGMA_LOW_ESTIMATE, SIGMA_HIGH_ESTIMATE]`` for all rows
    simultaneously, replacing the Python row loop with array operations.

    Parameters
    ----------
    d : npt.NDArray, shape (N, N)
        Distances with each row's ``rho`` already subtracted (not yet clipped).
    n_neighbors : int
        Target effective number of neighbours per row.

    Returns
    -------
    sigma : npt.NDArray, shape (N,)
        Estimated sigma for each row.
    """
    n = d.shape[0]
    d_clipped = np.clip(d, 0.0, np.inf)
    lower = np.full(n, base.SIGMA_LOW_ESTIMATE, dtype=np.float64)
    upper = np.full(n, base.SIGMA_HIGH_ESTIMATE, dtype=np.float64)
    sigma = (lower + upper) / 2

    for _ in range(iterations):
        sigma = (lower + upper) / 2
        prob = np.exp(-d_clipped / sigma[:, None])
        k = np.power(2.0, np.clip(prob.sum(axis=1), 0, 31.0))
        below = k < n_neighbors
        lower = np.where(below, sigma, lower)
        upper = np.where(below, upper, sigma)

    return sigma


def calculate_high_dimensional_probability_matrix(
    dist: npt.NDArray,
    n_neighbors: int,
) -> npt.NDArray:
    """Calculate the high dimensional probability matrix from the adjacency
    matrix representation of the graph.

    Parameters
    ----------
    dist : npt.NDArray
    n_neighbors : int

    Returns
    -------
    prob : npt.NDArray
    """

    # per-row nearest-neighbour distance (rho). Each row has exactly one zero
    # (the diagonal), so the second-smallest entry is the nearest neighbour.
    # np.partition avoids the O(N log N) full sort of the old comprehension.
    rho = np.partition(dist, 1, axis=1)[:, 1]

    # shift each row by its rho, estimate all sigmas at once, then form the
    # probabilities in a single vectorised exp instead of a Python row loop.
    d = dist - rho[:, None]
    sigma = estimate_sigma_vectorized(d, n_neighbors)
    prob = np.exp(-np.clip(d, 0.0, np.inf) / sigma[:, None])

    # make the distances compatible by enforcing symmetry
    prob = symmetrize_probability_matrix_umap(prob)
    return prob


def symmetrize_probability_matrix_tsne(prob: npt.NDArray) -> npt.NDArray:
    return prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))


def symmetrize_probability_matrix_umap(prob: npt.NDArray) -> npt.NDArray:
    return (prob + np.transpose(prob)) / 2


def find_hyperparameters(min_dist: float):
    """
    Parameters
    ----------

    Returns
    -------
    a : float
    b : float
    """
    x = np.linspace(0.0, 3.0, 300, dtype=np.float64)

    def f(x, min_dist):
        y = np.exp(-x + min_dist)
        y[x <= min_dist] = 1.0
        return y

    dist_low_dim = lambda x, a, b: 1 / (1 + a * x ** (2 * b))
    p, _ = optimize.curve_fit(dist_low_dim, x, f(x, min_dist))
    a, b = p
    return a, b


@jax.jit
def jax_euclidean_distances(i, j):
    """Calculate the Euclidean distances between two arrays.

    Parameters
    ----------

    Returns
    -------
    """
    M = i.shape[0]
    N = j.shape[0]
    I_dots = jnp.reshape(jnp.sum((i * i), axis=1), (M, 1)) * jnp.ones(shape=(1, N))
    J_dots = jnp.sum((j * j), axis=1) * jnp.ones(shape=(M, 1))
    D_squared = I_dots + J_dots - 2 * jnp.dot(i, j.T)
    return jnp.maximum(0.0, D_squared)


@jax.jit
def jax_cross_entropy_gradient_2(p, y, a, b, *, eps: float = 1e-8):
    """Calculate the BCE gradient.

    Parameters
    ----------

    Returns
    -------
    """
    n = y.shape[0]
    d = jax_euclidean_distances(y, y)
    w = jnp.power(1.0 + a * (d + eps) ** b, -1)
    w = w * (1 - jnp.identity(n))
    w_sum = jnp.sum(w, axis=1, keepdims=True)
    q = w / (w_sum + eps)
    q = jnp.clip(q, eps, 1.0 - eps)
    loss = -jnp.sum(p * jnp.log(q) + (1 - p) * jnp.log(1 - q))
    return loss


@partial(jax.jit, static_argnames=("n_steps",))
def _sgd_chunk(p, y, a, b, learning_rate, n_steps):
    """Run ``n_steps`` full-batch SGD updates entirely on-device.

    Keeping the inner loop inside a single ``jax.lax.fori_loop`` avoids a
    host<->device round-trip and a Python dispatch on every iteration (the old
    loop synced ``loss`` back to the host each step to update the progress
    bar, which serialised the GPU). The update rule is identical to the
    previous ``y = y - learning_rate * grad`` so results are unchanged.
    """
    grad_fn = jax.value_and_grad(jax_cross_entropy_gradient_2, argnums=1)

    def body(_, carry):
        y, _last_loss = carry
        loss, grad = grad_fn(p, y, a, b)
        return (y - learning_rate * grad, loss)

    y, last_loss = jax.lax.fori_loop(0, n_steps, body, (y, jnp.inf))
    return y, last_loss


def optimize_embedding(
    p: npt.NDArray,
    y: npt.NDArray,
    a: float,
    b: float,
    *,
    learning_rate: float,
    n_iterations: int,
    chunk: int = 20,
    progress: bool = True,
) -> npt.NDArray:
    """Optimise the low-dimensional embedding with on-device SGD.

    The iterations run on-device in blocks of ``chunk`` steps; the loss is only
    pulled back to the host once per block for the progress bar, instead of
    every iteration. This is the single code path used by both ``fit`` and the
    benchmark harness.
    """
    p = jnp.asarray(p)
    y = jnp.asarray(y)
    lr = jnp.asarray(learning_rate, dtype=y.dtype)

    remaining = n_iterations
    loss = np.inf
    pbar = tqdm(total=n_iterations, disable=not progress)
    while remaining > 0:
        steps = min(chunk, remaining)
        y, loss = _sgd_chunk(p, y, a, b, lr, steps)
        remaining -= steps
        if progress:
            pbar.update(steps)
            pbar.set_description(f"Embedding (loss: {float(loss):.3f})")
    pbar.close()

    return np.asarray(y)


class TemporalMAP(base.MapperBase):
    """TemporalMAP.

    Creates a low dimensional embedding of the input data that attempts to
    respect the temporal ordering of the data.

    Parameters
    ----------
    n_neighbors : int
        The maximum number of neighbors in the graph.
    min_dist : int
        The minimum distance in the low dimensional representation.
    n_components : int (default = 2)
        The number of dimensions of the low dimensional representation.
    window : int, None
        The window used by the dynamic time warping function. Balances
        local temporal features vs global trajectory warping.
    layout : str, InitialLayout 
        The initial layout methods, defaults to spectral embedding.
    aligner : AlignmentBase 
        An alignment method to construct the sparse distance graph.
    mask : bool
        Whether to mask the transport plan to the optimal path.

    Attributes
    ----------
    trajectories : list
    sequence_shapes : list

    """

    def __init__(
        self,
        *,
        n_neighbors: int = base.N_NEIGHBORS,
        min_dist: float = base.MIN_DIST,
        n_components: int = base.N_COMPONENTS,
        window: Optional[int] = None,
        layout: InitialLayout | base.LayoutBase | str = "SPECTRAL",
        aligner: Optional[base.AlignmentBase] = None,
        mask: bool = True,
        n_jobs: Optional[int] = None,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.window = window
        self.mask = mask
        self.n_jobs = n_jobs

        self._sequences = []
        self._P = None
        self._embedding = None
        self._distance_matrix = None


        # set the initial layout method
        self._set_initial_layout_method(layout)

        # default to DTW for alignment
        self.aligner = DTWAlignment(window=window) if aligner is None else aligner

    def _set_initial_layout_method(self, layout: InitialLayout | base.LayoutBase | str) -> None:
        """Set the initial layout method"""
        if isinstance(layout, str):
            self._layout = InitialLayout[layout.upper()]
        elif isinstance(layout, (InitialLayout, base.LayoutBase)):
            self._layout = layout
        else:
            raise TypeError(f"Layout method not recognized: {layout}")
        
    def _initialize(self, sequences: List[npt.NDArray]) -> tuple[npt.NDArray, float, float]:
        """Initialize the parameters of the fit"""
        for seq in sequences:
            if not isinstance(seq, np.ndarray):
                raise TypeError("Trajectories should be numpy arrays")
            if seq.shape[-1] < self.n_components:
                raise ValueError("Trajectories should be high dimensional")

        self._sequences = sequences

        if self.distance_matrix is None:
            _ = self.calculate_distance_matrix(sequences)

        prob = calculate_high_dimensional_probability_matrix(
            self.distance_matrix, self.n_neighbors
        )

        self._P = prob

        a, b = find_hyperparameters(self.min_dist)
        return prob, a, b


    def calculate_distance_matrix(
        self,
        sequences: List[npt.NDArray],
    ) -> npt.NDArray:
        """Calculate the distance matrix."""
        self._distance_matrix = calculate_distance_matrix(
            sequences, self.aligner, mask=self.mask, n_jobs=self.n_jobs
        )
        return self._distance_matrix

    def fit(
        self,
        sequences: List[npt.NDArray],
        learning_rate: float = base.LEARNING_RATE,
        max_iterations: int = base.MAX_ITERATIONS,
    ) -> npt.NDArray:
        """
        Parameters
        ----------
        sequences : list of arrays
            These should be high dimensional arrays that represent the trajectories.
            [(n_i, m), (n_j, m), ..., (n_k, m)] where n_i, is the length of
            trajectory i and m is the number of features for each timepoint.
        learning_rate : float
            The learning rate for the optimization.
        max_iterations : int
            The maximum number of interations for the optimization.

        Returns
        -------
        y : npt.NDArray (N, n_components)
            The embedding in `n_components` dimensions.
        """

        prob, a, b = self._initialize(sequences)

        y = self._layout(sequences, n_components=self.n_components)
        y = optimize_embedding(
            prob, y, a, b, learning_rate=learning_rate, n_iterations=max_iterations
        )

        self._embedding = y
        self._P = prob

        return y


class DefaultUMAP(base.MapperBase):
    """Simple wrapper around UMAP to provide comparison with TMAP"""

    def __init__(
        self, 
        *, 
        min_dist: int = base.MIN_DIST, 
        n_neighbors: int = 1, 
        n_components: int = base.N_COMPONENTS
    ):
        self._umap = umap.UMAP()
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.window = None

    def fit(self, sequences):

        self._sequences = sequences
        x = np.concatenate(self._sequences, axis=0)

        y = self._umap.fit_transform(
            x,
            min_dist=self.min_dist,
            n_neighbors=self.n_neighbors,
        )

        self._embedding = y
        return y
