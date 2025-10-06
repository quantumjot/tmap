"""Temporal UMAP implementation for temporal sequence embeddings.

This module implements TemporalMAP, a variant of UMAP designed specifically for
temporal sequence data. It includes both parallel and sequential implementations
for distance matrix computation, and JAX-optimized functions for probability
matrix calculations.
"""
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import numpy.typing as npt
import umap
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from scipy import optimize
from tqdm import tqdm

from tmap import base
from tmap.alignment import DTWAlignment, OTAlignment
from tmap.layout import InitialLayout


def intra_sequence_feature_dist(seq: npt.NDArray) -> list[float]:
    """Calculate the intra sequence feature distance.

    Computes pairwise Euclidean distances between consecutive timepoints in a
    sequence. The final node has an infinite distance to prevent linking
    between different sequences in the concatenated representation.

    Parameters
    ----------
    seq : npt.NDArray
        Input sequence of shape (n, features).

    Returns
    -------
    list of float
        List of n squared L2 distances between consecutive points, with the
        last element set to infinity.
    """
    norm_delta_sq = np.linalg.norm(np.diff(seq, axis=0), 2, axis=1) ** 2
    dist = norm_delta_sq.tolist() + [np.inf]
    assert len(dist) == seq.shape[0]
    return dist


def _compute_sequence_pair_alignment(args):
    """Helper function for alignment computation.

    Computes alignment between a single pair of sequences and returns the
    appropriate slice indices and distance matrix.

    Parameters
    ----------
    args : tuple
        Tuple containing (i, j, seq_i, seq_j, aligner, mask, seq_shapes) where:
        - i, j: sequence indices
        - seq_i, seq_j: sequence arrays
        - aligner: alignment method instance
        - mask: whether to mask to optimal path
        - seq_shapes: list of sequence lengths

    Returns
    -------
    tuple
        (sx, sy, distances) where sx and sy are slice objects for indexing
        the full distance matrix, and distances is the pairwise alignment matrix.
    """
    i, j, seq_i, seq_j, aligner, mask, seq_shapes = args

    alignment_output = aligner(seq_i, seq_j, mask=mask).astype(np.float32)

    sx = slice(sum(seq_shapes[:i]), sum(seq_shapes[: i + 1]), 1)
    sy = slice(sum(seq_shapes[:j]), sum(seq_shapes[: j + 1]), 1)

    if isinstance(aligner, OTAlignment):
        # for OT, output is a weighted transport plan (correspondence in [0, 1])
        # Convert to distance: use negative log transform for numerical stability
        # Higher correspondence -> lower distance
        distances = -np.log(alignment_output + 1e-10)
        return (sx, sy, distances)
    elif isinstance(aligner, DTWAlignment):
        # for DTW, the output is the cost matrix, which is already a
        # distance matrix (low cost = low distance), so we can use it directly
        return (sx, sy, alignment_output)
    else:
        return (sx, sy, alignment_output)


def calculate_distance_matrix(
    sequences: List[npt.NDArray],
    aligner: base.AlignmentBase,
    *,
    mask: bool = True,
    max_workers: Optional[int] = None,
    use_parallel: bool = True
) -> npt.NDArray:
    """Calculate the distance matrix with automatic parallel/sequential processing.

    Parameters
    ----------
    sequences : list
        The sequences to align and calculate the distance matrix.
    aligner : AlignmentBase
        An alignment method to construct the sparse distance graph.
    mask : bool
        Whether to mask the transport plan to the optimal path.
    max_workers : int, optional
        Maximum number of worker threads. If None, uses cpu_count().
    use_parallel : bool
        Whether to attempt parallel processing. Automatically falls back to
        sequential for small workloads.

    Returns
    -------
    distance_matrix : array
        An array representing the distance matrix between all pairs of sequences.
    """
    seq_shapes = [s.shape[0] for s in sequences]
    n = sum(seq_shapes)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    # Prepare all sequence pairs
    tasks = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            tasks.append((i, j, sequences[i], sequences[j], aligner, mask, seq_shapes))

    # Decide whether to use parallel processing
    if max_workers is None:
        max_workers = min(cpu_count(), len(sequences))

    use_parallel = use_parallel and len(tasks) > 1 and max_workers > 1

    if use_parallel:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_sequence_pair_alignment, task): task for task in tasks}

            # Collect all results first to avoid race condition
            results = []
            with tqdm(total=len(tasks), desc=f"{aligner.name} (masking={mask})") as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

            # Write to distance matrix sequentially
            for sx, sy, alignment_result in results:
                distance_matrix[sx, sy] = alignment_result
    else:
        # Sequential processing
        for task in tqdm(tasks, desc=f"{aligner.name} (masking={mask})"):
            sx, sy, alignment_result = _compute_sequence_pair_alignment(task)
            distance_matrix[sx, sy] = alignment_result

    # Add local connectivity of trajectories
    local = [intra_sequence_feature_dist(seq) for seq in sequences]
    distance_matrix[np.eye(n, k=1).astype(bool)] = np.concatenate(local)[:-1]

    # Make the matrix symmetric by averaging with transpose
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    distance_matrix[distance_matrix == 0] = base.EPSILON_WEIGHT
    distance_matrix[np.eye(n).astype(bool)] = 0.0

    return distance_matrix


def high_dimensional_probability(d: npt.NDArray, sigma: float) -> npt.NDArray:
    """Compute high-dimensional probability from distances using Gaussian kernel.

    Parameters
    ----------
    d : npt.NDArray
        Distance values.
    sigma : float
        Bandwidth parameter for the Gaussian kernel.

    Returns
    -------
    npt.NDArray
        Probability values computed as exp(-d / sigma).
    """
    d = np.clip(d, 0.0, np.inf)  # clamp to greater than zero
    assert sigma > 0.0
    return np.exp(-d / sigma)


def estimate_sigma_vectorized(
    dist: npt.NDArray,
    rho: npt.NDArray,
    n_neighbors: int,
    iterations: int = 20,
    tolerance: float = 1e-5
) -> npt.NDArray:
    """Vectorized binary search to estimate sigma for all rows simultaneously.

    Performs binary search for all rows at once using vectorized operations,
    providing significant speedup over row-by-row processing.

    Parameters
    ----------
    dist : npt.NDArray
        Distance matrix of shape (n, n).
    rho : npt.NDArray
        Minimum non-zero distance for each row, shape (n,).
    n_neighbors : int
        Target number of effective neighbors.
    iterations : int
        Maximum number of binary search iterations.
    tolerance : float
        Convergence tolerance for the search.

    Returns
    -------
    npt.NDArray
        Array of sigma values for each row, shape (n,).
    """
    n = dist.shape[0]

    # Initialize sigma bounds for all rows
    sigma_low = np.full(n, base.SIGMA_LOW_ESTIMATE, dtype=np.float32)
    sigma_high = np.full(n, base.SIGMA_HIGH_ESTIMATE, dtype=np.float32)

    # Adjust distances by rho
    d_adj = dist - rho[:, None]
    d_adj = np.clip(d_adj, 0.0, np.inf)

    # Vectorized binary search
    for _ in range(iterations):
        sigma_mid = (sigma_low + sigma_high) / 2

        # Compute effective neighbors for all rows simultaneously
        # prob[i,j] = exp(-d_adj[i,j] / sigma_mid[i])
        prob = np.exp(-d_adj / sigma_mid[:, None])
        k_values = np.power(2, np.clip(np.sum(prob, axis=1), 0, 31.0))

        # Update bounds for all rows
        too_small = k_values < n_neighbors
        sigma_low = np.where(too_small, sigma_mid, sigma_low)
        sigma_high = np.where(~too_small, sigma_mid, sigma_high)

        # Check global convergence (can exit early if all converged)
        if np.all(np.abs(k_values - n_neighbors) <= tolerance):
            break

    return sigma_mid


def calculate_high_dimensional_probability_matrix(
    dist: npt.NDArray,
    n_neighbors: int,
) -> npt.NDArray:
    """Vectorized calculation of high dimensional probability matrix.

    Uses vectorized sigma estimation for significant speedup compared to
    row-by-row processing.

    Parameters
    ----------
    dist : npt.NDArray
        Distance matrix of shape (n, n).
    n_neighbors : int
        Target number of effective neighbors for adaptive bandwidth selection.

    Returns
    -------
    npt.NDArray
        Symmetrized probability matrix of shape (n, n).
    """
    # Calculate the minimum (non-zero) distance for each row
    rho = np.array([sorted(dist[i])[1] for i in range(dist.shape[0])], dtype=np.float32)

    # Vectorized sigma estimation for all rows
    sigmas = estimate_sigma_vectorized(dist, rho, n_neighbors)

    # Compute probabilities using vectorized sigmas
    d_adj = dist - rho[:, None]
    d_adj = np.clip(d_adj, 0.0, np.inf)
    prob = np.exp(-d_adj / sigmas[:, None])

    # Make the distances compatible by enforcing symmetry
    prob = (prob + prob.T) / 2
    return prob.astype(np.float32)


def find_hyperparameters(min_dist: float):
    """Find UMAP curve parameters a and b for low-dimensional representation.

    Fits the UMAP low-dimensional similarity curve to match the target
    distribution based on min_dist parameter.

    Parameters
    ----------
    min_dist : float
        Minimum distance parameter controlling how tightly points can be
        packed in the low-dimensional space.

    Returns
    -------
    a : float
        UMAP curve parameter a.
    b : float
        UMAP curve parameter b (exponent).
    """
    x = np.linspace(0.0, 3.0, 300, dtype=np.float32)

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
    """Calculate pairwise squared Euclidean distances between two arrays.

    Efficiently computes ||i - j||^2 for all pairs using the expansion:
    ||i - j||^2 = ||i||^2 + ||j||^2 - 2*i·j

    Parameters
    ----------
    i : jnp.ndarray
        First array of shape (M, d).
    j : jnp.ndarray
        Second array of shape (N, d).

    Returns
    -------
    jnp.ndarray
        Squared Euclidean distance matrix of shape (M, N), with negative
        values clipped to 0.
    """
    M = i.shape[0]
    N = j.shape[0]
    I_dots = jnp.reshape(jnp.sum((i * i), axis=1), (M, 1)) * jnp.ones(shape=(1, N))
    J_dots = jnp.sum((j * j), axis=1) * jnp.ones(shape=(M, 1))
    D_squared = I_dots + J_dots - 2 * jnp.dot(i, j.T)
    return jnp.maximum(0.0, D_squared)


@jax.jit
def jax_cross_entropy_gradient_2(p, y, a, b, *, eps: float = 1e-8):
    """Calculate binary cross-entropy loss for UMAP optimization.

    Computes the loss between high-dimensional probabilities (p) and
    low-dimensional probabilities (q) derived from current embedding (y).

    Parameters
    ----------
    p : jnp.ndarray
        High-dimensional probability matrix of shape (n, n).
    y : jnp.ndarray
        Current low-dimensional embedding of shape (n, n_components).
    a : float
        UMAP curve parameter a.
    b : float
        UMAP curve parameter b.
    eps : float
        Small epsilon for numerical stability.

    Returns
    -------
    float
        Cross-entropy loss value.
    """
    n = y.shape[0]
    d_squared = jax_euclidean_distances(y, y)
    # Take square root to get actual distances, as hyperparameter fitting expects d^(2b)
    d = jnp.sqrt(d_squared + eps)
    # UMAP low-dimensional similarity: 1 / (1 + a * d^(2b))
    w = jnp.power(1.0 + a * jnp.power(d, 2 * b), -1)
    w = w * (1 - jnp.identity(n))
    w_sum = jnp.sum(w, axis=1, keepdims=True)
    q = w / (w_sum + eps)
    q = jnp.clip(q, eps, 1.0 - eps)
    # Use proper UMAP cross-entropy: -sum(p * log(q))
    # The previous binary cross-entropy formula -sum(p * log(q) + (1-p) * log(1-q))
    # is incorrect since p can be > 1 (not normalized), making (1-p) negative
    loss = -jnp.sum(p * jnp.log(q + eps))
    return loss


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
    use_parallel_distance_matrix : bool
        Whether to use parallel processing for distance matrix computation.
    max_workers : int, optional
        Maximum number of worker threads for parallel processing.
    random_state : int, np.random.RandomState, or None
        Random state for reproducibility. If int, used as seed. If None, uses default random state.

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
        use_parallel_distance_matrix: bool = True,
        max_workers: Optional[int] = None,
        random_state: Optional[int | np.random.RandomState] = None,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.window = window
        self.mask = mask
        self.use_parallel_distance_matrix = use_parallel_distance_matrix
        self.max_workers = max_workers

        # Initialize random state
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState()

        self._sequences = []
        self._P = None
        self._embedding = None
        self._distance_matrix = None


        # set the initial layout method
        self._set_initial_layout_method(layout)

        # default to DTW for alignment
        self.aligner = DTWAlignment(window=window) if aligner is None else aligner

    def _set_initial_layout_method(self, layout: InitialLayout | base.LayoutBase | str) -> None:
        """Set the initial layout method.

        Parameters
        ----------
        layout : InitialLayout, LayoutBase, or str
            Layout method specification.

        Raises
        ------
        TypeError
            If layout type is not recognized.
        """
        from tmap.layout import RandomLayout

        if isinstance(layout, str):
            layout_enum = InitialLayout[layout.upper()]
            # Instantiate the layout class
            if layout_enum == InitialLayout.RANDOM:
                self._layout = layout_enum.value(random_state=self.random_state)
            else:
                self._layout = layout_enum.value()
        elif isinstance(layout, InitialLayout):
            # If it's the enum, instantiate the class
            if layout == InitialLayout.RANDOM:
                self._layout = layout.value(random_state=self.random_state)
            else:
                self._layout = layout.value()
        elif isinstance(layout, base.LayoutBase):
            # Already an instance
            self._layout = layout
        else:
            raise TypeError(f"Layout method not recognized: {layout}")

    def _initialize(self, sequences: List[npt.NDArray]) -> tuple[npt.NDArray, float, float]:
        """Initialize the parameters of the fit.

        Computes distance matrix and probability matrix, and finds hyperparameters.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.

        Returns
        -------
        prob : npt.NDArray
            High-dimensional probability matrix.
        a : float
            UMAP hyperparameter a.
        b : float
            UMAP hyperparameter b.

        Raises
        ------
        TypeError
            If sequences are not numpy arrays.
        ValueError
            If sequences have fewer dimensions than n_components.
        """
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
        """Calculate the distance matrix.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.

        Returns
        -------
        npt.NDArray
            Distance matrix between all sequence pairs.
        """
        self._distance_matrix = calculate_distance_matrix(
            sequences,
            self.aligner,
            mask=self.mask,
            max_workers=self.max_workers,
            use_parallel=self.use_parallel_distance_matrix
        )
        return self._distance_matrix

    def fit(
        self,
        sequences: List[npt.NDArray],
        learning_rate: float = base.LEARNING_RATE,
        max_iterations: int = base.MAX_ITERATIONS,
    ) -> npt.NDArray:
        """Fit TemporalMAP to sequences and return low-dimensional embeddings.

        Computes pairwise sequence alignments, constructs a high-dimensional
        probability matrix, initializes the layout, and optimizes the embedding
        using gradient descent to minimize cross-entropy loss.

        Parameters
        ----------
        sequences : list of arrays
            High-dimensional arrays representing trajectories. Each array has
            shape (n_i, m) where n_i is the length of trajectory i and m is
            the number of features for each timepoint.
        learning_rate : float
            The learning rate for gradient descent optimization.
        max_iterations : int
            The maximum number of iterations for the optimization.

        Returns
        -------
        y : npt.NDArray (N, n_components)
            The embedding in `n_components` dimensions, where N is the total
            number of timepoints across all sequences.
        """

        prob, a, b = self._initialize(sequences)

        y = self._layout(sequences, n_components=self.n_components)
        grad_fn = jax.value_and_grad(jax_cross_entropy_gradient_2, argnums=1)
        loss = np.inf

        embed_pbar = tqdm(range(max_iterations))
        for _ in embed_pbar:
            embed_pbar.set_description(f"Embedding (loss: {loss:.3f})")
            loss, grad = grad_fn(prob, y, a, b)
            y = y - learning_rate * grad

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
        """Fit standard UMAP to sequences.

        Parameters
        ----------
        sequences : list of npt.NDArray
            Input sequences.

        Returns
        -------
        npt.NDArray
            UMAP embeddings.
        """
        self._sequences = sequences
        x = np.concatenate(self._sequences, axis=0)

        y = self._umap.fit_transform(
            x,
            min_dist=self.min_dist,
            n_neighbors=self.n_neighbors,
        )

        self._embedding = y
        return y
