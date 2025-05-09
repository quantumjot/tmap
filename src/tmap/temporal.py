from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import umap

from scipy import optimize
from tqdm import tqdm

from tmap import base, layout
from tmap.alignment import DTWAlignment, OTAlignment


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


def calculate_distance_matrix(
    sequences: List[npt.NDArray], aligner: base.AlignmentBase,
) -> npt.NDArray:
    """Calculate the distance matrix.

    Parameters
    ----------
    sequences : 
    aligner :


    Returns
    -------
    distance_matrix :
        An array representing the distance matrix between all pairs of sequences in the input.

    Notes
    -----
    This could be a sparse matrix in practice.
    """

    seq_shapes = [s.shape[0] for s in sequences]
    n = sum(seq_shapes)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for i in tqdm(range(len(sequences)), desc=aligner.name):
        for j in range(i + 1, len(sequences)):

            seq_i, seq_j = sequences[i], sequences[j]
            mask = aligner(seq_i, seq_j)
            
            sx = slice(sum(seq_shapes[:i]), sum(seq_shapes[: i + 1]), 1)
            sy = slice(sum(seq_shapes[:j]), sum(seq_shapes[: j + 1]), 1)
            distance_matrix[sx, sy] = mask

    # TODO(arl): should consider the local connectivity of the trajectory too
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

    # calculate the minimum (non-zero) distance for each row
    rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
    prob = np.zeros_like(dist, dtype=np.float64)

    for row in range(prob.shape[0]):
        d = dist[row, ...] - rho[row]
        sigma = estimate_sigma(d, n_neighbors)
        prob[row, ...] = high_dimensional_probability(d, sigma)

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
    pre_embedding_fn : Callable
        A function to initialize the embedding in low dimensional space
    distance_fn : Callable
        A function to calculate the distances in sequence space.

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
        layout: layout.InitialLayout = layout.InitialLayout.SPECTRAL,
        aligner: Optional[base.AlignmentBase] = None
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.window = window

        self._sequences = []
        self._P = None
        self._embedding = None
        self._distance_matrix = None
        self._layout = layout

        # Default to DTW
        self.aligner = DTWAlignment(window=window) if aligner is None else aligner


    def calculate_distance_matrix(
        self,
        sequences: List[npt.NDArray],
    ) -> npt.NDArray:
        """Calculate the distance matrix."""
        self._distance_matrix = calculate_distance_matrix(sequences, self.aligner)
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

        a, b = find_hyperparameters(self.min_dist)

        np.random.seed(123)
        x = np.concatenate(sequences, axis=0)
        y = self._layout(x, n_components=self.n_components)
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

    def __init__(self, *, min_dist: int = base.MIN_DIST, n_neighbors: int = 1):
        self._umap = umap.UMAP()
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
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
