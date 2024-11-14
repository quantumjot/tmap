import numpy as np
from dtaidistance import dtw, dtw_ndim

from scipy import optimize
from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm
from typing import Callable, List, Optional

from jax import grad, jit
import jax.numpy as jnp


EPSILON_WEIGHT = np.inf
N_NEIGHBORS = 15
MIN_DIST = 0.25
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 200
LATEN_DIMS = 32


def masked_path(paths, best_path) -> np.ndarray:
    """ Calculate the adjacency matrix in high dimensional space.

    Parameters
    ----------

    Returns
    -------
    """
    i, j = zip(*best_path)
    paths = paths[1:, 1:]
    masked = np.ones(paths.shape) * 0
    dpath = paths[i, j] #- np.concatenate([[0,], paths[i, j]])[:-1]
    masked[i, j] = dpath
    return masked


def calculate_distance_matrix(
    sequences: List[np.ndarray], window: Optional[int] = None
) -> np.ndarray:
    """Calculate the distance matrix.

    Parameters
    ----------


    Returns
    -------


    Notes
    -----
    This could be a sparse matrix in practice.
    """
    seq_shapes = [s.shape[0] for s in sequences]

    n = sum(seq_shapes)

    distance_matrix = np.ones((n, n)) * 0

    for i in tqdm(range(len(sequences)), desc="DTW"):
        for j in range(i + 1, len(sequences)):

            s1, s2 = sequences[i], sequences[j]

            d, paths = dtw_ndim.warping_paths(s1, s2, window=window)
            best_path = dtw.best_path(paths)

            mask = masked_path(paths, best_path)

            sx = slice(sum(seq_shapes[:i]), sum(seq_shapes[: i + 1]), 1)
            sy = slice(sum(seq_shapes[:j]), sum(seq_shapes[: j + 1]), 1)
            distance_matrix[sx, sy] = mask

    # now make the matrix symmetric
    distance_matrix = distance_matrix + distance_matrix.T
    distance_matrix[distance_matrix == 0] = EPSILON_WEIGHT
    distance_matrix[np.eye(n).astype(bool)] = 0.0

    return distance_matrix


def high_dimensional_probability(d: np.ndarray, sigma: float) -> np.ndarray:
    d = np.clip(d, 0., np.inf) # clamp to greater than zero
    assert sigma > 0.0
    return np.exp( -d / sigma)


def estimate_sigma(
    d: np.ndarray, n_neighbors: int, iterations: int = 20, tolerance: float = 1e-5
):
    """
    Parameters
    ----------

    Returns
    -------
    """

    k = lambda p: np.power(2.0, np.sum(p))
    k_of_sigma = lambda sigma: k(high_dimensional_probability(d, sigma))

    sigma_lower_estimate = 0.0
    sigma_upper_estimate = 1000.0

    for iter in range(iterations):
        sigma_estimate = (sigma_lower_estimate + sigma_upper_estimate) / 2
        if k_of_sigma(sigma_estimate) < n_neighbors:
            sigma_lower_estimate = sigma_estimate
        else:
            sigma_upper_estimate = sigma_estimate
        if np.abs(n_neighbors - k_of_sigma(sigma_estimate)) <= tolerance:
            break

    return sigma_estimate


def calculate_high_dimensional_probability_matrix(
    dist: np.ndarray, n_neighbors: int,
) -> np.ndarray:
    """Calculate the high dimensional probability matrix from the adjacency
    matrix representation of the graph.

    Parameters
    ----------
    dist : np.ndarray
    n_neighbors : int

    Returns
    -------
    prob : np.ndarray
    """

    # calculate the minimum (non-zero) distance for each row
    rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
    prob = np.zeros_like(dist, dtype=np.float32)

    for row in range(prob.shape[0]):
        d = dist[row, ...] - rho[row]
        sigma = estimate_sigma(d, n_neighbors)
        prob[row, ...] = high_dimensional_probability(d, sigma)

    # make the distances compatible by enforcing symmetry
    prob = symmetrize_probability_matrix_umap(prob)

    return prob



def symmetrize_probability_matrix_tsne(prob: np.ndarray) -> np.ndarray:
    return prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))


def symmetrize_probability_matrix_umap(prob: np.ndarray) -> np.ndarray:
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

    x = np.linspace(0, 3, 300)

    def f(x, min_dist):
        y = []
        for i in range(len(x)):
            if(x[i] <= min_dist):
                y.append(1)
            else:
                y.append(np.exp(- x[i] + min_dist))
        return y

    dist_low_dim = lambda x, a, b: 1 / (1 + a*x**(2*b))

    p , _ = optimize.curve_fit(dist_low_dim, x, f(x, min_dist))

    a = p[0]
    b = p[1]

    return 1.0, 1.0 #a, b


@jit
def jax_euclidean_distances(i, j):
    """
    Parameters
    ----------

    Returns
    -------
    """
    M = i.shape[0]
    N = j.shape[0]
    I_dots = jnp.reshape(jnp.sum((i*i), axis=1), (M, 1)) * jnp.ones(shape=(1, N))
    J_dots = jnp.sum((j*j), axis=1) * jnp.ones(shape=(M, 1))
    D_squared =  I_dots + J_dots - 2*jnp.dot(i, j.T)
    return D_squared


@jit
def jax_inverse_dist(a, b, d_squared):
    """
    Parameters
    ----------

    Returns
    -------
    """
    return jnp.power(1.0 + a * jnp.power(d_squared, b), -1)


@jit
def jax_cross_entropy(p, y, a, b):
    """
    Parameters
    ----------

    Returns
    -------
    """
    d_squared = jax_euclidean_distances(y, y)
    q = jax_inverse_dist(a, b, d_squared)
    return -p * jnp.log(q + 0.01) - (1 - p) * jnp.log(1 - q + 0.01)


@jit
def jax_cross_entropy_gradient(p, y, a, b):
    """
    Parameters
    ----------

    Returns
    -------
    """
    n = y.shape[0]
    d = jax_euclidean_distances(y, y)
    y_diff = jnp.expand_dims(y, 1) - jnp.expand_dims(y, 0)
    inv_dist = jnp.power(1 + a * d**b, -1)
    q = jnp.dot(1 - p, jnp.power(0.001 + d, -1))
    q = q * (1-jnp.identity(n))
    q = q / jnp.sum(q, axis=1, keepdims=True)
    fact = jnp.expand_dims(a * p * (1e-8 + d)**(b-1) - q, 2)
    return 2 * b * jnp.sum(fact * y_diff * jnp.expand_dims(inv_dist, 2), axis=1)


class TemporalMAP:
    """ TemporalMAP.

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
    pre_embedding_fn : Callable
        A function to initialize the embedding in low dimensional space
    distance_fn : Callable
        A function to calculate the distances in sequence space.

    Attributes
    ----------
    trajectories : list
    sequence_shapes : list

    """
    def __init__(self,
        n_neighbors: int = N_NEIGHBORS,
        min_dist: float = MIN_DIST,
        n_components: int = 2,
        pre_embedding_fn: Callable = SpectralEmbedding,
        distance_fn: Callable = dtw_ndim.warping_paths,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components

        self._sequences = []
        self._P = None
        self._embedding = None

    def fit(
        self,
        sequences: List[np.ndarray],
        learning_rate: float = LEARNING_RATE,
        max_iterations: int = MAX_ITERATIONS,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        sequences : list of arrays
        learning_rate : float
        max_iterations : int

        Returns
        -------
        y : np.ndarray (N, n_components)
            The embedding in `n_components` dimensions.
        """

        self._sequences = sequences

        dist = calculate_distance_matrix(sequences, window=window)
        prob = calculate_high_dimensional_probability_matrix(dist, self.n_neighbors)

        a, b = find_hyperparameters(self.min_dist)

        np.random.seed(42)
        X_train = np.concatenate(sequences, axis=0)
        model = SpectralEmbedding(
            n_components=self.n_components, n_neighbors=X_train.shape[-1]
        )
        y = model.fit_transform(X_train)

        # now do gradient descent to find the embedding
        for i in tqdm(range(max_iterations), desc="Embedding"):
            y = y - learning_rate * jax_cross_entropy_gradient(prob, y, a, b)

        self._embedding = y
        self._P = prob

        return y

    def transform(self):
        raise NotImplementedError

    @property
    def sequence_shapes(self) -> List[int]:
        """Shapes/Lengths of the sequences used.

        Returns
        -------
        """
        return [s.shape[0] for s in self._sequences]

    @property
    def trajectories(self) -> List[np.ndarray]:
        """Trajectories in the low dimensional representation.

        Returns
        -------
        """
        seq = self.sequence_shapes
        slice_seq = lambda idx: slice(sum(seq[:idx]), sum(seq[:idx+1]), 1)
        return [self._embeddings[slice_seq(i), ...] for i in range(len(seq))]
