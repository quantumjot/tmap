from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt
import umap

from scipy import optimize, sparse
from tqdm import tqdm

from tmap import base
from tmap.alignment import DTWAlignment, OTAlignment
from tmap.embedding import (  # noqa: F401  (re-exported for back-compat)
    edges_from_matrix,
    jax_cross_entropy_gradient_2,
    jax_euclidean_distances,
    optimize_embedding,
    optimize_embedding_dense,
    optimize_embedding_sampled,
    sampled_embedding_loss,
    _sgd_chunk,
)
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


def _pair_alignment_triplets(aligner, seq_i, seq_j, mask):
    """Align one pair and return the edges as ``(rows, cols, vals)`` triplets.

    Prefers the aligner's sparse path (no dense rectangular block); falls back
    to extracting the nonzero cells of a dense alignment output for aligners
    that do not support ``sparse=True``.
    """
    try:
        out = aligner(seq_i, seq_j, mask=mask, sparse=True)
    except TypeError:
        out = aligner(seq_i, seq_j, mask=mask)
    if isinstance(out, tuple):
        rows, cols, vals = out
        return np.asarray(rows), np.asarray(cols), np.asarray(vals, dtype=np.float32)
    block = np.asarray(out)
    rows, cols = np.nonzero(block)
    return rows, cols, block[rows, cols].astype(np.float32)


def _align_pair_worker(args):
    """Module-level worker so pairs can be dispatched to a process pool."""
    aligner, i, j, seq_i, seq_j, mask = args
    return i, j, _pair_alignment_triplets(aligner, seq_i, seq_j, mask)


def calculate_distance_matrix(
    sequences: List[npt.NDArray],
    aligner: base.AlignmentBase,
    *,
    mask: bool = True,
    n_jobs: Optional[int] = None,
) -> sparse.csr_matrix:
    """Calculate the sparse distance matrix.

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
    distance_matrix : scipy.sparse.csr_matrix
        The symmetric distance graph. Only edges (finite pairwise distances)
        are stored; absent entries denote unconnected node pairs (the dense
        representation's ``inf``) and the diagonal is implicitly zero. Use
        :func:`densify_distance_matrix` to recover the dense form.
    """

    seq_shapes = [s.shape[0] for s in sequences]
    offsets = np.concatenate([[0], np.cumsum(seq_shapes)]).astype(np.int64)
    n = int(offsets[-1])

    pairs = [(i, j) for i in range(len(sequences)) for j in range(i + 1, len(sequences))]
    desc = aligner.name + f" (masking={mask})"

    if n_jobs in (None, 1):
        results = [
            (i, j, _pair_alignment_triplets(aligner, sequences[i], sequences[j], mask))
            for i, j in tqdm(pairs, desc=desc)
        ]
    else:
        max_workers = None if n_jobs == -1 else n_jobs
        # threads when the aligner releases the GIL (compiled DTW), else processes
        use_threads = getattr(aligner, "releases_gil", False)
        if use_threads:
            def _align(pair):
                i, j = pair
                return i, j, _pair_alignment_triplets(aligner, sequences[i], sequences[j], mask)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_align, pairs), total=len(pairs), desc=desc))
        else:
            work = [(aligner, i, j, sequences[i], sequences[j], mask) for i, j in pairs]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    tqdm(executor.map(_align_pair_worker, work), total=len(pairs), desc=desc)
                )

    all_rows, all_cols, all_vals = [], [], []
    for i, j, (rows, cols, vals) in results:
        if isinstance(aligner, OTAlignment):
            # for OT, output is a weighted transport plan (correspondence);
            # convert transported mass to a distance
            vals = (1.0 / (vals + 1e-9)).astype(np.float32)
        # for DTW the output is already a distance (path cost), use directly

        if j == i + 1:
            # the sequence-boundary cell (last node of i, first node of j) sits
            # on the global super-diagonal, which is reserved for the
            # intra-sequence local distances (inf across boundaries) — an
            # alignment cell landing there is discarded
            boundary = (rows == seq_shapes[i] - 1) & (cols == 0)
            if boundary.any():
                keep = ~boundary
                rows, cols, vals = rows[keep], cols[keep], vals[keep]

        all_rows.append(rows + offsets[i])
        all_cols.append(cols + offsets[j])
        all_vals.append(vals)

    # consider the local connectivity of the trajectory too: consecutive nodes
    # within each sequence are linked by their squared feature distance
    for i, seq in enumerate(sequences):
        local = np.linalg.norm(np.diff(seq, axis=0), 2, axis=1) ** 2
        idx = offsets[i] + np.arange(seq.shape[0] - 1)
        all_rows.append(idx)
        all_cols.append(idx + 1)
        all_vals.append(local.astype(np.float32))

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)

    # zero distances denote "no edge" (the dense assembly mapped them to inf),
    # and infinite distances are non-edges by definition
    keep = (vals != 0.0) & np.isfinite(vals)
    rows, cols, vals = rows[keep], cols[keep], vals[keep]

    # all edges above are strictly upper-triangular and unique; mirror them to
    # make the graph symmetric
    distance_matrix = sparse.coo_matrix(
        (
            np.concatenate([vals, vals]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    distance_matrix.sort_indices()

    return distance_matrix


def densify_distance_matrix(dist) -> npt.NDArray:
    """Dense form of a sparse distance graph.

    Non-edges become ``inf`` (:data:`base.EPSILON_WEIGHT`) and the diagonal is
    zero, matching the historical dense representation.
    """
    if not sparse.issparse(dist):
        return dist
    dense = np.full(dist.shape, base.EPSILON_WEIGHT, dtype=np.float32)
    coo = dist.tocoo()
    dense[coo.row, coo.col] = coo.data
    np.fill_diagonal(dense, 0.0)
    return dense


def _calculate_distance_matrix_dense(
    sequences: List[npt.NDArray],
    aligner: base.AlignmentBase,
    *,
    mask: bool = True,
) -> npt.NDArray:
    """Reference dense implementation of :func:`calculate_distance_matrix`.

    Retained (serial only) to verify that the sparse assembly reproduces the
    original dense semantics exactly; used by the equivalence tests.
    """
    seq_shapes = [s.shape[0] for s in sequences]
    n = sum(seq_shapes)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    pairs = [(i, j) for i in range(len(sequences)) for j in range(i + 1, len(sequences))]

    for i, j in pairs:
        alignment_output = np.asarray(
            aligner(sequences[i], sequences[j], mask=mask), dtype=np.float32
        )
        sx = slice(sum(seq_shapes[:i]), sum(seq_shapes[: i + 1]), 1)
        sy = slice(sum(seq_shapes[:j]), sum(seq_shapes[: j + 1]), 1)

        if isinstance(aligner, OTAlignment):
            # for OT, output is a weighted transport plan (correspondence)
            distance_matrix[sx, sy] = 1.0 / (alignment_output + 1e-9)
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
    dist,
    n_neighbors: int,
):
    """Calculate the high dimensional probability matrix from the adjacency
    matrix representation of the graph.

    Parameters
    ----------
    dist : scipy.sparse matrix or npt.NDArray
        The distance graph. Sparse input (the default output of
        :func:`calculate_distance_matrix`) produces a sparse probability
        matrix; dense input keeps the historical dense path.
    n_neighbors : int

    Returns
    -------
    prob : scipy.sparse.csr_matrix or npt.NDArray
        Same layout as the input: symmetric membership strengths with a unit
        diagonal; absent sparse entries are exactly the zeros of the dense
        form (non-edges have probability ``exp(-inf) == 0``).
    """
    if sparse.issparse(dist):
        return _sparse_probability_matrix(dist.tocsr(), n_neighbors)

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


def _sparse_probability_matrix(dist: sparse.csr_matrix, n_neighbors: int) -> sparse.csr_matrix:
    """Sparse-native equivalent of the dense probability computation.

    Operates only on the stored edges; the maths matches the dense path
    exactly: non-edges (dense ``inf``) contribute ``exp(-inf) == 0`` to every
    row sum, and the diagonal (dense ``0 - rho`` clipped to 0) contributes
    exactly 1, which is added explicitly.
    """
    n = dist.shape[0]
    indptr = dist.indptr
    row_nnz = np.diff(indptr)
    if np.any(row_nnz == 0):
        raise ValueError(
            "distance graph has isolated nodes (rows without any edge); "
            "rho/sigma are undefined for such rows"
        )
    row_ids = np.repeat(np.arange(n), row_nnz)

    # per-row nearest-neighbour distance over the stored edges (the dense
    # second-smallest: the diagonal zero is the smallest, edges are > 0)
    rho = np.minimum.reduceat(dist.data, indptr[:-1])

    d = np.clip(dist.data - rho[row_ids], 0.0, np.inf)
    sigma = _estimate_sigma_sparse(d, indptr, row_ids, n, n_neighbors)
    prob_data = np.exp(-d / sigma[row_ids])

    prob = sparse.csr_matrix(
        (prob_data, dist.indices.copy(), indptr.copy()), shape=(n, n)
    )
    # symmetrise (UMAP mean) and add the unit diagonal
    prob = (prob + prob.T) * 0.5
    prob = (prob + sparse.identity(n, format="csr")).tocsr()
    prob.sort_indices()
    return prob


def _estimate_sigma_sparse(
    d: npt.NDArray,
    indptr: npt.NDArray,
    row_ids: npt.NDArray,
    n: int,
    n_neighbors: int,
    iterations: int = 20,
) -> npt.NDArray:
    """Per-row sigma bisection over the stored edges only.

    Identical to :func:`estimate_sigma_vectorized` except the per-row
    probability sum runs over the row's stored edges plus 1.0 for the
    diagonal, instead of a full dense row (whose non-edges contribute zero).
    """
    lower = np.full(n, base.SIGMA_LOW_ESTIMATE, dtype=np.float64)
    upper = np.full(n, base.SIGMA_HIGH_ESTIMATE, dtype=np.float64)
    sigma = (lower + upper) / 2

    for _ in range(iterations):
        sigma = (lower + upper) / 2
        prob = np.exp(-d / sigma[row_ids])
        row_sum = np.add.reduceat(prob, indptr[:-1]) + 1.0  # +1.0 = diagonal
        k = np.power(2.0, np.clip(row_sum, 0, 31.0))
        below = k < n_neighbors
        lower = np.where(below, sigma, lower)
        upper = np.where(below, upper, sigma)

    return sigma


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
    optimizer : str
        Embedding optimiser: ``"sampled"`` (default, UMAP-style stochastic
        optimisation over graph edges with negative sampling, O(nnz) per
        epoch) or ``"dense"`` (original full-batch gradient descent, O(N^2)
        per iteration, reproduces embeddings from earlier versions exactly).
    n_negative : int
        Negative samples per edge per epoch (sampled optimiser only).
    repulsion_strength : float
        Weight of the repulsive term (sampled optimiser only).
    random_state : int, optional
        Seed for the stochastic optimiser; fixed seed gives deterministic
        embeddings.

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
        optimizer: str = "sampled",
        n_negative: int = base.N_NEGATIVE,
        repulsion_strength: float = base.REPULSION_STRENGTH,
        random_state: Optional[int] = None,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.window = window
        self.mask = mask
        self.n_jobs = n_jobs
        self.optimizer = optimizer
        self.n_negative = n_negative
        self.repulsion_strength = repulsion_strength
        self.random_state = random_state

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
        learning_rate: Optional[float] = None,
        max_iterations: int = base.MAX_ITERATIONS,
    ) -> npt.NDArray:
        """
        Parameters
        ----------
        sequences : list of arrays
            These should be high dimensional arrays that represent the trajectories.
            [(n_i, m), (n_j, m), ..., (n_k, m)] where n_i, is the length of
            trajectory i and m is the number of features for each timepoint.
        learning_rate : float, optional
            The learning rate for the optimization. ``None`` (default) uses
            the per-optimiser default (1.0 with linear decay for
            ``optimizer="sampled"``, 0.1 constant for ``optimizer="dense"``).
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
            prob,
            y,
            a,
            b,
            learning_rate=learning_rate,
            n_iterations=max_iterations,
            optimizer=self.optimizer,
            n_negative=self.n_negative,
            repulsion_strength=self.repulsion_strength,
            random_state=self.random_state,
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
