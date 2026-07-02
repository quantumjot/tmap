"""Low-dimensional embedding optimisers.

Two optimisers are provided:

``optimizer="sampled"`` (default)
    UMAP-style stochastic optimisation over the *edges* of the sparse
    high-dimensional graph with negative sampling for repulsion. Cost per
    epoch is O(nnz * n_negative) instead of O(N^2), which makes the
    embedding stage scale with the graph rather than the square of the
    number of nodes. Uses the standard UMAP per-pair kernel
    ``q = 1 / (1 + a * d^2b)`` and is deterministic for a fixed
    ``random_state``.

``optimizer="dense"``
    The original full-batch gradient descent on the dense cross-entropy
    with row-normalised ``q``. Kept for exact backward compatibility with
    embeddings produced by earlier versions; O(N^2) per iteration.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from scipy import sparse
from tqdm import tqdm

from tmap import base

# Minimum squared distance used inside the UMAP gradient coefficients to
# avoid division by zero / unbounded powers when two points coincide.
_MIN_DSQ = 1e-3

# UMAP-conventional elementwise gradient clip.
_GRAD_CLIP = 4.0


# ---------------------------------------------------------------------------
# dense (full-batch) optimiser — original implementation, unchanged maths
# ---------------------------------------------------------------------------


@jax.jit
def jax_euclidean_distances(i, j):
    """Calculate the (squared) Euclidean distances between two arrays.

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


def optimize_embedding_dense(
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
    """Optimise the embedding with the original dense full-batch SGD."""
    if sparse.issparse(p):
        p = p.toarray()
    p = jnp.asarray(p)
    y = jnp.asarray(y)
    lr = jnp.asarray(learning_rate, dtype=y.dtype)

    remaining = n_iterations
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


# ---------------------------------------------------------------------------
# sampled (sparse edge + negative sampling) optimiser
# ---------------------------------------------------------------------------


def edges_from_matrix(
    P,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Extract the directed edge list of the probability graph.

    Parameters
    ----------
    P : npt.NDArray or scipy.sparse matrix, shape (N, N)
        Symmetric probability/membership matrix. Off-diagonal nonzeros are
        the graph edges; the diagonal is ignored.

    Returns
    -------
    heads, tails : npt.NDArray of int32, shape (E,)
        Directed edge endpoints. Both directions of each undirected edge are
        present (P is symmetric), sorted by ``heads`` so scatter-adds in the
        optimiser hit contiguous segments.
    weights : npt.NDArray of float32, shape (E,)
        Membership strength p_ij for each edge.
    """
    coo = sparse.coo_matrix(P)
    weights = coo.data.astype(np.float32)
    # drop the diagonal and any weight that underflows to zero in float32 —
    # both gradient terms are weighted by p, so those edges contribute nothing
    keep = (coo.row != coo.col) & (weights > 0)
    heads = coo.row[keep]
    tails = coo.col[keep]
    weights = weights[keep]

    order = np.lexsort((tails, heads))
    return (
        heads[order].astype(np.int32),
        tails[order].astype(np.int32),
        weights[order],
    )


def _sampled_gradient(y, heads, tails, weights, negs, a, b, gamma):
    """UMAP attraction/repulsion gradient accumulated per node.

    Uses the closed-form UMAP gradient coefficients for the per-pair kernel
    ``q = 1 / (1 + a * dsq**b)`` (``dsq`` is the squared low-D distance,
    matching the convention of :func:`find_hyperparameters`).

    Both terms are weighted by the edge membership strength ``p``: reference
    UMAP samples each edge with frequency proportional to p and triggers
    ``n_negative`` repulsive updates per sampled edge, so the *expected*
    per-epoch update for an edge is ``p * (attraction + repulsion)``. This
    vectorised form applies that expectation directly.
    """
    n_vertices = y.shape[0]
    yh = y[heads]
    yt = y[tails]

    # attraction along graph edges
    diff = yh - yt
    dsq = jnp.maximum(jnp.sum(diff * diff, axis=1), _MIN_DSQ)
    coef_a = (-2.0 * a * b * dsq ** (b - 1.0)) / (1.0 + a * dsq**b)
    grad_a = (weights * coef_a)[:, None] * diff

    # repulsion from uniformly sampled negatives
    yn = y[negs]
    diff_n = yh[:, None, :] - yn
    dsq_n = jnp.sum(diff_n * diff_n, axis=2)
    coef_r = (2.0 * gamma * b) / (
        (_MIN_DSQ + dsq_n) * (1.0 + a * jnp.maximum(dsq_n, _MIN_DSQ) ** b)
    )
    grad_r = jnp.sum((weights[:, None] * coef_r)[:, :, None] * diff_n, axis=1)

    return jax.ops.segment_sum(
        grad_a + grad_r, heads, num_segments=n_vertices, indices_are_sorted=True
    )


@partial(jax.jit, static_argnames=("n_steps", "n_negative"))
def _sgd_sampled_chunk(
    heads,
    tails,
    weights,
    y,
    a,
    b,
    learning_rate,
    gamma,
    epoch_start,
    n_epochs_total,
    key,
    n_steps,
    n_negative,
):
    """Run ``n_steps`` sampled-SGD epochs on-device.

    The learning rate decays linearly over the *global* epoch index
    (``epoch_start + e``) so chunking for the progress bar does not change
    the schedule. The PRNG key is threaded through the loop carry, so the
    negative samples — and therefore the result — are fully determined by
    the initial key.
    """
    n_vertices = y.shape[0]

    def body(e, carry):
        y, key = carry
        key, sub = jax.random.split(key)
        negs = jax.random.randint(sub, (heads.shape[0], n_negative), 0, n_vertices)
        grad = _sampled_gradient(y, heads, tails, weights, negs, a, b, gamma)
        alpha = learning_rate * (1.0 - (epoch_start + e) / n_epochs_total)
        y = y + alpha * jnp.clip(grad, -_GRAD_CLIP, _GRAD_CLIP)
        return (y, key)

    return jax.lax.fori_loop(0, n_steps, body, (y, key))


@partial(jax.jit, static_argnames=("n_negative",))
def _sampled_loss(heads, tails, weights, y, a, b, key, n_negative, *, eps=1e-8):
    """Deterministic evaluation loss for the sampled objective.

    Exact attractive cross-entropy over all edges plus a Monte-Carlo estimate
    of the repulsive term using negatives drawn from the *given* key — pass a
    fixed key to get a loss that is comparable across iterations.
    """
    n_vertices = y.shape[0]
    diff = y[heads] - y[tails]
    dsq = jnp.sum(diff * diff, axis=1)
    q = 1.0 / (1.0 + a * jnp.maximum(dsq, _MIN_DSQ) ** b)
    loss_attract = -jnp.sum(weights * jnp.log(q + eps))

    negs = jax.random.randint(key, (heads.shape[0], n_negative), 0, n_vertices)
    diff_n = y[heads][:, None, :] - y[negs]
    dsq_n = jnp.sum(diff_n * diff_n, axis=2)
    q_n = 1.0 / (1.0 + a * jnp.maximum(dsq_n, _MIN_DSQ) ** b)
    loss_repel = -jnp.sum(jnp.log(1.0 - q_n + eps)) / n_negative

    return loss_attract + loss_repel


def sampled_embedding_loss(
    edges: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    y: npt.NDArray,
    a: float,
    b: float,
    *,
    n_negative: int = 5,
    seed: int = 0,
) -> float:
    """Evaluate the sampled objective with a fixed seed (deterministic)."""
    heads, tails, weights = edges
    loss = _sampled_loss(
        jnp.asarray(heads),
        jnp.asarray(tails),
        jnp.asarray(weights),
        jnp.asarray(y),
        a,
        b,
        jax.random.PRNGKey(seed),
        n_negative,
    )
    return float(loss)


def optimize_embedding_sampled(
    edges: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    y: npt.NDArray,
    a: float,
    b: float,
    *,
    learning_rate: float,
    n_iterations: int,
    n_negative: int = 5,
    gamma: float = 1.0,
    random_state: int | None = None,
    chunk: int = 20,
    progress: bool = True,
) -> npt.NDArray:
    """Optimise the embedding over the graph edges with negative sampling.

    Parameters
    ----------
    edges : tuple of arrays
        ``(heads, tails, weights)`` directed edge list, e.g. from
        :func:`edges_from_matrix`.
    y : npt.NDArray, shape (N, n_components)
        Initial embedding.
    a, b : float
        Kernel hyperparameters from :func:`find_hyperparameters`.
    learning_rate : float
        Initial learning rate; decays linearly to zero over the run.
    n_iterations : int
        Number of epochs (each epoch visits every edge once).
    n_negative : int
        Negative samples drawn per edge per epoch.
    gamma : float
        Repulsion strength.
    random_state : int, optional
        Seed for the negative sampling; a fixed seed makes the result
        deterministic. ``None`` uses seed 0.
    """
    heads, tails, weights = edges
    heads = jnp.asarray(heads)
    tails = jnp.asarray(tails)
    weights = jnp.asarray(weights)
    y = jnp.asarray(y, dtype=jnp.float32)
    key = jax.random.PRNGKey(0 if random_state is None else int(random_state))

    done = 0
    pbar = tqdm(total=n_iterations, disable=not progress)
    while done < n_iterations:
        steps = min(chunk, n_iterations - done)
        y, key = _sgd_sampled_chunk(
            heads,
            tails,
            weights,
            y,
            a,
            b,
            learning_rate,
            gamma,
            done,
            n_iterations,
            key,
            steps,
            n_negative,
        )
        done += steps
        if progress:
            loss = _sampled_loss(
                heads, tails, weights, y, a, b, jax.random.PRNGKey(0), n_negative
            )
            pbar.update(steps)
            pbar.set_description(f"Embedding (loss: {float(loss):.3f})")
    pbar.close()

    return np.asarray(y)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def optimize_embedding(
    p,
    y: npt.NDArray,
    a: float,
    b: float,
    *,
    learning_rate: float | None = None,
    n_iterations: int,
    chunk: int = 20,
    progress: bool = True,
    optimizer: str = "sampled",
    n_negative: int = 5,
    repulsion_strength: float = 1.0,
    random_state: int | None = None,
) -> npt.NDArray:
    """Optimise the low-dimensional embedding.

    ``optimizer="sampled"`` (default) runs the O(nnz) UMAP-style stochastic
    optimiser over the nonzero edges of ``p``; ``optimizer="dense"`` runs the
    original O(N^2) full-batch gradient descent and reproduces embeddings
    from earlier versions exactly.

    ``learning_rate=None`` selects the per-optimiser default: the two
    optimisers have different gradient scales (the sampled path uses the
    UMAP-conventional initial rate of 1.0 with linear decay, the dense path
    keeps the historical 0.1 constant rate).
    """
    if optimizer == "dense":
        return optimize_embedding_dense(
            p,
            y,
            a,
            b,
            learning_rate=base.LEARNING_RATE if learning_rate is None else learning_rate,
            n_iterations=n_iterations,
            chunk=chunk,
            progress=progress,
        )
    if optimizer != "sampled":
        raise ValueError(f"Unknown optimizer: {optimizer!r} (use 'sampled' or 'dense')")

    edges = edges_from_matrix(p)
    return optimize_embedding_sampled(
        edges,
        y,
        a,
        b,
        learning_rate=base.LEARNING_RATE_SAMPLED if learning_rate is None else learning_rate,
        n_iterations=n_iterations,
        n_negative=n_negative,
        gamma=repulsion_strength,
        random_state=random_state,
        chunk=chunk,
        progress=progress,
    )
