"""Evaluation metrics for tmap embeddings and alignment graphs.

The canonical home for measurement, so that separate lines of work are scored
the same way and their numbers are comparable. Four groups:

*Supervised* (require ground-truth trajectory groups, e.g. from
:func:`tmap.simulate.simulate_trajectories` with ``return_labels=True``):
:func:`knn_purity`, :func:`silhouette`.

*Graph-referenced* — score the embedding against the graph tmap actually
optimises, rather than against the high-dimensional input:
:func:`graph_neighbor_preservation`.

*Temporal* — use the fact that node order within a trajectory is elapsed time:
:func:`temporal_coherence` (is the next timepoint nearby?) and
:func:`velocity_coherence` (do neighbouring trajectories *flow* the same way?).

*Alignment-level* — score the correspondence structure before embedding:
:func:`transitivity_violation`, :func:`edge_recall`.

Note on references: :func:`knn_preservation` and ``trustworthiness`` compare the
embedding to the concatenated high-dimensional input. That is the standard DR
measure, but tmap's graph comes from DTW/OT alignment and *not* from a
high-dimensional kNN graph, so a change can legitimately improve the alignment
while moving these numbers. Prefer :func:`graph_neighbor_preservation` and the
supervised metrics when judging alignment changes.
"""
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from scipy import sparse
from sklearn.manifold import trustworthiness  # re-exported
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "node_labels",
    "knn_purity",
    "silhouette",
    "graph_neighbor_preservation",
    "temporal_coherence",
    "velocity_coherence",
    "knn_preservation",
    "trustworthiness",
    "correspondence_map",
    "transitivity_violation",
    "edge_recall",
    "summarize",
]


def _embedded_knn(y: npt.NDArray, k: int) -> npt.NDArray:
    """Indices of each row's ``k`` nearest neighbours in ``y``, excluding self.

    Calling ``kneighbors()`` with no argument queries the fitted data and
    already drops each point's self-match, so ``n_neighbors=k`` returns exactly
    ``k`` genuine neighbours. (Passing ``X`` explicitly would instead include
    self as column 0; the historical ``n_neighbors=k + 1`` followed by
    ``[:, 1:]`` silently discarded each point's *nearest* neighbour and scored
    ranks 2..k+1.)
    """
    k_eff = min(k, y.shape[0] - 1)
    if k_eff < 1:
        raise ValueError(f"need at least 2 points to compute neighbours, got {y.shape[0]}")
    nn = NearestNeighbors(n_neighbors=k_eff).fit(y)
    return nn.kneighbors(return_distance=False)


def node_labels(seq_lengths: Sequence[int], trajectory_labels: npt.NDArray) -> npt.NDArray:
    """Expand per-trajectory labels to per-node labels.

    Node order matches the concatenation order used throughout tmap, so the
    result aligns with the rows of an embedding.
    """
    trajectory_labels = np.asarray(trajectory_labels)
    if len(seq_lengths) != trajectory_labels.shape[0]:
        raise ValueError(
            f"{len(seq_lengths)} sequences but {trajectory_labels.shape[0]} labels"
        )
    return np.repeat(trajectory_labels, np.asarray(seq_lengths))


# --- supervised ------------------------------------------------------------


def knn_purity(y: npt.NDArray, labels: npt.NDArray, *, k: int = 15) -> float:
    """Mean fraction of each node's embedded kNN sharing its label.

    1.0 is perfect group separation. The chance level is the expected fraction
    of same-label points under a random embedding, so compare against that
    rather than against 0.
    """
    nn_y = _embedded_knn(y, k)
    labels = np.asarray(labels)
    return float(np.mean(labels[nn_y] == labels[:, None]))


def silhouette(y: npt.NDArray, labels: npt.NDArray) -> float:
    """Silhouette score of the ground-truth groups in the embedding, in [-1, 1]."""
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(silhouette_score(y, labels))


# --- graph-referenced ------------------------------------------------------


def graph_neighbor_preservation(P, y: npt.NDArray, *, k: int = 15) -> float:
    """Mean overlap of each node's top-k graph neighbours with its embedded kNN.

    Scores the embedding against the probability graph tmap optimises, so it is
    the right reference when the graph itself has not changed.
    """
    P = sparse.csr_matrix(P)
    nn_y = _embedded_knn(y, k)
    overlaps = []
    for u in range(P.shape[0]):
        lo, hi = P.indptr[u], P.indptr[u + 1]
        cols, data = P.indices[lo:hi], P.data[lo:hi]
        keep = cols != u  # drop the self/diagonal entry
        cols, data = cols[keep], data[keep]
        if cols.size == 0:
            continue
        topk = cols[np.argsort(data)[::-1][:k]]
        overlaps.append(len(set(topk) & set(nn_y[u])) / min(k, topk.size))
    return float(np.mean(overlaps)) if overlaps else float("nan")


# --- temporal --------------------------------------------------------------


def temporal_coherence(seq_lengths: Sequence[int], y: npt.NDArray, *, k: int = 15) -> float:
    """Fraction of nodes whose next timepoint is among their embedded kNN."""
    nn_y = _embedded_knn(y, k)
    hits = total = 0
    offset = 0
    for length in seq_lengths:
        for t in range(length - 1):
            u = offset + t
            if (u + 1) in nn_y[u]:
                hits += 1
            total += 1
        offset += length
    return hits / total if total else float("nan")


def velocity_coherence(
    seq_lengths: Sequence[int], y: npt.NDArray, *, k: int = 15
) -> float:
    """Mean cosine alignment between a node's velocity and its neighbours'.

    The displacement ``y[u + 1] - y[u]`` within a trajectory is the embedded
    velocity. This returns the mean cosine similarity between each node's
    velocity and the velocities of its ``k`` embedded neighbours, so it measures
    whether nearby trajectories *flow the same way* rather than merely sit
    close together.

    +1 means locally co-flowing, 0 means neighbouring velocities are unrelated,
    -1 means neighbours systematically counter-flow. A symmetric point-cloud
    embedding has no mechanism to control this, so it is the natural target
    metric for a directional regularizer.

    Nodes at the end of a trajectory have no velocity and are excluded, as are
    nodes whose velocity is exactly zero.
    """
    seq_lengths = list(seq_lengths)
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(np.int64)

    idx = np.concatenate(
        [np.arange(offsets[s], offsets[s + 1] - 1) for s in range(len(seq_lengths))]
    )
    if idx.size < 2:
        return float("nan")

    velocity = y[idx + 1] - y[idx]
    norm = np.linalg.norm(velocity, axis=1)
    alive = norm > 0
    if alive.sum() < 2:
        return float("nan")

    idx, velocity, norm = idx[alive], velocity[alive], norm[alive]
    unit = velocity / norm[:, None]

    # neighbours are sought among velocity-bearing nodes only, in embedding space
    nn_local = _embedded_knn(y[idx], k)
    return float(np.mean(np.einsum("ij,ikj->ik", unit, unit[nn_local])))


# --- high-dimensional reference -------------------------------------------


def knn_preservation(x: npt.NDArray, y: npt.NDArray, *, k: int = 15) -> float:
    """Mean fraction of each point's high-D kNN preserved in the embedding."""
    nn_x = _embedded_knn(x, k)
    nn_y = _embedded_knn(y, k)
    return float(
        np.mean([len(set(a) & set(b)) / a.size for a, b in zip(nn_x, nn_y)])
    )


# --- alignment-level -------------------------------------------------------


def correspondence_map(aligner, seq_i: npt.NDArray, seq_j: npt.NDArray) -> npt.NDArray:
    """Best-matching index in ``seq_j`` for each index of ``seq_i``.

    Returns ``-1`` where the aligner leaves an index uncorresponded. Aligners
    exposing ``transport_mass_to_distance`` (i.e. OT, whose values are
    transported mass, larger being better) are converted to distances first, so
    the minimum is always the best match.
    """
    rows, cols, vals = aligner(seq_i, seq_j, mask=True, sparse=True)
    if hasattr(aligner, "transport_mass_to_distance"):
        vals = aligner.transport_mass_to_distance(vals)

    best = np.full(seq_i.shape[0], -1, dtype=np.int64)
    if len(rows) == 0:
        return best

    # first entry per row after sorting by (row, value) is that row's best match
    order = np.lexsort((vals, rows))
    r, c = np.asarray(rows)[order], np.asarray(cols)[order]
    first = np.ones(r.size, dtype=bool)
    first[1:] = r[1:] != r[:-1]
    best[r[first]] = c[first]
    return best


def transitivity_violation(
    sequences: list[npt.NDArray],
    aligner,
    *,
    n_triples: int = 20,
    seed: Optional[int] = None,
) -> float:
    """Mean normalised disagreement between composed and direct correspondences.

    For a triple ``(i, j, k)``, aligning ``i->j`` then ``j->k`` should land where
    aligning ``i->k`` directly lands. This reports the mean absolute index
    disagreement, normalised by the length of trajectory ``k``, so 0 is
    perfectly consistent and 1 is maximally inconsistent. Independent pairwise
    alignment offers no guarantee here, which is what a global/joint alignment
    is meant to fix.

    For scale: uniformly random correspondences give roughly 1/3.
    """
    if len(sequences) < 3:
        raise ValueError("need at least 3 sequences to form a triple")

    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(n_triples):
        i, j, k = rng.choice(len(sequences), size=3, replace=False)
        m_ij = correspondence_map(aligner, sequences[i], sequences[j])
        m_jk = correspondence_map(aligner, sequences[j], sequences[k])
        m_ik = correspondence_map(aligner, sequences[i], sequences[k])

        valid = (m_ij >= 0) & (m_ik >= 0)
        if not valid.any():
            continue
        composed = m_jk[m_ij[valid]]
        direct = m_ik[valid]
        ok = composed >= 0
        if not ok.any():
            continue
        span = max(sequences[k].shape[0] - 1, 1)
        errors.append(np.mean(np.abs(composed[ok] - direct[ok])) / span)

    return float(np.mean(errors)) if errors else float("nan")


def edge_recall(full, screened) -> float:
    """Fraction of the all-pairs graph's edges retained by a screened graph.

    Compares stored structure, not values, so it is unaffected by edge
    reweighting. 1.0 means no edge was dropped.
    """
    full = sparse.csr_matrix(full)
    screened = sparse.csr_matrix(screened)
    if full.shape != screened.shape:
        raise ValueError(f"shape mismatch: {full.shape} vs {screened.shape}")
    if full.nnz == 0:
        return float("nan")

    a, b = full.copy(), screened.copy()
    a.data = np.ones_like(a.data)
    b.data = np.ones_like(b.data)
    kept = a.multiply(b)
    return float(kept.nnz / full.nnz)


# --- battery ---------------------------------------------------------------


def summarize(
    y: npt.NDArray,
    *,
    seq_lengths: Sequence[int],
    labels: Optional[npt.NDArray] = None,
    P=None,
    x: Optional[npt.NDArray] = None,
    k: int = 15,
) -> dict:
    """Run the standard metric battery, skipping anything whose inputs are absent.

    ``labels`` are per-trajectory and are expanded internally.
    """
    out = {
        "temporal_coherence": temporal_coherence(seq_lengths, y, k=k),
        "velocity_coherence": velocity_coherence(seq_lengths, y, k=k),
    }
    if labels is not None:
        per_node = node_labels(seq_lengths, labels)
        out["knn_purity"] = knn_purity(y, per_node, k=k)
        out["silhouette"] = silhouette(y, per_node)
    if P is not None:
        out["graph_neighbor_preservation"] = graph_neighbor_preservation(P, y, k=k)
    if x is not None:
        out["knn_preservation"] = knn_preservation(x, y, k=k)
        out["trustworthiness"] = float(trustworthiness(x, y, n_neighbors=k))
    return out
