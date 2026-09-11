# tmap
Temporal UMAP

**WORK IN PROGRESS**

A UMAP-style dimensionality reduction for collections of trajectories. Rather
than building a k-nearest-neighbour graph over independent points, tmap builds
its graph from pairwise trajectory *alignments* (dynamic time warping or optimal
transport). The resulting embedding respects the temporal ordering within each
trajectory and the correspondence between trajectories that follow similar
paths at different rates or phases.

![output](https://github.com/user-attachments/assets/34d8fd33-3165-4f81-bdd0-c11cc71f0f41)

## installation

```sh
git clone https://github.com/quantumjot/tmap
pip install -e .
```

## usage

Input is a *list* of `(n_i, m)` arrays — one per trajectory, with `n_i`
timepoints of `m` features. Trajectories may have different lengths and need not
be synchronized.

```python
from tmap import TemporalMAP
from tmap.simulate import simulate_trajectories

trajectories = simulate_trajectories()   # list of (n_i, m) arrays
mapper = TemporalMAP(n_neighbors=30, min_dist=0.01, random_state=42)
y = mapper.fit(trajectories)             # (N, 2), N = sum of n_i
```

Key options:

- `aligner` — `DTWAlignment(window=...)` (default) or `OTAlignment()` from
  `tmap.alignment`. The DTW `window` balances local temporal features against
  global trajectory warping.
- `layout` — initial layout, `"SPECTRAL"` (default), `"TEMPORAL"`, `"UMAP"` or
  `"RANDOM"`; see `tmap.layout`.
- `optimizer` — `"sampled"` (default; stochastic, `O(nnz)` per epoch) or
  `"dense"` (full-batch, `O(N²)` per iteration).
- `n_components` — embedding dimensionality, defaults to `2`.
- `n_jobs` — workers for the pairwise alignments; `-1` uses all cores.
- `random_state` — seed the stochastic optimiser for reproducible embeddings.

### plotting

```python
from tmap.utils import plot_embeddings

plot_embeddings(mapper, title="tmap")
```

`DefaultUMAP` wraps plain UMAP with the same interface, for side-by-side
comparison:

```python
from tmap import DefaultUMAP

umapper = DefaultUMAP(n_neighbors=30, min_dist=0.01, random_state=42)
_ = umapper.fit(trajectories)
plot_embeddings(umapper, title="umap")
```

## more

- [`examples/fit.ipynb`](examples/fit.ipynb) — a worked example on real tracking
  data, comparing tmap against UMAP across `n_neighbors`.
- [`docs/method-math.md`](docs/method-math.md) — the mathematical specification
  of the pipeline.
