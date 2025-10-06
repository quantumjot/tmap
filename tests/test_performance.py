"""Performance tests comparing parallel vs non-parallel implementations."""
import time
import numpy as np
import pytest
from pathlib import Path

from tmap import TemporalMAP
from tmap.alignment import DTWAlignment, OTAlignment
from tmap.layout import SpectralLayout


@pytest.fixture
def small_sequences():
    """Small test sequences (5 sequences, 10-15 timepoints each)."""
    np.random.seed(42)
    return [
        np.random.random((10, 3)),
        np.random.random((12, 3)),
        np.random.random((15, 3)),
        np.random.random((11, 3)),
        np.random.random((13, 3)),
    ]


@pytest.fixture
def medium_sequences():
    """Medium test sequences (10 sequences, 20-30 timepoints each)."""
    np.random.seed(42)
    return [np.random.random((np.random.randint(20, 30), 10)) for _ in range(10)]


@pytest.fixture
def large_sequences():
    """Large test sequences (25 sequences, 30-50 timepoints each)."""
    np.random.seed(42)
    return [np.random.random((np.random.randint(30, 50), 10)) for _ in range(25)]


@pytest.fixture
def real_data():
    """Load real tracking data if available."""
    data_path = Path("examples/data/trimmed_tracks.npy")
    if data_path.exists():
        tracks_raw = np.load(data_path)
        tracks = [tracks_raw[tracks_raw[:, 0] == idx, 1:] for idx in np.unique(tracks_raw[:, 0])]
        return tracks[:20]  # Use first 20 tracks for testing
    return None


def measure_time(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


@measure_time
def fit_tmap(sequences, **kwargs):
    """Fit TemporalMAP with timing."""
    # Extract max_iterations from kwargs if present, otherwise default to 100
    max_iterations = kwargs.pop('max_iterations', 100)
    # Extract other fit parameters
    aligner = kwargs.pop('aligner', None)
    layout = kwargs.pop('layout', None)

    # Create mapper with remaining kwargs
    init_kwargs = kwargs.copy()
    if aligner is not None:
        init_kwargs['aligner'] = aligner
    if layout is not None:
        init_kwargs['layout'] = layout

    mapper = TemporalMAP(**init_kwargs)
    embedding = mapper.fit(sequences, max_iterations=max_iterations)
    return embedding


class TestDistanceMatrixPerformance:
    """Test performance of distance matrix calculation."""

    @pytest.mark.parametrize("use_parallel", [False, True])
    def test_distance_matrix_small(self, small_sequences, use_parallel):
        """Test distance matrix calculation on small dataset."""
        mapper = TemporalMAP(
            use_parallel_distance_matrix=use_parallel,
        )

        start = time.perf_counter()
        dist_matrix = mapper.calculate_distance_matrix(small_sequences)
        elapsed = time.perf_counter() - start

        assert dist_matrix.shape[0] == sum(s.shape[0] for s in small_sequences)
        print(f"\nDistance matrix (parallel={use_parallel}): {elapsed:.3f}s")

    @pytest.mark.parametrize("use_parallel", [False, True])
    def test_distance_matrix_medium(self, medium_sequences, use_parallel):
        """Test distance matrix calculation on medium dataset."""
        mapper = TemporalMAP(
            use_parallel_distance_matrix=use_parallel,
        )

        start = time.perf_counter()
        dist_matrix = mapper.calculate_distance_matrix(medium_sequences)
        elapsed = time.perf_counter() - start

        assert dist_matrix.shape[0] == sum(s.shape[0] for s in medium_sequences)
        print(f"\nDistance matrix (parallel={use_parallel}): {elapsed:.3f}s")

    @pytest.mark.parametrize("use_parallel", [False, True])
    def test_distance_matrix_large(self, large_sequences, use_parallel):
        """Test distance matrix calculation on large dataset."""
        mapper = TemporalMAP(
            use_parallel_distance_matrix=use_parallel,
        )

        start = time.perf_counter()
        dist_matrix = mapper.calculate_distance_matrix(large_sequences)
        elapsed = time.perf_counter() - start

        assert dist_matrix.shape[0] == sum(s.shape[0] for s in large_sequences)
        print(f"\nDistance matrix (parallel={use_parallel}): {elapsed:.3f}s")

    def test_distance_matrix_speedup(self, large_sequences):
        """Compare parallel vs sequential speedup."""
        # Sequential
        mapper_seq = TemporalMAP(use_parallel_distance_matrix=False)
        start = time.perf_counter()
        dist_seq = mapper_seq.calculate_distance_matrix(large_sequences)
        time_seq = time.perf_counter() - start

        # Parallel
        mapper_par = TemporalMAP(use_parallel_distance_matrix=True)
        start = time.perf_counter()
        dist_par = mapper_par.calculate_distance_matrix(large_sequences)
        time_par = time.perf_counter() - start

        speedup = time_seq / time_par
        print(f"\nSequential: {time_seq:.3f}s, Parallel: {time_par:.3f}s, Speedup: {speedup:.2f}x")

        # Verify results are similar
        np.testing.assert_allclose(dist_seq, dist_par, rtol=1e-5)


class TestProbabilityMatrixPerformance:
    """Test performance of probability matrix calculation."""

    @pytest.mark.parametrize("use_parallel", [False, True])
    def test_probability_matrix_medium(self, medium_sequences, use_parallel):
        """Test probability matrix calculation performance."""
        mapper = TemporalMAP(
            use_parallel_distance_matrix=use_parallel,
        )

        # Time the full fit including probability matrix calculation
        start = time.perf_counter()
        _ = mapper.fit(medium_sequences, max_iterations=10)
        elapsed = time.perf_counter() - start

        print(f"\nFull fit (parallel={use_parallel}): {elapsed:.3f}s")

    def test_probability_matrix_speedup(self, large_sequences):
        """Compare sequential vs parallel processing."""
        # Sequential version
        mapper_seq = TemporalMAP(
            use_parallel_distance_matrix=False,
        )
        start = time.perf_counter()
        _ = mapper_seq.fit(large_sequences, max_iterations=10)
        time_seq = time.perf_counter() - start

        # Parallel version
        mapper_par = TemporalMAP(
            use_parallel_distance_matrix=True,
        )
        start = time.perf_counter()
        _ = mapper_par.fit(large_sequences, max_iterations=10)
        time_par = time.perf_counter() - start

        speedup = time_seq / time_par
        print(f"\nSequential: {time_seq:.3f}s, Parallel: {time_par:.3f}s, Speedup: {speedup:.2f}x")


class TestFullPipelinePerformance:
    """Test full pipeline with different configurations."""

    @pytest.mark.parametrize("config", [
        {"use_parallel_distance_matrix": False},
        {"use_parallel_distance_matrix": True},
    ])
    def test_full_pipeline_medium(self, medium_sequences, config):
        """Test full pipeline with different optimization configurations."""
        embedding, elapsed = fit_tmap(medium_sequences, **config, max_iterations=100)

        assert embedding.shape[0] == sum(s.shape[0] for s in medium_sequences)
        assert embedding.shape[1] == 2

        config_str = f"parallel={config['use_parallel_distance_matrix']}"
        print(f"\nFull pipeline ({config_str}): {elapsed:.3f}s")

    def test_full_pipeline_comparison(self, large_sequences):
        """Compare all four configurations on large dataset."""
        configs = [
            ("Sequential", {"use_parallel_distance_matrix": False}),
            ("Parallel", {"use_parallel_distance_matrix": True}),
        ]

        results = {}
        for name, config in configs:
            _, elapsed = fit_tmap(large_sequences, **config, max_iterations=100)
            results[name] = elapsed
            print(f"\n{name}: {elapsed:.3f}s")

        # Calculate speedups relative to slowest
        baseline = max(results.values())
        print("\nSpeedups relative to baseline:")
        for name, elapsed in results.items():
            speedup = baseline / elapsed
            print(f"  {name}: {speedup:.2f}x")


class TestWorkerScaling:
    """Test scaling with different numbers of workers."""

    @pytest.mark.parametrize("max_workers", [1, 2, 4, None])
    def test_worker_scaling(self, large_sequences, max_workers):
        """Test performance with different worker counts."""
        mapper = TemporalMAP(
            use_parallel_distance_matrix=True,
            max_workers=max_workers,
        )

        start = time.perf_counter()
        _ = mapper.calculate_distance_matrix(large_sequences)
        elapsed = time.perf_counter() - start

        worker_str = f"workers={max_workers if max_workers else 'auto'}"
        print(f"\nDistance matrix ({worker_str}): {elapsed:.3f}s")


class TestAlignerPerformance:
    """Test performance with different alignment methods."""

    @pytest.mark.parametrize("aligner_cls,aligner_kwargs", [
        (DTWAlignment, {"window": 5}),
        (DTWAlignment, {"window": 10}),
        (DTWAlignment, {"window": None}),
    ])
    def test_dtw_windows(self, medium_sequences, aligner_cls, aligner_kwargs):
        """Test DTW performance with different window sizes."""
        aligner = aligner_cls(**aligner_kwargs)
        mapper = TemporalMAP(
            aligner=aligner,
            use_parallel_distance_matrix=True,
        )

        start = time.perf_counter()
        _ = mapper.calculate_distance_matrix(medium_sequences)
        elapsed = time.perf_counter() - start

        window_str = f"window={aligner_kwargs.get('window', 'None')}"
        print(f"\nDTW ({window_str}): {elapsed:.3f}s")


@pytest.mark.skipif(
    not Path("examples/data/trimmed_tracks.npy").exists(),
    reason="Real data not available"
)
class TestRealDataPerformance:
    """Test performance on real tracking data."""

    def test_real_data_comparison(self, real_data):
        """Compare configurations on real data."""
        if real_data is None:
            pytest.skip("Real data not available")

        configs = [
            ("Sequential", {"use_parallel_distance_matrix": False}),
            ("Parallel", {"use_parallel_distance_matrix": True}),
        ]

        for name, config in configs:
            _, elapsed = fit_tmap(
                real_data,
                **config,
                aligner=DTWAlignment(window=5),
                layout=SpectralLayout(),
                max_iterations=200
            )
            print(f"\nReal data {name}: {elapsed:.3f}s")
