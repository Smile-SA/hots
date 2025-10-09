"""Performance benchmark tests for basic operations."""

import importlib.util

import pytest

# Skip the whole module if pytest-benchmark isn't installed
_has_bench = importlib.util.find_spec('pytest_benchmark') is not None
pytestmark = pytest.mark.skipif(not _has_bench, reason='pytest-benchmark not installed')


def test_perf_example(benchmark):
    """Benchmark a simple data transformation to validate performance test setup."""
    data = list(range(1000))

    def run():
        """Double each element in the list (dummy workload)."""
        return [x * 2 for x in data]

    result = benchmark(run)
    assert len(result) == 1000
