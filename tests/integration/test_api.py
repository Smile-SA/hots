"""Integration tests for the scheduler API endpoints."""

import pytest
try:
    import httpx
except ImportError:
    httpx = None

pytestmark = pytest.mark.skipif(httpx is None, reason='httpx not installed')


def test_health_endpoint(test_settings):
    """Verify that the /health endpoint is reachable and returns a valid status code."""
    base = test_settings['API_BASE']
    with httpx.Client(timeout=5.0) as client:
        r = client.get(f'{base}/health')
        assert r.status_code in (200, 204)
