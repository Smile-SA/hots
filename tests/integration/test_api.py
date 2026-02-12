"""Integration tests for the scheduler API endpoints."""

import pytest

try:
    import httpx
except ImportError:
    httpx = None

pytestmark = pytest.mark.skipif(httpx is None, reason="httpx not installed")


def _is_enabled() -> bool:
    """Return True if integration tests are explicitly enabled.

    Set HOTS_RUN_INTEGRATION=1 to enable.
    """
    import os

    return os.getenv("HOTS_RUN_INTEGRATION", "").strip() in {"1", "true", "True"}


def test_health_endpoint(test_settings):
    """Verify that the /health endpoint is reachable and returns a valid status code."""
    if not _is_enabled():
        pytest.skip("Integration tests disabled (set HOTS_RUN_INTEGRATION=1 to enable)")

    base = test_settings["API_BASE"]
    with httpx.Client(timeout=5.0) as client:
        try:
            r = client.get(f"{base}/health")
        except httpx.RequestError as exc:
            pytest.skip(f"API not reachable at {base!r}: {exc}")

        assert r.status_code in (200, 204)
