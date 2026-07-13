"""Unit tests for App graceful shutdown (flag instead of SystemExit)."""

from hots.core.app import App


def _minimal_app(tmp_path):
    """Build an App shell without running the full pipeline."""
    app = App.__new__(App)
    app._stop_requested = False
    app.run_id = "test-run"
    app.results_file = tmp_path / "metrics.csv"

    class _Instance:
        metrics_history = [{"loop_nb": 1, "moves__moves_count": 2.0}]
        run_kpis = None

    app.instance = _Instance()
    return app


def test_shutdown_sets_flag_and_does_not_raise(tmp_path):
    app = _minimal_app(tmp_path)

    # Must not raise SystemExit so HOTS can be embedded as a library
    app.shutdown(signum=2, frame=None)

    assert app._stop_requested is True
    # Metrics collected so far are still flushed
    assert app.results_file.exists()
    assert app.results_file.stat().st_size > 0
