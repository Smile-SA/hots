"""End-to-end 'happy path' test using tests/data/thesis_ex_10."""

import importlib.util
from pathlib import Path

from hots.config.loader import load_config
from hots.core.app import App

from pyomo.opt import SolverFactory

import pytest

# Skip if Pyomo is not installed, since the app depends on it
_pyomo_spec = importlib.util.find_spec('pyomo.environ')
_pyomo_installed = _pyomo_spec is not None
pytestmark = pytest.mark.skipif(
    not _pyomo_installed, reason='pyomo.environ (Pyomo) not installed'
)


@pytest.mark.e2e
def test_thesis_ex_10_happy_path(tests_data_directory, tmp_path):
    """
    Load the thesis_ex_10 dataset, run a short HOTS run, and assert:
    - metrics_history is non-empty
    - metrics CSV is written
    - moves log is written and non-empty
    """
    # Path to a base JSON config
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / 'hots' / 'config' / 'sample_config_file.json'

    # Use your real loader
    cfg = load_config(config_path)

    solver_name = getattr(cfg, 'solver', 'glpk')
    solver = SolverFactory(solver_name)
    if not solver.available(False):
        pytest.skip(f"Solver '{solver_name}' not available on this runner")

    # Point the connector to the tiny test dataset
    data_dir = tests_data_directory / 'thesis_ex_10'
    params = cfg.connector.parameters

    params['data_folder'] = str(data_dir)
    # These defaults match your existing test data schema
    params.setdefault('file_name', 'container_usage.csv')
    params.setdefault('tick_field', 'timestamp')
    params.setdefault('individual_field', 'container_id')
    params.setdefault('host_field', 'machine_id')
    params.setdefault('metrics', ['cpu'])
    params.setdefault('sep_time', 3)
    params.setdefault('tick_increment', 1)
    # Output moves file into a tmp folder so tests are self-contained
    params['outfile'] = str(tmp_path / 'moves.jsonl')
    params['reset_on_start'] = True

    # Redirect reporting outputs into tmp_path as well
    cfg.reporting.results_folder = tmp_path
    cfg.reporting.metrics_file = tmp_path / 'metrics.csv'
    cfg.reporting.plots_folder = tmp_path / 'plots'

    # Optional: keep time_limit None so the run stops when data is exhausted
    cfg.time_limit = None

    app = App(cfg)
    app.run()  # should complete on the thesis_ex_10 dataset

    # 1) In-memory metrics
    assert app.instance.metrics_history, 'metrics_history should not be empty after a run'

    # 2) Metrics CSV exists and is non-empty
    metrics_path = cfg.reporting.metrics_file
    assert metrics_path.exists()
    assert metrics_path.stat().st_size > 0

    # 3) Moves file exists (and ideally non-empty)
    moves_path = Path(params['outfile'])
    assert moves_path.exists()
    assert moves_path.stat().st_size > 0
