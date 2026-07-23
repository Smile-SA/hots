"""End-to-end 'happy path' test using tests/data/thesis_ex_10."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hots.config.loader import load_config
from hots.core.app import App

# Skip if Pyomo is not installed, since the app depends on it
_pyomo_spec = importlib.util.find_spec("pyomo.environ")
_pyomo_installed = _pyomo_spec is not None
pytestmark = pytest.mark.skipif(not _pyomo_installed, reason="pyomo.environ (Pyomo) not installed")

if _pyomo_installed:
    # Registers the solver plugins; without it SolverFactory('glpk')
    # returns an UnknownSolver and the availability check always skips.
    import pyomo.environ  # noqa: F401
    from pyomo.opt import SolverFactory


def _prepared_config(tests_data_directory, tmp_path):
    """Load the sample config and point it at the tiny test dataset."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "hots" / "config" / "sample_config_file.json")

    solver_name = getattr(cfg, "solver", "glpk")
    if not SolverFactory(solver_name).available(False):
        pytest.skip(f"Solver '{solver_name}' not available on this runner")

    params = cfg.connector.parameters
    params["data_folder"] = str(tests_data_directory / "thesis_ex_10")
    params.setdefault("file_name", "container_usage.csv")
    params.setdefault("sep_time", 3)
    params.setdefault("tick_increment", 1)
    params["outfile"] = str(tmp_path / "moves.jsonl")
    params["reset_on_start"] = True

    cfg.reporting.results_folder = tmp_path
    cfg.reporting.metrics_file = tmp_path / "metrics.csv"
    cfg.reporting.plots_folder = tmp_path / "plots"
    cfg.time_limit = None
    return cfg


@pytest.mark.e2e
def test_initial_placement_path(tests_data_directory, tmp_path):
    """Run with the first-placement heuristic enabled.

    This path is disabled in the sample configs, so without this test nothing
    exercises PlacementPlugin.initial -> allocation_distant_pairwise, and a
    regression there is invisible.
    """
    cfg = _prepared_config(tests_data_directory, tmp_path)
    cfg.problem.parameters["initial_placement"] = True

    app = App(cfg)
    app.run()

    assert app.instance.metrics_history

    initial = next(m for m in app.instance.metrics_history if "initial_moves" in m)
    assert initial["initial_clusters"] > 0

    moves_path = Path(cfg.connector.parameters["outfile"])
    assert moves_path.exists()

    written = [json.loads(line) for line in moves_path.read_text().splitlines() if line.strip()]
    assert written, "the first placement should have produced moves"

    # Every emitted move must satisfy the connector contract, and a container
    # must appear at most once in the initial placement (one row per container,
    # not one per tick).
    initial_moves = written[: int(initial["initial_moves"])]
    for move in initial_moves:
        assert {"container_name", "old_host", "new_host"} <= set(move)

    moved = [m["container_name"] for m in initial_moves]
    assert len(moved) == len(set(moved)), "a container was placed more than once"


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
    config_path = repo_root / "hots" / "config" / "sample_config_file.json"

    # Use your real loader
    cfg = load_config(config_path)

    solver_name = getattr(cfg, "solver", "glpk")
    solver = SolverFactory(solver_name)
    if not solver.available(False):
        pytest.skip(f"Solver '{solver_name}' not available on this runner")

    # Point the connector to the tiny test dataset
    data_dir = tests_data_directory / "thesis_ex_10"
    params = cfg.connector.parameters

    params["data_folder"] = str(data_dir)
    # These defaults match your existing test data schema
    params.setdefault("file_name", "container_usage.csv")
    params.setdefault("tick_field", "timestamp")
    params.setdefault("individual_field", "container_id")
    params.setdefault("host_field", "machine_id")
    params.setdefault("metrics", ["cpu"])
    params.setdefault("sep_time", 3)
    params.setdefault("tick_increment", 1)
    # Output moves file into a tmp folder so tests are self-contained
    params["outfile"] = str(tmp_path / "moves.jsonl")
    params["reset_on_start"] = True

    # Redirect reporting outputs into tmp_path as well
    cfg.reporting.results_folder = tmp_path
    cfg.reporting.metrics_file = tmp_path / "metrics.csv"
    cfg.reporting.plots_folder = tmp_path / "plots"

    # Optional: keep time_limit None so the run stops when data is exhausted
    cfg.time_limit = None

    app = App(cfg)
    app.run()  # should complete on the thesis_ex_10 dataset

    # 1) In-memory metrics
    assert app.instance.metrics_history, "metrics_history should not be empty after a run"

    # Exactly one metrics row per streaming loop (regression: rows were appended twice)
    loop_rows = [m["loop_nb"] for m in app.instance.metrics_history if "loop_nb" in m]
    assert loop_rows, "streaming loops should have recorded metrics"
    assert len(loop_rows) == len(set(loop_rows)), (
        "each streaming loop must append exactly one metrics row"
    )

    # 2) Metrics CSV exists and is non-empty
    metrics_path = cfg.reporting.metrics_file
    assert metrics_path.exists()
    assert metrics_path.stat().st_size > 0

    # 3) Moves file exists (and ideally non-empty)
    moves_path = Path(params["outfile"])
    assert moves_path.exists()
    assert moves_path.stat().st_size > 0


@pytest.mark.e2e
def test_streaming_loop_reuses_the_persistent_model(tests_data_directory, tmp_path):
    """On no-new-container ticks, the loop must mutate the persistent Pyomo
    model in place rather than rebuild it from scratch every time.

    Regression guard for evaluator.eval_solutions: PyomoModel.build() should
    only be called during pre_loop (once per pb_number), never again inside
    the streaming loop as long as the container set doesn't change; the
    incremental update methods should be used instead.
    """
    from hots.plugins.optimization.pyomo_model import PyomoModel

    cfg = _prepared_config(tests_data_directory, tmp_path)

    with (
        patch.object(PyomoModel, "build", autospec=True, side_effect=PyomoModel.build) as build,
        patch.object(
            PyomoModel, "update_w", autospec=True, side_effect=PyomoModel.update_w
        ) as update_w,
        patch.object(
            PyomoModel,
            "update_adjacency_constraints",
            autospec=True,
            side_effect=PyomoModel.update_adjacency_constraints,
        ) as update_adjacency_constraints,
    ):
        app = App(cfg)
        app.run()

    loop_rows = [m["loop_nb"] for m in app.instance.metrics_history if "loop_nb" in m]
    assert loop_rows, "the toy dataset should exercise at least one streaming loop"

    # pre_loop() builds both models exactly once each; no new containers appear
    # in this dataset, so the streaming loop must never call build() again.
    assert build.call_count == 2, "build() should only run during pre_loop"
    assert update_w.call_count > 0, "the clustering model should be updated in place"
    assert update_adjacency_constraints.call_count > 0
