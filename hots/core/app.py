# hots/core/app.py

"""HOTS – application main process."""

import importlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hots.config.loader import AppConfig
from hots.core.instance import Instance
from hots.evaluation.evaluator import EvalSnapshot, eval_solutions
from hots.plugins import (
    ClusteringFactory,
    ConnectorFactory,
    OptimizationFactory,
)
from hots.plugins.clustering.builder import (
    build_post_clust_matrices,
    build_pre_clust_matrices,
)
from hots.utils.signals import setup_signal_handlers
from hots.visualization.plot import plot_containers_groupby_nodes


class App:
    """Application entry point for HOTS."""

    def __init__(self, config: AppConfig):
        """Initialize the App with a given configuration."""
        self.config = config
        self.instance = Instance(config)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")

        # Factories
        self.connector = ConnectorFactory.create(config.connector, self.instance)
        self.instance._load_initial_data(self.connector)
        self.clustering = ClusteringFactory.create(config.clustering, self.instance)
        self.clust_opt = OptimizationFactory.create(config.optimization, self.instance, pb_number=1)
        self.problem_opt = OptimizationFactory.create(
            config.optimization, self.instance, pb_number=2
        )
        # dynamically load the problem plugin (e.g. 'placement')
        problem_type = config.problem.type.lower()
        module_path = f"hots.plugins.problem.{problem_type}"
        cls_name = f"{problem_type.title()}Plugin"
        mod = importlib.import_module(module_path)
        problem_cls = getattr(mod, cls_name)
        self.problem = problem_cls(self.instance)

        # State that persists across iterations
        self.prev_snapshot: EvalSnapshot | None = None
        self.results_file: Path = config.reporting.metrics_file

        setup_signal_handlers(self.shutdown)

    # -------------------------
    # Main run loop
    # -------------------------
    def run(self):
        """Run the initial evaluation and streaming update loop."""
        t_start = time.time()
        end_time = 0
        if self.config.time_limit is None:
            end_time = float("inf")
        else:
            end_time = t_start + self.config.time_limit
        logging.info("Starting HOTS run – preprocessing")

        if self.config.connector.type.lower() == "kafka":
            # Clear any residual offsets/state
            self.instance.clear_kafka_topics()

        # Initial window
        n_clusters, n_initial_moves = self._initial_phase()

        self.pre_loop()

        # record a minimal “initial” metric
        self.instance.metrics_history.append(
            {
                "initial_clusters": n_clusters,
                "initial_moves": n_initial_moves,
            }
        )

        # Streaming loop
        self._streaming_loop(end_time=end_time)

        self._finalize_run_kpis()
        self._persist_metrics()
        t_total = time.time() - t_start
        logging.info("Finished HOTS run in %.3f seconds", t_total)

    def _initial_phase(self) -> tuple[int, int]:
        """Run initial clustering + optional first placement."""
        logging.info("Starting HOTS run – preprocessing")

        # Clear any residual offsets/state
        self.instance.clear_kafka_topics()

        # ===== Initial window =====
        logging.info("Analysis period (initial window)")
        labels = np.asarray(self.clustering.fit(self.instance.df_indiv))
        n_clusters = int(len(np.unique(labels)))
        logging.info("Initial clustering produced %d clusters", n_clusters)

        if self.config.problem.parameters.get("initial_placement", True):
            logging.info("Running first placement heuristic")
            initial_moves = self.problem.initial(
                labels,
                self.instance.df_indiv,
                self.instance.df_host,
            )
            self.connector.apply_moves(
                initial_moves, self.config.connector.parameters.get("sep_time")
            )
            logging.info("Applied initial placement moves")
            logging.info("First placement produced %d moves", len(initial_moves))
            n_initial_moves = len(initial_moves)
        else:
            logging.info("Skipping first placement (keeping existing)")
            n_initial_moves = 0

        _, _ = plot_containers_groupby_nodes(
            self.instance.df_indiv,
            viz=self.config.viz,
            max_cap=self.instance.df_meta["cpu"].max(),
            sep_time=self.config.connector.parameters.get("sep_time"),
            fields=self.instance.fields,
        )

        return n_clusters, n_initial_moves

    def pre_loop(self):
        """Build and solve optimization models before performing streaming loop."""
        logging.info("Building first optimization models...")
        cfg = self.instance.config.connector.parameters
        # Clustering model
        build_pre_clust_matrices(
            self.instance.df_indiv,
            cfg.get("tick_field"),
            cfg.get("individual_field"),
            cfg.get("metrics"),
            self.instance.get_id_map(),
            self.clustering,
        )

        self.clust_opt.build(u_mat=self.clustering.u_mat, w_mat=self.clustering.w_mat)
        self.clust_opt.solve()
        self.clust_opt.fill_dual_values()
        # Problem model
        v_mat = self.problem.build_place_adj_matrix(
            self.instance.df_indiv, self.instance.get_id_map()
        )
        dv_mat = build_post_clust_matrices(self.clustering.clust_mat)
        self.problem_opt.build(u_mat=self.clustering.u_mat, v_mat=v_mat, dv_mat=dv_mat)
        self.problem_opt.solve()
        self.problem_opt.fill_dual_values()

    def _streaming_loop(self, *, end_time: float) -> None:
        """Streaming update loop."""
        tick_field = self.config.connector.parameters.get("tick_field")
        window_duration = self.config.connector.parameters.get("window_duration")

        # Initial streaming bounds are derived from the last tick in the initial df
        tmax = self.instance.df_indiv[tick_field].max()
        tmax += self.connector.tick_increment
        tmin = tmax - (window_duration - 1)

        loop_nb = 1
        while True:
            if time.time() >= end_time:
                break

            logging.info("Starting loop #%d", loop_nb)

            df_new = self.connector.load_next()
            if df_new is None:
                break

            # Update ingestion state
            self.instance.update_data(df_new)
            working_df = self.instance.get_working_df(tmin, tmax)

            # Evaluate new solution + metrics
            moves, metrics, self.prev_snapshot = eval_solutions(
                self.instance,
                self.clustering,
                self.clust_opt,
                self.problem_opt,
                self.problem,
                working_df,
                prev_snapshot=self.prev_snapshot,
            )
            metrics.update(
                {
                    "run_id": self.run_id,
                    "loop_nb": loop_nb,
                    "tick_start": tmin,
                    "tick_end": tmax,
                    "window_duration": window_duration,
                }
            )
            self.instance.metrics_history.append(metrics)

            logging.info(
                "Loop %d moves: %d containers",
                loop_nb,
                len(metrics["moving_containers"]),
            )

            self.connector.apply_moves(moves, tmax)
            self.instance.metrics_history.append(metrics)

            # Advance window
            tmax += self.connector.tick_increment
            tmin = tmax - (window_duration - 1)
            loop_nb += 1

    def _finalize_run_kpis(self) -> None:
        df = pd.DataFrame(self.instance.metrics_history)
        if df.empty:
            return

        run_kpis = {
            "run_id": self.run_id,
            "total_steps": int(len(df)),
            "total_moves": float(
                df.get("moves__moves_count", pd.Series(dtype=float)).fillna(0).sum()
            )
            if "moves__moves_count" in df.columns
            else 0.0,
            "avg_moved_ratio": float(df.get("plc__moved_ratio", pd.Series(dtype=float)).mean())
            if "plc__moved_ratio" in df.columns
            else 0.0,
            "max_conf_density": float(df.get("place_conf__density", pd.Series(dtype=float)).max())
            if "place_conf__density" in df.columns
            else 0.0,
        }
        self.instance.run_kpis = run_kpis

    def _persist_metrics(self):
        out = pd.DataFrame(self.instance.metrics_history)
        if out.empty:
            return

        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.results_file, index=False)

        # Try parquet
        parquet_path = self.results_file.with_suffix(".parquet")
        try:
            out.to_parquet(parquet_path, index=False)
            logging.info("Step metrics written to %s", parquet_path)
        except ImportError as e:
            logging.info(
                "Parquet support not available (%s). Skipping parquet export, CSV written to %s",
                e.__class__.__name__,
                self.results_file,
            )

        # Run-level KPIs
        if self.instance.run_kpis:
            run_path = self.results_file.with_name(self.results_file.stem + "_run.json")
            pd.Series(self.instance.run_kpis).to_json(run_path)
            logging.info("Run KPIs written to %s", run_path)

    def shutdown(self, signum=None, frame=None):
        """Handle shutdown signals gracefully."""
        logging.info("Shutting down...")
        try:
            self._persist_metrics()
        finally:
            raise SystemExit(0)
