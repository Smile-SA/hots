# hots/core/app.py

"""HOTS – application main process."""

import importlib
import logging
import time
from typing import Optional

from hots.config.loader import AppConfig
from hots.core.instance import Instance
from hots.evaluation.evaluator import (
    EvalSnapshot,
    eval_bilevel_step
)
from hots.plugins import (
    ClusteringFactory,
    ConnectorFactory,
    OptimizationFactory,
)
from hots.plugins.clustering.builder import (
    build_adjacency_matrix,
    build_matrix_indiv_attr,
    build_similarity_matrix
)
from hots.utils.signals import setup_signal_handlers

import numpy as np

import pandas as pd


class App:
    """Application entry point for HOTS."""

    def __init__(self, config: AppConfig):
        """Initialize the App with a given configuration."""
        self.config = config
        self.instance = Instance(config)

        # Factories
        self.connector = ConnectorFactory.create(config.connector, self.instance)
        self.clustering = ClusteringFactory.create(config.clustering, self.instance)
        self.clust_opt = OptimizationFactory.create(
            config.optimization, self.instance, pb_number=1
        )
        self.problem_opt = OptimizationFactory.create(
            config.optimization, self.instance, pb_number=2
        )
        # dynamically load the problem plugin (e.g. 'placement')
        problem_type = config.problem.type.lower()
        module_path = f'hots.plugins.problem.{problem_type}'
        cls_name = f'{problem_type.title()}Plugin'
        mod = importlib.import_module(module_path)
        problem_cls = getattr(mod, cls_name)
        self.problem = problem_cls(self.instance)

        # State that persists across iterations
        self.prev_snapshot: Optional[EvalSnapshot] = None
        self.metrics_history = []

        setup_signal_handlers(self.shutdown)

    # -------------------------
    # Main run loop
    # -------------------------
    def run(self):
        """Run the initial evaluation and streaming update loop."""
        t_start = time.time()
        logging.info('Starting HOTS run – preprocessing')

        # Clear any residual offsets/state
        self.instance.clear_kafka_topics()

        # ===== Initial window =====
        logging.info('Analysis period (initial window)')
        labels = np.asarray(self.clustering.fit(self.instance.df_indiv))
        n_clusters = int(len(np.unique(labels)))
        logging.info('Initial clustering produced %d clusters', n_clusters)

        if self.config.problem.parameters.get('initial_placement', True):
            logging.info('Running first placement heuristic')
            initial_moves = self.problem.initial(
                labels,
                self.instance.df_indiv,
                self.instance.df_host,
            )
            logging.info('First placement produced %d moves', len(initial_moves))
        else:
            logging.info('Skipping first placement (keeping existing)')
            initial_moves = []

        self.connector.apply_moves(initial_moves)
        logging.info('Applied initial placement moves')

        self.pre_loop(labels)

        # Placement uses clustering labels
        problem = self._solve_problem(labels)

        # Evaluate (no "previous" yet, so deltas will be None/0)
        snapshot, metrics = eval_bilevel_step(
            instance=self.instance,
            labels=labels,
            problem=problem,
            prev_snapshot=None,
            tick=self.instance.current_tick if hasattr(self.instance, 'current_tick') else 0,
        )
        self.prev_snapshot = snapshot
        self.metrics_history.append(metrics)
        self._persist_metrics()

        # ===== Streaming loop =====
        logging.info('Entering streaming update loop')
        for batch in self.connector.stream():  # yields new dataframes or payloads
            # 1) Ingest/merge new data into the instance
            self.instance.update_with_batch(batch)

            # 2) Re-run clustering on new state
            labels = np.asarray(self.clustering.fit(self.instance.df_indiv))
            n_clusters = int(len(np.unique(labels)))
            logging.info('Re-clustered: %d clusters', n_clusters)

            # 3) Re-solve problem using clustering output
            problem = self._solve_problem(labels)

            # 4) Evaluate bi-level (compare to previous)
            snapshot, metrics = eval_bilevel_step(
                instance=self.instance,
                labels=labels,
                problem=problem,
                prev_snapshot=self.prev_snapshot,
                tick=self.instance.current_tick if hasattr(self.instance, 'current_tick') else 0,
            )
            self.prev_snapshot = snapshot
            self.metrics_history.append(metrics)
            self._persist_metrics()

        t_total = time.time() - t_start
        logging.info('Finished HOTS run in %.3f seconds', t_total)

    def pre_loop(self, labels):
        """Build and solve optimization models before performing streaming loop."""
        logging.info('Building first clustering model...')
        mat = build_matrix_indiv_attr(
            self.instance.df_indiv,
            self.instance.config.tick_field,
            self.instance.config.individual_field,
            self.instance.config.metrics,
            self.instance.get_id_map()
        )
        u_mat = build_adjacency_matrix(labels)
        w_mat = build_similarity_matrix(mat)
        self.clust_opt.build(u_mat=u_mat, w_mat=w_mat)
        self.clust_opt.solve(
            solver=self.config.optimization.parameters.get('solver', 'glpk'),
        )

        # 3) Extract dual values
        clust_dual = self.clust_opt.fill_dual_values()
        print(clust_dual)
        input('dual')

    # -------------------------
    # Helpers
    # -------------------------
    def _solve_problem(self, labels):
        """Solve the problem problem given clustering labels."""
        if hasattr(self.problem_opt, 'solve'):
            return self.problem_opt.solve(labels=labels, instance=self.instance)

        raise RuntimeError(
            'Placement optimizer must implement .solve(labels=..., instance=...).'
        )

    def _persist_metrics(self):
        """Append metrics to CSV and keep an in-memory history."""
        out = pd.DataFrame(self.metrics_history)
        out.to_csv(self.instance.results_file, index=False, mode='w')
        logging.debug('Metrics written to %s', self.instance.results_file)

    def shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print('Shutting down...')
        exit(0)
