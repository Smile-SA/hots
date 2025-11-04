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
    eval_bilevel_step,
    eval_solutions
)
from hots.plugins import (
    ClusteringFactory,
    ConnectorFactory,
    OptimizationFactory,
)
from hots.plugins.clustering.builder import (
    build_post_clust_matrices,
    build_pre_clust_matrices
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
        end_time = 0
        if self.config.time_limit is not None:
            end_time = t_start + self.config.time_limit
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

        # TODO not really
        self.connector.apply_moves(initial_moves)
        logging.info('Applied initial placement moves')

        self.pre_loop(labels)

        # Evaluate (no "previous" yet, so deltas will be None/0)
        # snapshot, metrics = eval_bilevel_step(
        #     instance=self.instance,
        #     labels=labels,
        #     problem=problem,
        #     prev_snapshot=None,
        #     tick=self.instance.current_tick if hasattr(self.instance, 'current_tick') else 0,
        # )
        # self.prev_snapshot = snapshot
        # self.metrics_history.append(metrics)
        # self._persist_metrics()

        # record a minimal “initial” metric
        self.instance.metrics_history.append({
            'initial_clusters': n_clusters,
            'initial_moves': len(initial_moves),
        })

        # Streaming loop
        tmin = self.instance.df_indiv[self.config.tick_field].min()
        tmax = self.instance.df_indiv[self.config.tick_field].max()
        loop_nb = 1
        while True:
            if self.config.time_limit is not None and time.time() >= end_time:
                self.shutdown()
                break
            logging.info('Starting loop #%d', loop_nb)
            df_new = self.instance.reader.load_next()
            if df_new is None:
                break

            # Update ingestion state
            self.instance.update_data(df_new)
            working_df = self.instance.get_working_df(tmin, tmax)

            # Evaluate new solution + metrics
            sol2, metrics = eval_solutions(
                self.instance,
                self.clustering,
                self.clust_opt,
                self.problem_opt,
                self.problem,
                working_df
            )
            logging.info(
                'Loop %d moves: %d containers', loop_nb, len(metrics['moving_containers'])
            )

            logging.info('Applying moves for loop #%d', loop_nb)
            self.connector.apply_moves(sol2)
            logging.info('Moves applied for loop #%d', loop_nb)

            self.instance.metrics_history.append(metrics)

            tmax += self.config.reader.parameters.get('tick_increment', 2)
            tmin = tmax - (self.config.reader.parameters.get('tick_increment', 2) - 1)
            loop_nb += 1


        # ===== Streaming loop =====
        logging.info('Entering streaming update loop')
        for batch in self.connector.stream():  # yields new dataframes or payloads
            # 1) Ingest/merge new data into the instance
            print(self.instance.df_indiv)
            self.instance.update_with_batch(batch)
            print(self.instance.df_indiv)
            input()

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
        logging.info('Building first optimization models...')
        # Clustering model
        (clust_mat, u_mat, w_mat) = build_pre_clust_matrices(
            self.instance.df_indiv,
            self.instance.config.tick_field,
            self.instance.config.individual_field,
            self.instance.config.metrics,
            self.instance.get_id_map(),
            labels
        )

        self.clust_opt.build(u_mat=u_mat, w_mat=w_mat)
        self.clust_opt.solve()
        clust_duals = self.clust_opt.fill_dual_values()
        print('clust_dual', clust_duals)
        # Problem model
        v_mat = self.problem.build_place_adj_matrix(
            self.instance.df_indiv,
            self.instance.get_id_map())
        dv_mat = build_post_clust_matrices(clust_mat)
        self.problem_opt.build(u_mat=u_mat, v_mat=v_mat, dv_mat=dv_mat)
        self.problem_opt.solve()
        problem_duals = self.problem_opt.fill_dual_values()
        print('place_dual', problem_duals)

    def _persist_metrics(self):
        """Append metrics to CSV and keep an in-memory history."""
        out = pd.DataFrame(self.metrics_history)
        out.to_csv(self.instance.results_file, index=False, mode='w')
        logging.debug('Metrics written to %s', self.instance.results_file)

    def shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print('Shutting down...')
        exit(0)
