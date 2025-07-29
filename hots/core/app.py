# hots/core/app.py

"""HOTS."""

import importlib
import logging
import time

from hots.config.loader import AppConfig
from hots.core.instance import Instance
from hots.evaluation.evaluator import eval_solutions
from hots.plugins import (
    ClusteringFactory,
    ConnectorFactory,
    OptimizationFactory,
)
from hots.reporting.writer import write_metrics
from hots.utils.signals import setup_signal_handlers


class App:
    """Application entry point for HOTS."""

    def __init__(self, config: AppConfig):
        """Initialize the App with a given configuration."""
        self.config = config
        self.instance = Instance(config)
        self.clustering = ClusteringFactory.create(
            config.clustering,
            self.instance,
        )
        self.optimization = OptimizationFactory.create(
            config.optimization,
            self.instance,
        )
        self.connector = ConnectorFactory.create(
            config.connector,
            self.instance,
        )
        # dynamically load the problem plugin (e.g. 'placement')
        problem_type = config.problem.type.lower()
        module_path = f'hots.plugins.problem.{problem_type}'
        cls_name = f'{problem_type.title()}Plugin'
        mod = importlib.import_module(module_path)
        problem_cls = getattr(mod, cls_name)
        self.problem = problem_cls(config.problem.parameters, self.instance)

        setup_signal_handlers(self.shutdown)

    def run(self):
        """Run the initial evaluation and streaming update loop."""
        t_start = time.time()
        logging.info('Starting HOTS run – preprocessing')

        # Clear any residual offsets/state
        self.instance.clear_kafka_topics()

        logging.info('Running initial evaluation')
        # Initial evaluation
        sol2, metrics = eval_solutions(
            self.instance.df_indiv,
            self.instance.df_host,
            self.clustering.fit(self.instance.df_indiv),
            self.clustering,
            self.optimization,
            self.problem,
            self.instance,
        )
        logging.info('Initial silhouette: %.4f', metrics['silhouette'])
        logging.info('Initial moves: %d containers', len(metrics['moving_containers']))

        logging.info('Applying initial moves')
        self.connector.apply_moves(sol2)
        logging.info('Initial moves applied')

        self.instance.metrics_history.append(metrics)

        # Streaming loop
        loop_nb = 1
        while True:
            logging.info('Starting loop #%d', loop_nb)
            df_new = self.instance.reader.load_next()
            if df_new is None:
                break

            # Update ingestion state
            self.instance.update_data(df_new)

            # Evaluate new solution + metrics
            sol2, metrics = eval_solutions(
                self.instance.df_indiv,
                self.instance.df_host,
                self.clustering.fit(self.instance.df_indiv),
                self.clustering,
                self.optimization,
                self.problem,
                self.instance,
            )
            logging.info(
                'Loop %d moves: %d containers', loop_nb, len(metrics['moving_containers'])
            )

            logging.info('Applying moves for loop #%d', loop_nb)
            self.connector.apply_moves(sol2)
            logging.info('Moves applied for loop #%d', loop_nb)

            self.instance.metrics_history.append(metrics)

            loop_nb += 1

        logging.info(
            'Writing out metrics history (%d records)',
            len(self.instance.metrics_history)
        )
        for m in self.instance.metrics_history:
            write_metrics(m, self.instance.results_file)

        t_total = time.time() - t_start
        logging.info('Finished HOTS run in %.3f seconds', t_total)

    def shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print('Shutting down...')
        exit(0)
