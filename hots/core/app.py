import signal
from config.loader import AppConfig
from core.instance import Instance
from utils.signals import setup_signal_handlers
from plugins import (
    ClusteringFactory,
    OptimizationFactory,
    HeuristicFactory,
    ConnectorFactory
)
from evaluation.evaluator import eval_solutions
from reporting.writer import write_metrics

class App:
    def __init__(self, config: AppConfig):
        self.config = config
        self.instance = Instance(config)
        self.clustering = ClusteringFactory.create(config.clustering, self.instance)
        self.optimization = OptimizationFactory.create(config.optimization, self.instance)
        self.heuristic = HeuristicFactory.create(config.heuristic, self.instance)
        self.connector = ConnectorFactory.create(config.connector, self.instance)
        setup_signal_handlers(self.shutdown)

    def run(self):
        # clear any residual offsets/state
        self.instance.clear_kafka_topics()

        # ---- initial evaluation ----
        sol2, metrics = eval_solutions(
            self.instance.df_indiv,
            self.instance.df_host,
            self.clustering.fit(self.instance.df_indiv),
            self.clustering,
            self.optimization,
            self.heuristic,
            self.instance
        )
        self.connector.apply_moves(sol2)
        self.instance.metrics_history.append(metrics)

        # ---- streaming loop ----
        while True:
            df_new = self.instance.reader.load_next()
            if df_new is None:
                break

            # update ingestion state
            self.instance.update_data(df_new)

            # evaluate new solution  metrics
            sol2, metrics = eval_solutions(
                self.instance.df_indiv,
                self.instance.df_host,
                self.clustering.fit(self.instance.df_indiv),
                self.clustering,
                self.optimization,
                self.heuristic,
                self.instance
            )
            self.connector.apply_moves(sol2)
            self.instance.metrics_history.append(metrics)
        
        for m in self.instance.metrics_history:
            write_metrics(m, self.instance.results_file)

    def shutdown(self, signum, frame):
        print('Shutting down...')
        exit(0)
