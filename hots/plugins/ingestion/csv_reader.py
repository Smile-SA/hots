"""CSV‑based ingestion plugin for HOTS."""

import csv
import queue
from pathlib import Path

from hots.core.interfaces import IngestionPlugin

import pandas as pd


class CSVReader(IngestionPlugin):
    """Ingestion plugin that reads data from a CSV file."""

    def __init__(self, parameters: dict, instance):
        """Initialize reader with file path and field mappings."""
        fname = parameters.get('file_name', 'container_usage.csv')
        self.csv_path = Path(parameters['data_folder']) / fname
        self.meta_path = Path(parameters['data_folder']) / 'node_meta.csv'
        self.tick_field = parameters['tick_field']
        self.indiv_field = parameters['individual_field']
        self.host_field = parameters['host_field']
        self.metrics = parameters['metrics']
        self.sep_time = parameters.get('sep_time', 5)
        self.tick_increment = parameters.get('tick_increment', 1)

        self.csv_file = open(self.csv_path)
        self.csv_reader = csv.reader(self.csv_file)
        next(self.csv_reader, None)

        self.queue = queue.Queue()
        self.current_time = 0

    def load_initial(self):
        """Load the first batch of rows from [0, sep_time]."""
        df_indiv = self._get_batch(window_end=self.sep_time)
        self.current_time = self.sep_time
        df_meta = pd.read_csv(self.meta_path, index_col=False)
        return df_indiv, None, df_meta

    def load_next(self):
        """Load the next time‐window batch of rows of size tick_increment."""
        next_window_end = self.current_time + self.tick_increment
        df = self._get_batch(window_end=next_window_end)
        if df.empty:
            return None
        self.current_time = next_window_end
        return df

    def _get_batch(self, window_end: int) -> pd.DataFrame:
        """
        Collect rows with timestamp <= window_end.

        Uses an internal queue as a small buffer for the next unread CSV row.
        Assumes the CSV is sorted by timestamp in ascending order.
        """
        rows = []
        while True:
            if self.queue.empty():
                row = next(self.csv_reader, None)
                if row is None:
                    break
                self.queue.put(row)

            # Peek at next row's timestamp (first column in the CSV)
            ts = int(self.queue.queue[0][0])

            # Include rows up to the provided window end (inclusive)
            if ts <= window_end:
                row = self.queue.get()
                rows.append({
                    self.tick_field: ts,
                    self.indiv_field: row[1],
                    self.host_field: row[2],
                    self.metrics[0]: float(row[3]),
                })
            else:
                break

        return pd.DataFrame(rows)

    def close(self) -> None:
        """Close the underlying CSV file."""
        self.csv_file.close()
