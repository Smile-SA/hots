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
        self.tick_field = parameters['tick_field']
        self.indiv_field = parameters['individual_field']
        self.host_field = parameters['host_field']
        self.metrics = parameters['metrics']
        self.tick_increment = parameters.get('tick_increment', 1)

        self.csv_file = open(self.csv_path)
        self.csv_reader = csv.reader(self.csv_file)
        next(self.csv_reader, None)

        self.queue = queue.Queue()
        self.current_time = 0

    def load_initial(self) -> pd.DataFrame:
        """Load the first batch of rows from the CSV."""
        df = self._get_batch()
        self.current_time += self.tick_increment
        return df, None, None

    def load_next(self) -> pd.DataFrame:
        """Load the next time‐window batch of rows from the CSV."""
        df = self._get_batch()
        if df.empty:
            return None
        self.current_time += self.tick_increment
        return df

    def _get_batch(self) -> pd.DataFrame:
        """Collect rows up to the current time window."""
        rows = []
        while True:
            if self.queue.empty():
                row = next(self.csv_reader, None)
                if row is None:
                    break
                self.queue.put(row)

            ts = int(self.queue.queue[0][0])
            if ts <= self.current_time + self.tick_increment:
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
