# hots/plugins/connector/file_connector.py

"""File‑based connector plugin for HOTS."""

import csv
import json
import queue
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from hots.core.interfaces import ConnectorPlugin

import pandas as pd


class FileConnector(ConnectorPlugin):
    """Connector plugin that read CSV data and writes moves to a JSONL file."""

    # process-level guard so we only truncate once per path per run
    _initialized_paths: set[str] = set()

    def __init__(self, params):
        """Initialize with output file path."""
        fname = params.get('file_name', 'container_usage.csv')
        self.csv_path = Path(params['data_folder']) / fname
        self.meta_path = Path(params['data_folder']) / 'node_meta.csv'
        self.tick_field = params['tick_field']
        self.indiv_field = params['individual_field']
        self.host_field = params['host_field']
        self.metrics = params['metrics']
        self.sep_time = params.get('sep_time', 5)
        self.tick_increment = params.get('tick_increment', 1)

        self.csv_file = open(self.csv_path)
        self.csv_reader = csv.reader(self.csv_file)
        next(self.csv_reader, None)

        self.queue = queue.Queue()
        self.current_time = 0
        # container_id -> sorted list of (move_time, target_node)
        self._moves: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        self.outfile = Path(params.get('outfile', 'moves.jsonl'))
        self.reset_on_start: bool = params.get('reset_on_start', True)

        self.outfile.parent.mkdir(parents=True, exist_ok=True)
        self._prepare_outfile_once()

    def _prepare_outfile_once(self) -> None:
        key = str(self.outfile.resolve())
        if key in self._initialized_paths:
            return

        if self.reset_on_start:
            self.outfile.write_text('', encoding='utf-8')

        self._initialized_paths.add(key)

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
                indiv = row[1]
                host = row[2]
                host = self.host_at(indiv, ts, host)
                rows.append({
                    self.tick_field: ts,
                    self.indiv_field: indiv,
                    self.host_field: host,
                    self.metrics[0]: float(row[3]),
                })
            else:
                break

        return pd.DataFrame(rows)

    def close(self) -> None:
        """Close the underlying CSV file."""
        self.csv_file.close()

    def add_moves(self, current_time: int, moves: List[Tuple[str, str]]):
        """
        Register moves happening at `current_time`.
        `moves` is [(container_id, target_node), ...]. Last one wins for same container/time.
        """
        # Coalesce duplicate containers in this batch so "last one wins"
        last = {}
        for c, n in moves:
            last[c] = n
        for c, n in last.items():
            lst = self._moves[c]
            # keep list sorted by time; append if strictly increasing time (fast path)
            if not lst or current_time >= lst[-1][0]:
                # avoid duplicates (same target as previous) to keep it compact
                if not lst or lst[-1][0] != current_time or lst[-1][1] != n:
                    lst.append((current_time, n))
            else:
                # rare case: late insertion; keep it sorted
                lst.append((current_time, n))
                lst.sort(key=lambda x: x[0])

    def host_at(self, container_id: str, ts: int, original_host: str) -> str:
        """Return host to use at timestamp ts, applying latest move with time <= ts."""
        lst = self._moves.get(container_id)
        if not lst:
            return original_host
        # binary search rightmost index where move_time <= ts
        idx = bisect_right(lst, (ts, chr(0x10FFFF))) - 1
        if idx >= 0:
            return lst[idx][1]
        return original_host

    def apply_moves(self, moves, current_time):
        """Write moves (list of dicts) to the output file."""
        # solution may be a list of move dicts, or a model with extract_moves()

        if not moves:
            return  # nothing to do

        self.add_moves(current_time, moves)

        # always append during the run
        with self.outfile.open('a', encoding='utf-8') as f:
            for move in moves:
                f.write(json.dumps(move, ensure_ascii=False) + '\n')
