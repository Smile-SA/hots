# hots/plugins/connector/file_connector.py

"""File‑based connector plugin for HOTS."""

import json
from pathlib import Path

from hots.core.interfaces import ConnectorPlugin


class FileConnector(ConnectorPlugin):
    """Connector plugin that writes moves to a JSONL file."""

    # process-level guard so we only truncate once per path per run
    _initialized_paths: set[str] = set()

    def __init__(self, params, reader):
        """Initialize with output file path."""
        self.outfile = Path(params.get('outfile', 'moves.jsonl'))
        self.reset_on_start: bool = params.get('reset_on_start', True)

        self.outfile.parent.mkdir(parents=True, exist_ok=True)
        self._prepare_outfile_once()
        self.reader = reader

    def _prepare_outfile_once(self) -> None:
        key = str(self.outfile.resolve())
        if key in self._initialized_paths:
            return

        if self.reset_on_start:
            self.outfile.write_text('', encoding='utf-8')

        self._initialized_paths.add(key)

    def apply_moves(self, moves, current_time):
        """Write moves (list of dicts) to the output file."""
        # solution may be a list of move dicts, or a model with extract_moves()

        if not moves:
            return  # nothing to do

        self.reader.add_moves(current_time, moves)

        # always append during the run
        with self.outfile.open('a', encoding='utf-8') as f:
            for move in moves:
                f.write(json.dumps(move, ensure_ascii=False) + '\n')
