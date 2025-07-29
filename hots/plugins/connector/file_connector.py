# hots/plugins/connector/file_connector.py

"""File‑based connector plugin for HOTS."""

import json
import os
from pathlib import Path

from hots.core.interfaces import ConnectorPlugin


class FileConnector(ConnectorPlugin):
    """Connector plugin that writes moves to a JSONL file."""

    def __init__(self, params, instance):
        """Initialize with output file path."""
        self.outfile = params.get('outfile', 'moves.jsonl')
        from pathlib import Path
        Path(self.outfile).parent.mkdir(parents=True, exist_ok=True)

    def apply_moves(self, solution):
        """Write moves (list of dicts) to the output file."""
        # solution may be a list of move dicts, or a model with extract_moves()
        if isinstance(solution, list):
            moves = solution
        else:
            moves = solution.extract_moves()
        # ensure output directory exists
        outdir = Path(self.outfile).parent
        outdir.mkdir(parents=True, exist_ok=True)
        with open(self.outfile, 'a') as f:
            for move in moves:
                # one JSON per line
                f.write(json.dumps(move) + os.linesep)
