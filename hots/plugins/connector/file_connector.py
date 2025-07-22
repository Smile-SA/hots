# hots/plugins/connector/file_connector.py

"""File‑based connector plugin for HOTS."""

import json

from core.interfaces import ConnectorPlugin


class FileConnector(ConnectorPlugin):
    """Connector plugin that writes moves to a JSONL file."""

    def __init__(self, params, instance):
        """Initialize with output file path."""
        self.outfile = params.get('outfile', 'moves.jsonl')
        from pathlib import Path
        Path(self.outfile).parent.mkdir(parents=True, exist_ok=True)

    def apply_moves(self, solution):
        """Write relocation moves to a JSONL file."""
        moves = solution.extract_moves()
        with open(self.outfile, 'a') as f:
            for m in moves:
                f.write(json.dumps(m) + '\n')
