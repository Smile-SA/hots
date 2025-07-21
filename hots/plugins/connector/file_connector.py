import json
from core.interfaces import ConnectorPlugin

class FileConnector(ConnectorPlugin):
    def __init__(self, params, instance):
        self.outfile = params.get('outfile', 'moves.jsonl')

    def apply_moves(self, solution):
        moves = solution.extract_moves()
        with open(self.outfile, 'a') as f:
            for m in moves:
                f.write(json.dumps(m) + '\\n')
