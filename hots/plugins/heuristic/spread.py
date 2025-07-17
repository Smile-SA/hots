from core.interfaces import HeuristicPlugin
from itertools import combinations

class SpreadHeuristic(HeuristicPlugin):
    def __init__(self, params, instance):
        self.method = params.get('method','spread')
        self.min_nodes = params.get('min_nodes',1)

    def adjust(self, solution):
        # existing spread / pairwise logic ported here
        return solution
