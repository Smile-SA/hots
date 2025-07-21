from core.interfaces import HeuristicPlugin
from itertools import combinations

class SpreadHeuristic(HeuristicPlugin):
    def __init__(self, params, instance):
        self.method = params.get('method','spread')
        self.min_nodes = params.get('min_nodes',1)

    def adjust(self, solution, moving_containers=None):
        """
        Apply the spread heuristic. The optional moving_containers list
        is provided by the evaluator but not used here.
        """
        # existing spread / pairwise logic ported here
        # you can use moving_containers if you want to bias your moves
        return solution
