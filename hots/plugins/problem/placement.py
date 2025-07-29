"""Placement problem plugin for HOTS."""

from typing import Any, Dict

from hots.core.interfaces import ProblemPlugin


class PlacementPlugin(ProblemPlugin):
    """Handles the 'placement' business problem."""

    def __init__(self, params: Dict[str, Any], instance):
        """Initialize with placement‑specific parameters and instance."""
        self.instance = instance
        self.params = params

    def adjust(self, solution: Any, **kwargs) -> Any:
        """
        Given the raw optimization solution (and optional conflict data),
        return the final placement moves (e.g. solution.extract_moves()).
        """
        return solution
