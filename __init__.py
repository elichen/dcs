"""
Direct Control System (DCS) for Fetch Robot

A mathematical control framework providing instant, precise manipulation
without machine learning or training.
"""

__version__ = "1.0.0"
__author__ = "Claude"

# Make core components easily importable
from dcs.core.fetch_ik_solver import FetchIKSolver
from dcs.core.fetch_path_planner import FetchPathPlanner
from dcs.core.fetch_claude_controller import FetchClaudeController

__all__ = [
    "FetchIKSolver",
    "FetchPathPlanner", 
    "FetchClaudeController"
]