"""
GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectory Generation

A PyTorch implementation of the GoalFlow paper for end-to-end autonomous driving.
"""

__version__ = "0.1.0"

from goalflow.models import GoalFlowModel
from goalflow.config import load_config

__all__ = ["GoalFlowModel", "load_config"]
