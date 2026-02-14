"""Goal point selector."""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class GoalSelector:
    """Select top-k goals based on scores."""

    def __init__(
        self,
        top_k: int = 10,
        sampling_strategy: str = "top_k",
        temperature: float = 1.0,
    ):
        """
        Args:
            top_k: Number of top goals to select
            sampling_strategy: "top_k" or "sampling"
            temperature: Temperature for softmax sampling
        """
        self.top_k = top_k
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature

    def select(
        self,
        goal_positions: torch.Tensor,
        scores: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Select top-k goals.

        Args:
            goal_positions: (B, K, 2) all candidate goal positions
            scores: (B, K) goal scores
            ground_truth: (B, 2) ground truth goal position (for training)

        Returns:
            Dictionary with:
                - selected_goals: (B, top_k, 2) selected goal positions
                - selected_indices: (B, top_k) indices of selected goals
                - selected_scores: (B, top_k) scores of selected goals
        """
        B, K, _ = goal_positions.shape

        if self.sampling_strategy == "top_k":
            # Deterministic top-k selection
            top_scores, top_indices = torch.topk(scores, self.top_k, dim=1)
        elif self.sampling_strategy == "sampling":
            # Stochastic sampling with temperature
            probs = torch.softmax(scores / self.temperature, dim=-1)
            top_indices = torch.multinomial(probs, self.top_k, replacement=False)
            top_scores = scores.gather(1, top_indices)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # Gather selected goals
        selected_goals = []
        for b in range(B):
            goals_b = goal_positions[b, top_indices[b]]
            selected_goals.append(goals_b)

        selected_goals = torch.stack(selected_goals, dim=0)

        return {
            "selected_goals": selected_goals,
            "selected_indices": top_indices,
            "selected_scores": top_scores,
        }

    def select_with_ground_truth(
        self,
        goal_positions: torch.Tensor,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Select goals ensuring ground truth is included (for training).

        Args:
            goal_positions: (B, K, 2) all candidate goal positions
            scores: (B, K) goal scores
            ground_truth: (B, 2) ground truth goal position

        Returns:
            Dictionary with selected goals (including ground truth)
        """
        B, K, _ = goal_positions.shape

        # Find closest goal to ground truth
        gt_expanded = ground_truth.unsqueeze(1)  # (B, 1, 2)
        distances = torch.norm(goal_positions - gt_expanded, dim=-1)  # (B, K)
        gt_indices = torch.argmin(distances, dim=1)  # (B,)

        # Get top-k-1 other goals
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(1, gt_indices.unsqueeze(1), False)

        scores_masked = scores.clone()
        scores_masked[~mask] = -float("inf")

        top_scores, top_indices = torch.topk(scores_masked, self.top_k - 1, dim=1)

        # Include ground truth index
        final_indices = torch.cat([gt_indices.unsqueeze(1), top_indices], dim=1)

        # Gather selected goals
        selected_goals = []
        for b in range(B):
            goals_b = goal_positions[b, final_indices[b]]
            selected_goals.append(goals_b)

        selected_goals = torch.stack(selected_goals, dim=0)
        selected_scores = scores.gather(1, final_indices)

        return {
            "selected_goals": selected_goals,
            "selected_indices": final_indices,
            "selected_scores": selected_scores,
        }


def create_goal_selector(config: dict) -> GoalSelector:
    """Create goal selector from config."""
    goal_config = config.get("goal", {})
    return GoalSelector(
        top_k=goal_config.get("top_k", 10),
        sampling_strategy=goal_config.get("sampling_strategy", "top_k"),
        temperature=goal_config.get("temperature", 1.0),
    )
