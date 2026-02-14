"""Trajectory planner: end-to-end planning module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from goalflow.models.planning.rectified_flow import RectifiedFlow, MultiStepRectifiedFlow
from goalflow.models.planning.trajectory_decoder import TrajectoryDecoder, EnvironmentEncoder
from goalflow.models.planning.trajectory_scorer import TrajectoryScorer


class TrajectoryPlanner(nn.Module):
    """
    Trajectory planner: generates and selects trajectories given goals and BEV.

    Combines:
    - TrajectoryDecoder: generates candidate trajectories
    - TrajectoryScorer: evaluates and selects best trajectory
    """

    def __init__(
        self,
        trajectory_dim: int = 128,
        bev_channels: int = 256,
        num_candidates: int = 6,
        flow_steps: int = 10,
    ):
        super().__init__()

        self.num_candidates = num_candidates
        self.trajectory_dim = trajectory_dim

        # Environment encoder to convert BEV to context
        self.env_encoder = EnvironmentEncoder(
            bev_channels=bev_channels,
            output_dim=trajectory_dim,
        )

        # Rectified Flow for trajectory generation
        self.flow = RectifiedFlow(
            hidden_dim=trajectory_dim,
            num_layers=3,
            num_steps=flow_steps,
        )

        # Multi-step flow for multiple candidates
        self.multi_flow = MultiStepRectifiedFlow(
            hidden_dim=trajectory_dim,
            num_layers=3,
            num_steps=flow_steps,
            num_candidates=num_candidates,
        )

        # Trajectory scorer
        self.scorer = TrajectoryScorer(
            trajectory_dim=trajectory_dim,
            bev_channels=bev_channels,
        )

    def forward(
        self,
        goals: torch.Tensor,
        bev_features: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate and score trajectories.

        Args:
            goals: (B, K, 2) goal positions (K >= num_candidates)
            bev_features: (B, C, H, W) BEV features
            ground_truth: (B, T, 2) ground truth trajectory (for training)

        Returns:
            Dictionary with:
                - trajectories: (B, num_candidates, T, 2) generated trajectories
                - scores: (B, num_candidates) trajectory scores
                - best_trajectory: (B, T, 2) selected best trajectory
        """
        B, K_full, _ = goals.shape

        # Encode BEV to context
        context = self.env_encoder(bev_features)  # (B, trajectory_dim)

        # Select top-k goals for generation
        K = min(K_full, self.num_candidates)
        goals_subset = goals[:, :K, :]

        # Generate trajectories using flow
        if K > 1:
            trajectories = self.multi_flow(goals_subset, context)
        else:
            # Single trajectory
            x_0 = torch.randn(B, 30, 2, device=goals.device)
            trajectories = self.flow.inference(x_0, goals_subset.squeeze(1), context)
            trajectories = trajectories.unsqueeze(1)

        # Score trajectories
        score_dict = self.scorer(trajectories, goals_subset, bev_features)
        scores = score_dict["scores"]

        # Select best
        best_trajectory, best_indices = self.scorer.select_best(trajectories, scores)

        return {
            "trajectories": trajectories,
            "scores": scores,
            "best_trajectory": best_trajectory,
            "best_indices": best_indices,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        ground_truth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute planning loss.

        Args:
            predictions: output from forward()
            ground_truth: (B, T, 2) ground truth trajectory

        Returns:
            Dictionary with losses
        """
        # Select trajectory closest to ground truth
        trajectories = predictions["trajectories"]  # (B, K, T, 2)
        B, K, T, _ = trajectories.shape

        # Find closest trajectory to ground truth
        gt_expanded = ground_truth.unsqueeze(1)  # (B, 1, T, 2)
        distances = torch.norm(trajectories - gt_expanded, dim=-1).mean(dim=-1)  # (B, K)
        closest_indices = distances.argmin(dim=1)  # (B,)

        # Get closest trajectory
        closest_trajs = []
        for b in range(B):
            closest_trajs.append(trajectories[b, closest_indices[b]])
        closest_trajs = torch.stack(closest_trajs, dim=0)

        # MSE loss
        loss_mse = nn.functional.mse_loss(closest_trajs, ground_truth)

        return {
            "loss_planning": loss_mse,
        }


def create_trajectory_planner(config: dict) -> TrajectoryPlanner:
    """Create trajectory planner from config."""
    model_config = config.get("model", {})
    planning_config = model_config.get("planning", {})
    perception_config = model_config.get("perception", {})

    return TrajectoryPlanner(
        trajectory_dim=planning_config.get("trajectory_dim", 128),
        bev_channels=perception_config.get("image_feature_dim", 256),
        num_candidates=planning_config.get("num_candidates", 6),
        flow_steps=planning_config.get("flow_steps", 10),
    )
