"""GoalFlow main model."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from goalflow.models.perception.transfuser import TransFuser
from goalflow.models.goal.goal_module import GoalModule
from goalflow.models.planning.trajectory_planner import TrajectoryPlanner


class GoalFlowModel(nn.Module):
    """
    GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectory Generation.

    End-to-end autonomous driving model that:
    1. Encodes sensor inputs (images + LiDAR) into BEV features
    2. Selects goal points from a learned vocabulary
    3. Generates trajectories using rectified flow
    4. Scores and selects the best trajectory
    """

    def __init__(
        self,
        num_cameras: int = 3,
        image_feature_dim: int = 256,
        bev_feature_dim: int = 256,
        num_clusters: int = 256,
        top_k_goals: int = 10,
        num_candidates: int = 6,
    ):
        super().__init__()

        # Perception module (TransFuser)
        self.perception = TransFuser(
            image_feature_dim=image_feature_dim,
            lidar_feature_dim=image_feature_dim,
            bev_feature_dim=bev_feature_dim,
            num_cameras=num_cameras,
            transformer_layers=4,
            num_heads=8,
        )

        # Goal module
        self.goal_module = GoalModule(
            num_clusters=num_clusters,
            top_k=top_k_goals,
            bev_channels=bev_feature_dim,
            use_dac=True,
        )

        # Planning module
        self.planner = TrajectoryPlanner(
            trajectory_dim=128,
            bev_channels=bev_feature_dim,
            num_candidates=num_candidates,
            flow_steps=10,
        )

    def forward(
        self,
        images: torch.Tensor,
        lidar_points: Optional[torch.Tensor] = None,
        ego_state: Optional[torch.Tensor] = None,
        ground_truth_goal: Optional[torch.Tensor] = None,
        ground_truth_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: (B, N, 3, H, W) multi-camera images
            lidar_points: (B, N_points, 4) LiDAR points
            ego_state: (B, 4) ego vehicle state (x, y, heading, speed)
            ground_truth_goal: (B, 2) ground truth goal position
            ground_truth_trajectory: (B, T, 2) ground truth future trajectory

        Returns:
            Dictionary with all outputs
        """
        # 1. Perception: encode images and LiDAR into BEV
        perception_output = self.perception(images, lidar_points)
        bev_features = perception_output["bev_features"]

        # 2. Goal: select goal points
        if ego_state is None:
            ego_state = torch.zeros(images.shape[0], 4, device=images.device)

        goal_output = self.goal_module(
            ego_state=ego_state,
            bev_features=bev_features,
            ground_truth=ground_truth_goal,
        )
        selected_goals = goal_output["selected_goals"]

        # 3. Planning: generate and score trajectories
        planning_output = self.planner(
            goals=selected_goals,
            bev_features=bev_features,
            ground_truth=ground_truth_trajectory,
        )

        # Combine all outputs
        output = {
            **perception_output,
            **goal_output,
            **planning_output,
        }

        return output

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        ground_truth_goal: Optional[torch.Tensor] = None,
        ground_truth_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: output from forward()
            ground_truth_goal: (B, 2) ground truth goal
            ground_truth_trajectory: (B, T, 2) ground truth trajectory

        Returns:
            Dictionary with losses
        """
        losses = {}

        # Goal loss (if ground truth available)
        if ground_truth_goal is not None:
            selected_goals = predictions["selected_goals"]
            # Find closest goal to GT
            gt_expanded = ground_truth_goal.unsqueeze(1)
            distances = torch.norm(selected_goals - gt_expanded, dim=-1)
            min_dist = distances.min(dim=1)[0]
            losses["loss_goal"] = min_dist.mean()

        # Planning loss (if ground truth available)
        if ground_truth_trajectory is not None:
            planning_losses = self.planner.compute_loss(
                predictions, ground_truth_trajectory
            )
            losses.update(planning_losses)

        # Total loss
        if losses:
            losses["loss_total"] = sum(losses.values())

        return losses


def create_model(config: dict) -> GoalFlowModel:
    """Create GoalFlow model from config."""
    model_config = config.get("model", {})
    perception_config = model_config.get("perception", {})
    goal_config = model_config.get("goal", {})
    planning_config = model_config.get("planning", {})

    return GoalFlowModel(
        num_cameras=config.get("data", {}).get("num_cameras", 3),
        image_feature_dim=perception_config.get("image_feature_dim", 256),
        bev_feature_dim=perception_config.get("image_feature_dim", 256),
        num_clusters=goal_config.get("num_clusters", 256),
        top_k_goals=goal_config.get("top_k", 10),
        num_candidates=planning_config.get("num_candidates", 6),
    )
