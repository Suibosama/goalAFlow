"""Trajectory scorer: score and select best trajectories."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TrajectoryScorer(nn.Module):
    """
    Trajectory scorer: evaluates generated trajectories and selects the best one.

    Implements the scoring function:
    f(τ) = -λ1*Φ(f_dis) + λ2*Φ(f_pg)
    """

    def __init__(
        self,
        trajectory_dim: int = 128,
        bev_channels: int = 256,
        lambda_progress: float = 1.0,
        lambda_collision: float = 2.0,
        lambda_comfort: float = 0.5,
    ):
        super().__init__()

        self.lambda_progress = lambda_progress
        self.lambda_collision = lambda_collision
        self.lambda_comfort = lambda_comfort

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Progress score head
        self.progress_head = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Collision score head
        self.collision_head = nn.Sequential(
            nn.Linear(128 + bev_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Comfort score head
        self.comfort_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute_progress_score(
        self,
        trajectory: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute progress score (how well trajectory reaches goal).

        Args:
            trajectory: (B, T, 2)
            goal: (B, 2)

        Returns:
            scores: (B,)
        """
        # Distance to goal at each timestep
        final_position = trajectory[:, -1, :]  # (B, 2)
        distance_to_goal = torch.norm(final_position - goal, dim=-1)  # (B,)

        # Progress = 1 / (1 + distance)
        progress = 1.0 / (1.0 + distance_to_goal)

        return progress

    def compute_collision_score(
        self,
        trajectory: torch.Tensor,
        bev_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute collision score (probability of collision).

        Simplified: check if trajectory goes through non-drivable areas.

        Args:
            trajectory: (B, T, 2)
            bev_features: (B, C, H, W)

        Returns:
            scores: (B,)
        """
        B, T, _ = trajectory.shape
        C, H, W = bev_features.shape[1:]

        # Sample BEV at trajectory points
        bev_range = 50.0
        grid_x = ((trajectory[..., 0] / bev_range + 1) * W / 2).long()
        grid_y = ((trajectory[..., 1] / bev_range + 1) * H / 2).long()

        # Clamp
        grid_x = torch.clamp(grid_x, 0, W - 1)
        grid_y = torch.clamp(grid_y, 0, H - 1)

        # Gather BEV features
        bev_flat = bev_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        grid_flat = grid_y * W + grid_x  # (B, T)

        bev_at_traj = []
        for b in range(B):
            feats = bev_flat[b, grid_flat[b]]  # (T, C)
            bev_at_traj.append(feats)
        bev_at_traj = torch.stack(bev_at_traj, dim=0)  # (B, T, C)

        # Average BEV features along trajectory
        bev_avg = bev_at_traj.mean(dim=1)  # (B, C)

        # Extract trajectory features
        traj_features = self.feature_extractor(trajectory)  # (B, T, 128)
        traj_avg = traj_features.mean(dim=1)  # (B, 128)

        # Compute collision probability
        combined = torch.cat([traj_avg, bev_avg], dim=-1)
        collision_logit = self.collision_head(combined).squeeze(-1)

        return torch.sigmoid(collision_logit)

    def compute_comfort_score(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute comfort score (smoothness of trajectory).

        Args:
            trajectory: (B, T, 2)

        Returns:
            scores: (B,)
        """
        # Compute accelerations
        velocity = trajectory[:, 1:, :] - trajectory[:, :-1, :]  # (B, T-1, 2)
        acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]  # (B, T-2, 2)

        # Comfort = 1 / (1 + mean acceleration magnitude)
        acc_magnitude = torch.norm(acceleration, dim=-1)  # (B, T-2)
        mean_acc = acc_magnitude.mean(dim=-1)  # (B,)

        comfort = 1.0 / (1.0 + mean_acc)

        return comfort

    def forward(
        self,
        trajectories: torch.Tensor,
        goals: torch.Tensor,
        bev_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Score trajectories.

        Args:
            trajectories: (B, K, T, 2) K trajectory candidates
            goals: (B, K, 2) corresponding goals
            bev_features: (B, C, H, W) BEV features

        Returns:
            Dictionary with:
                - scores: (B, K) overall scores
                - progress_scores: (B, K)
                - collision_scores: (B, K)
                - comfort_scores: (B, K)
        """
        B, K, T, _ = trajectories.shape

        # Compute individual scores
        progress_scores = []
        collision_scores = []
        comfort_scores = []

        for k in range(K):
            traj_k = trajectories[:, k, :, :]  # (B, T, 2)
            goal_k = goals[:, k, :]  # (B, 2)

            progress = self.compute_progress_score(traj_k, goal_k)
            progress_scores.append(progress)

            collision = self.compute_collision_score(traj_k, bev_features)
            collision_scores.append(collision)

            comfort = self.compute_comfort_score(traj_k)
            comfort_scores.append(comfort)

        progress_scores = torch.stack(progress_scores, dim=1)  # (B, K)
        collision_scores = torch.stack(collision_scores, dim=1)  # (B, K)
        comfort_scores = torch.stack(comfort_scores, dim=1)  # (B, K)

        # Combined score
        scores = (
            self.lambda_progress * progress_scores -
            self.lambda_collision * collision_scores +
            self.lambda_comfort * comfort_scores
        )

        return {
            "scores": scores,
            "progress_scores": progress_scores,
            "collision_scores": collision_scores,
            "comfort_scores": comfort_scores,
        }

    def select_best(
        self,
        trajectories: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select best trajectory based on scores.

        Args:
            trajectories: (B, K, T, 2)
            scores: (B, K)

        Returns:
            best_trajectories: (B, T, 2)
            best_indices: (B,)
        """
        best_indices = scores.argmax(dim=1)  # (B,)

        best_trajectories = []
        for b in range(trajectories.shape[0]):
            best_trajectories.append(trajectories[b, best_indices[b]])
        best_trajectories = torch.stack(best_trajectories, dim=0)

        return best_trajectories, best_indices
