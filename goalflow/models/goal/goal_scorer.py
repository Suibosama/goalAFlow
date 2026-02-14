"""Goal point scorer: Distance score + DAC score."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DistanceScorer(nn.Module):
    """Distance-based goal scorer."""

    def __init__(
        self,
        max_distance: float = 50.0,
        num_bins: int = 50,
        feature_dim: int = 64,
    ):
        super().__init__()
        self.max_distance = max_distance
        self.num_bins = num_bins

        # Histogram computation
        self.register_buffer(
            "distance_bins",
            torch.linspace(0, max_distance, num_bins),
        )

        # Scoring network: histogram (50) + goal_positions (2) + ego_state (4) = 56
        self.scoring_net = nn.Sequential(
            nn.Linear(num_bins + 6, feature_dim),  # histogram + goal_pos + ego_state
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def compute_distance_histogram(
        self,
        goal_positions: torch.Tensor,
        ego_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance histogram from goal positions.

        Args:
            goal_positions: (B, K, 2) goal positions in ego frame
            ego_position: (B, 3) ego position (x, y, heading)

        Returns:
            histogram: (B, K, num_bins) distance histograms
        """
        B, K, _ = goal_positions.shape

        # Compute distances
        distances = torch.norm(goal_positions, dim=-1)  # (B, K)

        # Compute histogram for each goal
        histograms = []
        for b in range(B):
            hist = torch.zeros(K, self.num_bins, device=goal_positions.device)
            for k in range(K):
                d = distances[b, k]
                # Simple bin assignment
                bin_idx = min(int(d / self.max_distance * self.num_bins), self.num_bins - 1)
                hist[k, bin_idx] = 1.0
            histograms.append(hist)

        return torch.stack(histograms, dim=0)

    def forward(
        self,
        goal_positions: torch.Tensor,
        ego_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance scores.

        Args:
            goal_positions: (B, K, 2) goal positions
            ego_state: (B, 4) ego state (x, y, heading, speed)

        Returns:
            scores: (B, K) distance scores
        """
        # Compute histograms
        histograms = self.compute_distance_histogram(goal_positions, ego_state)

        # Expand ego state
        ego_expanded = ego_state.unsqueeze(1).expand(-1, goal_positions.shape[1], -1)

        # Concatenate features
        features = torch.cat([histograms, goal_positions, ego_expanded], dim=-1)

        # Score
        scores = self.scoring_net(features).squeeze(-1)

        return scores


class DACScorer(nn.Module):
    """Drivability Area Classification scorer."""

    def __init__(
        self,
        bev_channels: int = 256,
        goal_channels: int = 64,
        num_classes: int = 2,  # drivable / not drivable
    ):
        super().__init__()

        # Goal feature extraction
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, goal_channels),
            nn.ReLU(),
            nn.Linear(goal_channels, goal_channels),
        )

        # Fusion + classification
        self.classifier = nn.Sequential(
            nn.Linear(bev_channels + goal_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Binary classification
        )

    def forward(
        self,
        goal_positions: torch.Tensor,
        bev_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DAC scores (drivability probability).

        Args:
            goal_positions: (B, K, 2) goal positions
            bev_features: (B, C, H, W) BEV features

        Returns:
            scores: (B, K) drivability scores
        """
        B, K, _ = goal_positions.shape
        C, H, W = bev_features.shape[1:]

        # Sample BEV features at goal positions
        # Convert positions to grid coordinates
        # Assume BEV spans [-50, 50] in both x and y
        bev_range = 50.0
        grid_x = ((goal_positions[..., 0] / bev_range + 1) * W / 2).long()
        grid_y = ((goal_positions[..., 1] / bev_range + 1) * H / 2).long()

        # Clamp to valid range
        grid_x = torch.clamp(grid_x, 0, W - 1)
        grid_y = torch.clamp(grid_y, 0, H - 1)

        # Gather features
        bev_flat = bev_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        grid_flat = grid_y * W + grid_x  # (B, K)

        bev_features_at_goals = []
        for b in range(B):
            feats = bev_flat[b, grid_flat[b]]  # (K, C)
            bev_features_at_goals.append(feats)

        bev_features_at_goals = torch.stack(bev_features_at_goals, dim=0)

        # Encode goal positions
        goal_features = self.goal_encoder(goal_positions)

        # Concatenate and classify
        combined = torch.cat([bev_features_at_goals, goal_features], dim=-1)
        scores = self.classifier(combined).squeeze(-1)

        return scores


class GoalScorer(nn.Module):
    """Combined goal scorer (Distance + DAC)."""

    def __init__(
        self,
        max_distance: float = 50.0,
        num_bins: int = 50,
        bev_channels: int = 256,
        distance_weight: float = 1.0,
        dac_weight: float = 1.0,
    ):
        super().__init__()

        self.distance_weight = distance_weight
        self.dac_weight = dac_weight

        self.distance_scorer = DistanceScorer(
            max_distance=max_distance,
            num_bins=num_bins,
        )
        self.dac_scorer = DACScorer(bev_channels=bev_channels)

    def forward(
        self,
        goal_positions: torch.Tensor,
        ego_state: torch.Tensor,
        bev_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined goal scores.

        Args:
            goal_positions: (B, K, 2) candidate goal positions
            ego_state: (B, 4) ego state
            bev_features: (B, C, H, W) BEV features

        Returns:
            Dictionary with:
                - total_scores: (B, K) combined scores
                - distance_scores: (B, K) distance scores
                - dac_scores: (B, K) DAC scores
        """
        # Distance scores
        distance_scores = self.distance_scorer(goal_positions, ego_state)
        distance_scores = torch.sigmoid(distance_scores)

        # DAC scores
        if bev_features is not None:
            dac_scores = self.dac_scorer(goal_positions, bev_features)
            dac_scores = torch.sigmoid(dac_scores)
        else:
            dac_scores = torch.ones_like(distance_scores)

        # Combined scores
        total_scores = (
            self.distance_weight * distance_scores +
            self.dac_weight * dac_scores
        )

        return {
            "total_scores": total_scores,
            "distance_scores": distance_scores,
            "dac_scores": dac_scores,
        }
