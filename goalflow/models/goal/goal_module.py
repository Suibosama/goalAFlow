"""Goal module: vocabulary + scorer + selector."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from goalflow.models.goal.goal_vocabulary import GoalVocabulary
from goalflow.models.goal.goal_scorer import GoalScorer
from goalflow.models.goal.goal_selector import GoalSelector


class GoalModule(nn.Module):
    """
    Goal point module for selecting destination goals.

    Combines:
    - GoalVocabulary: discrete set of candidate goal positions
    - GoalScorer: scores goals based on distance + drivability
    - GoalSelector: selects top-k goals for trajectory generation
    """

    def __init__(
        self,
        num_clusters: int = 256,
        top_k: int = 10,
        bev_channels: int = 256,
        use_dac: bool = True,
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.top_k = top_k

        # Vocabulary (not a learnable module, built from data)
        self.vocabulary: Optional[GoalVocabulary] = None

        # Scorer
        self.scorer = GoalScorer(
            max_distance=50.0,
            num_bins=50,
            bev_channels=bev_channels,
        )

        # Selector
        self.selector = GoalSelector(top_k=top_k)

        # Register cluster centers as buffer (non-learnable)
        self.register_buffer("cluster_centers", torch.zeros(num_clusters, 2))

    def build_vocabulary(self, trajectories) -> None:
        """Build goal vocabulary from training trajectories."""
        import numpy as np

        # Convert trajectories to endpoints
        endpoints = []
        for traj in trajectories:
            if isinstance(traj, torch.Tensor):
                traj = traj.cpu().numpy()
            if traj.ndim == 3:
                endpoints.append(traj[-1, :2])
            else:
                endpoints.append(traj[-2:])

        endpoints = np.array(endpoints)

        # Build vocabulary
        self.vocabulary = GoalVocabulary(
            num_clusters=self.num_clusters,
            cluster_dims=2,
        )
        self.vocabulary.build(endpoints)

        # Update buffer
        centers = torch.from_numpy(self.vocabulary.get_all_centers()).float()
        self.register_buffer("cluster_centers", centers)

    def get_candidate_goals(self, batch_size: int) -> torch.Tensor:
        """
        Get candidate goal positions.

        Args:
            batch_size: Batch size

        Returns:
            goal_positions: (B, K, 2) candidate positions
        """
        # Use vocabulary centers
        centers = self.cluster_centers  # (num_clusters, 2)

        # For simplicity, use all centers as candidates
        # In practice, might sample or filter based on context
        if centers.shape[0] >= self.top_k:
            # Sample top_k centers (can be made smarter)
            indices = torch.randperm(centers.shape[0])[:self.top_k]
            goals = centers[indices].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Repeat if fewer clusters than top_k
            repeats = (self.top_k + centers.shape[0] - 1) // centers.shape[0]
            goals = centers.repeat(repeats, 1)[:self.top_k].unsqueeze(0).expand(batch_size, -1, -1)

        return goals

    def forward(
        self,
        ego_state: torch.Tensor,
        bev_features: Optional[torch.Tensor] = None,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for goal selection.

        Args:
            ego_state: (B, 4) ego state (x, y, heading, speed)
            bev_features: (B, C, H, W) BEV features
            ground_truth: (B, 2) ground truth goal (optional, for training)

        Returns:
            Dictionary with:
                - selected_goals: (B, top_k, 2) selected goal positions
                - goal_scores: (B, top_k) scores for selected goals
                - candidate_goals: (B, top_k, 2) all candidate goals
        """
        B = ego_state.shape[0]

        # Get candidate goals
        candidate_goals = self.get_candidate_goals(B)  # (B, top_k, 2)

        # Score candidates
        scores_dict = self.scorer(candidate_goals, ego_state, bev_features)
        scores = scores_dict["total_scores"]

        # Select goals
        if ground_truth is not None and self.training:
            # Include ground truth during training
            selected = self.selector.select_with_ground_truth(
                candidate_goals, scores, ground_truth
            )
        else:
            selected = self.selector.select(candidate_goals, scores)

        return {
            "selected_goals": selected["selected_goals"],
            "goal_scores": selected["selected_scores"],
            "candidate_goals": candidate_goals,
        }


def create_goal_module(config: dict) -> GoalModule:
    """Create goal module from config."""
    model_config = config.get("model", {})
    goal_config = model_config.get("goal", {})
    perception_config = model_config.get("perception", {})

    return GoalModule(
        num_clusters=goal_config.get("num_clusters", 256),
        top_k=goal_config.get("top_k", 10),
        bev_channels=perception_config.get("image_feature_dim", 256),
        use_dac=goal_config.get("use_dac", True),
    )
