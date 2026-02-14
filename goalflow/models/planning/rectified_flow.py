"""Rectified Flow: Flow matching for trajectory generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class RectifiedFlow(nn.Module):
    """
    Rectified Flow for trajectory generation.

    Implements the flow matching approach for generating trajectories
    from goal points to final trajectories.

    Key idea: Learn to predict velocity v_t that moves from noise x_0
    to data x_1 along a straight line path.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_steps: int = 10,
        traj_dim: int = 2,  # trajectory dimension (x, y)
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.traj_dim = traj_dim

        # Input dimension: trajectory (2) + timestep (1) + goal (2) + context (hidden_dim)
        input_dim = traj_dim + 1 + traj_dim + hidden_dim  # 2 + 1 + 2 + hidden_dim

        # Velocity prediction network
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, traj_dim))  # Output velocity (2)
        self.velocity_net = nn.Sequential(*layers)

    def get_velocity(
        self,
        trajectory: torch.Tensor,
        timestep: torch.Tensor,
        goal: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity at given timestep.

        Args:
            trajectory: (B, T, D) current trajectory points
            timestep: (B,) timestep t in [0, 1]
            goal: (B, D) goal position
            context: (B, C) optional context features (BEV features)

        Returns:
            velocity: (B, T, D) predicted velocity
        """
        B, T, D = trajectory.shape

        # Expand timestep
        t = timestep.view(B, 1, 1).expand(B, T, 1)  # (B, T, 1)

        # Expand goal to (B, T, D_goal)
        if goal.dim() == 3:
            # Already expanded: (B, 1, D) -> (B, T, D)
            goal_expanded = goal.expand(-1, T, -1)
        else:
            # (B, D) -> (B, T, D)
            goal_expanded = goal.unsqueeze(1).expand(-1, T, -1)

        # Concatenate trajectory, timestep, goal
        if context is not None:
            context_expanded = context.unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([trajectory, t, goal_expanded, context_expanded], dim=-1)
        else:
            x = torch.cat([trajectory, t, goal_expanded], dim=-1)

        # Predict velocity
        velocity = self.velocity_net(x)

        return velocity

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            x_0: (B, T, D) noise trajectories
            x_1: (B, T, D) ground truth trajectories
            context: (B, C) context features

        Returns:
            Dictionary with predicted velocity and loss
        """
        B, T, D = x_0.shape

        # Sample timestep
        t = torch.rand(B, device=x_0.device)

        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # Target velocity: v = x_1 - x_0
        v_target = x_1 - x_0

        # Predicted velocity
        v_pred = self.get_velocity(x_t, t, x_1[:, -1, :], context)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return {
            "loss": loss,
            "v_pred": v_pred,
            "v_target": v_target,
        }

    def inference(
        self,
        x_0: torch.Tensor,
        goal: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference: generate trajectory from noise to goal.

        Args:
            x_0: (B, T, D) initial noise trajectory
            goal: (B, D) goal position
            context: (B, C) context features
            num_steps: number of integration steps

        Returns:
            Generated trajectory: (B, T, D)
        """
        if num_steps is None:
            num_steps = self.num_steps

        x_t = x_0.clone()

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full(
                (x_t.shape[0],),
                step * dt,
                device=x_t.device,
            )

            v = self.get_velocity(x_t, t, goal, context)

            x_t = x_t + v * dt

        return x_t


class MultiStepRectifiedFlow(nn.Module):
    """
    Multi-step Rectified Flow for generating multiple trajectory candidates.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_steps: int = 10,
        num_candidates: int = 6,
    ):
        super().__init__()

        self.num_candidates = num_candidates

        # Single flow model (shared)
        self.flow = RectifiedFlow(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_steps=num_steps,
        )

    def forward(
        self,
        goals: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate multiple trajectory candidates from multiple goals.

        Args:
            goals: (B, K, D) K goal positions
            context: (B, C) context features

        Returns:
            trajectories: (B, K, T, D) generated trajectories
        """
        B, K, D = goals.shape

        # Generate noise trajectories
        T = 30  # trajectory length
        x_0 = torch.randn(B, K, T, D, device=goals.device)

        # Generate for each goal
        trajectories = []
        for k in range(K):
            goal_k = goals[:, k, :]  # (B, D)
            traj_k = self.flow.inference(
                x_0[:, k, :, :],
                goal_k,
                context,
                num_steps,
            )
            trajectories.append(traj_k)

        trajectories = torch.stack(trajectories, dim=1)

        return trajectories
