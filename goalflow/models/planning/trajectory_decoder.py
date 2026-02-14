"""Trajectory decoder: encode trajectories and context, generate future trajectories."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TrajectoryEncoder(nn.Module):
    """Encode trajectory sequences."""

    def __init__(
        self,
        input_dim: int = 2,  # x, y
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectories.

        Args:
            trajectories: (B, T, D) or (B, K, T, D)

        Returns:
            encoded: (B, D) or (B, K, D)
        """
        if trajectories.dim() == 4:
            # (B, K, T, D) -> (B*K, T, D)
            B, K, T, D = trajectories.shape
            trajectories = trajectories.view(B * K, T, D)
            encoded = self._encode(trajectories)
            encoded = encoded.view(B, K, -1)
        else:
            encoded = self._encode(trajectories)

        return encoded

    def _encode(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Encode single trajectory."""
        # MLP encode each point
        x = self.encoder(trajectories)  # (B, T, hidden_dim)

        # Transformer encode
        x = self.transformer(x)  # (B, T, hidden_dim)

        # Pool to get trajectory-level representation
        x = x.mean(dim=1)  # (B, hidden_dim)

        return x


class EnvironmentEncoder(nn.Module):
    """Encode environment (BEV features) for trajectory generation."""

    def __init__(
        self,
        bev_channels: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()

        # BEV encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Project to output dimension
        self.projection = nn.Linear(256, output_dim)

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Encode BEV features.

        Args:
            bev_features: (B, C, H, W)

        Returns:
            encoded: (B, output_dim)
        """
        x = self.encoder(bev_features)  # (B, 256, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 256)
        x = self.projection(x)  # (B, output_dim)

        return x


class TrajectoryDecoder(nn.Module):
    """
    Trajectory decoder: generates future trajectories given goals and context.
    """

    def __init__(
        self,
        trajectory_dim: int = 128,
        bev_channels: int = 256,
        output_dim: int = 2,  # x, y
        num_layers: int = 3,
    ):
        super().__init__()

        self.trajectory_dim = trajectory_dim
        self.output_dim = output_dim

        # Encoders
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=output_dim,
            hidden_dim=trajectory_dim,
            num_layers=num_layers,
        )
        self.environment_encoder = EnvironmentEncoder(
            bev_channels=bev_channels,
            output_dim=trajectory_dim,
        )

        # Goal encoding
        self.goal_encoder = nn.Sequential(
            nn.Linear(output_dim, trajectory_dim),
            nn.ReLU(),
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=trajectory_dim,
            nhead=4,
            dim_feedforward=trajectory_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output head
        self.output_head = nn.Linear(trajectory_dim, output_dim)

    def forward(
        self,
        goals: torch.Tensor,
        bev_features: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectories.

        Args:
            goals: (B, K, D) goal positions
            bev_features: (B, C, H, W) BEV features
            history: (B, T_hist, D) historical trajectory

        Returns:
            Dictionary with:
                - trajectories: (B, K, T, D) generated trajectories
        """
        B, K, D = goals.shape

        # Encode environment
        env_features = self.environment_encoder(bev_features)  # (B, trajectory_dim)

        # Encode goals
        goal_features = self.goal_encoder(goals)  # (B, K, trajectory_dim)

        # Create decoder input (start from origin)
        T_future = 30
        decoder_input = torch.zeros(B, K, 1, self.trajectory_dim, device=goals.device)

        # Decode
        decoded = self.decoder(
            decoder_input,
            goal_features.unsqueeze(2).expand(-1, -1, T_future, -1),
        )  # (B, K, T, trajectory_dim)

        # Output trajectory
        trajectories = self.output_head(decoded)

        return {
            "trajectories": trajectories,
        }

    def decode_with_flow(
        self,
        goals: torch.Tensor,
        bev_features: torch.Tensor,
        flow_model: nn.Module,
    ) -> torch.Tensor:
        """
        Decode using Rectified Flow.

        Args:
            goals: (B, K, D) goal positions
            bev_features: (B, C, H, W) BEV features
            flow_model: RectifiedFlow model

        Returns:
            trajectories: (B, K, T, D) generated trajectories
        """
        B, K, D = goals.shape

        # Encode environment as context
        env_features = self.environment_encoder(bev_features)

        # Generate trajectories using flow
        trajectories = flow_model.forward(goals, env_features)

        return trajectories
