"""TransFuser: Multi-modal fusion for BEV perception."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ImageEncoder(nn.Module):
    """Image encoder using ResNet backbone."""

    def __init__(
        self,
        backbone: str = "resnet34",
        feature_dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Load backbone
        if backbone == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights

            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone_model = resnet34(weights=weights)
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
            backbone_out_dim = 512
        elif backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone_model = resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
            backbone_out_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection layer
        self.proj = nn.Conv2d(backbone_out_dim, feature_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, N, C, H, W) where N is number of cameras

        Returns:
            features: (B, N, feature_dim, H', W')
        """
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)

        features = self.backbone(images)
        features = self.proj(features)

        _, C_f, H_f, W_f = features.shape
        features = features.view(B, N, C_f, H_f, W_f)

        return features


class LiDAREncoder(nn.Module):
    """LiDAR point cloud encoder."""

    def __init__(
        self,
        feature_dim: int = 256,
        voxel_size: List[float] = [0.4, 0.4, 0.4],
        point_cloud_range: List[float] = [-50, -50, -3, 50, 50, 3],
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        # Simple voxelization + MLP (simplified PointPillars)
        # In practice, you'd use a proper voxelization layer
        self.voxel_grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ]

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(7, 64),  # x, y, z, intensity, room_x, room_y, room_z
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(
        self,
        points: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 4) or (total_points, 4) - x, y, z, intensity
            batch_indices: (total_points,) - batch index for each point

        Returns:
            features: (B, feature_dim, H, W) BEV features
        """
        if points.dim() == 2:
            # Single sample: (N, 4)
            points = points.unsqueeze(0)
            batch_indices = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)

        B, N, _ = points.shape

        # Flatten
        points = points.view(B * N, -1)

        # Normalize coordinates to voxel grid
        points[:, 0] = (points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        points[:, 1] = (points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        points[:, 2] = (points[:, 2] - self.point_cloud_range[2]) / self.voxel_size[2]

        # Simple voxel aggregation (mean pooling)
        # In practice, you'd use proper voxelization
        voxel_feats = self.mlp(points)

        # Create pseudo-BEV features (simplified)
        # Map to grid using spatial hashing
        grid_size = self.voxel_grid_size[0] * self.voxel_grid_size[1]
        bev_features = torch.zeros(
            B, self.feature_dim,
            self.voxel_grid_size[0],
            self.voxel_grid_size[1],
            device=points.device
        )

        # Aggregate points to grid (simplified)
        for b in range(B):
            mask = batch_indices == b
            if mask.sum() > 0:
                voxel_feats_b = voxel_feats[mask]
                points_b = points[mask]

                # Grid indices
                gx = torch.clamp(points_b[:, 0].long(), 0, self.voxel_grid_size[0] - 1)
                gy = torch.clamp(points_b[:, 1].long(), 0, self.voxel_grid_size[1] - 1)

                # Simple max pooling per cell
                for i in range(gx.shape[0]):
                    bev_features[b, :, gx[i], gy[i]] = torch.max(
                        bev_features[b, :, gx[i], gy[i]],
                        voxel_feats_b[i]
                    )

        return bev_features


class CrossAttention(nn.Module):
    """Cross-attention between image and LiDAR features."""

    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Cross attention with residual connections."""
        # query: (B, N, C)
        attn_out, _ = self.cross_attn(query, key, value)
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class TransFuser(nn.Module):
    """
    TransFuser: Transformer-based fusion for BEV perception.

    Fuses image and LiDAR features to produce BEV (Bird's Eye View) features.
    """

    def __init__(
        self,
        image_feature_dim: int = 256,
        lidar_feature_dim: int = 256,
        bev_feature_dim: int = 256,
        num_cameras: int = 3,
        transformer_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()

        self.image_feature_dim = image_feature_dim
        self.lidar_feature_dim = lidar_feature_dim
        self.bev_feature_dim = bev_feature_dim
        self.num_cameras = num_cameras

        # Encoders
        self.image_encoder = ImageEncoder(feature_dim=image_feature_dim)
        self.lidar_encoder = LiDAREncoder(feature_dim=lidar_feature_dim)

        # Projection to common dimension
        self.image_proj = nn.Conv2d(image_feature_dim, bev_feature_dim, kernel_size=1)
        self.lidar_proj = nn.Conv2d(lidar_feature_dim, bev_feature_dim, kernel_size=1)

        # Transformer layers for fusion
        self.transformer_layers = nn.ModuleList([
            CrossAttention(bev_feature_dim, num_heads)
            for _ in range(transformer_layers)
        ])

        # Output projection
        self.output_proj = nn.Conv2d(bev_feature_dim, bev_feature_dim, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
        lidar_points: Optional[torch.Tensor] = None,
        egopose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: (B, N, 3, H, W) - Multi-camera images
            lidar_points: (B, N_points, 4) - LiDAR points (x, y, z, intensity)
            egopose: (B, 4, 4) - Ego vehicle pose

        Returns:
            Dictionary containing:
                - bev_features: (B, bev_feature_dim, H_bev, W_bev)
                - image_features: (B, N, image_feature_dim, H', W')
                - lidar_features: (B, lidar_feature_dim, H_bev, W_bev)
        """
        B, N, C, H, W = images.shape

        # Encode images
        image_features = self.image_encoder(images)  # (B, N, C, H', W')

        # Process each camera view
        image_bev_features = []
        for n in range(N):
            feat = image_features[:, n]  # (B, C, H', W')
            feat = self.image_proj(feat)  # (B, bev_dim, H', W')
            image_bev_features.append(feat)

        # Average across cameras
        image_bev = torch.stack(image_bev_features, dim=1).mean(dim=1)  # (B, bev_dim, H', W')

        # Encode LiDAR
        if lidar_points is not None:
            lidar_bev = self.lidar_encoder(lidar_points)  # (B, lidar_dim, H_bev, W_bev)
            lidar_bev = self.lidar_proj(lidar_bev)
        else:
            # If no LiDAR, just use image features
            lidar_bev = torch.zeros_like(image_bev)

        # Resize to match if needed
        if image_bev.shape[2:] != lidar_bev.shape[2:]:
            image_bev = F.interpolate(
                image_bev,
                size=lidar_bev.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Flatten for transformer
        B_feat, C_feat, H_feat, W_feat = lidar_bev.shape
        lidar_flat = lidar_bev.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        image_flat = image_bev.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Cross-attention fusion
        fused = image_flat
        for layer in self.transformer_layers:
            fused = layer(fused, lidar_flat, lidar_flat)

        # Reshape back to BEV
        fused = fused.permute(0, 2, 1).reshape(B_feat, C_feat, H_feat, W_feat)

        # Final projection
        bev_features = self.output_proj(fused)

        return {
            "bev_features": bev_features,
            "image_features": image_features,
            "lidar_features": lidar_bev,
        }


class BEVEncoder(nn.Module):
    """BEV feature encoder for downstream tasks."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.output_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """Encode BEV features."""
        encoded = self.encoder(bev_features)
        return self.output_conv(encoded)
