"""Geometry utilities for GoalFlow."""

import numpy as np
import torch
from typing import Tuple, Optional


def convert_egopose_to_matrix(egopose: np.ndarray) -> np.ndarray:
    """
    Convert ego pose to transformation matrix.

    Args:
        egopose: (7,) [x, y, z, qw, qx, qy, qz] or (4, 4) transformation matrix

    Returns:
        (4, 4) transformation matrix
    """
    if egopose.shape == (4, 4):
        return egopose

    # Convert from quaternion
    x, y, z, qw, qx, qy, qz = egopose

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])

    # Transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Transform points using transformation matrix.

    Args:
        points: (N, D) points, D >= 3
        transform: (4, 4) transformation matrix

    Returns:
        Transformed points
    """
    if isinstance(points, np.ndarray):
        points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed = (transform @ points_h.T).T
        return transformed[:, :points.shape[1]]
    else:
        # PyTorch version
        points_h = torch.cat([points, torch.ones(*points.shape[:-1], 1, device=points.device)], dim=-1)
        transformed = (transform @ points_h.T).T
        return transformed[..., :points.shape[-1]]


def convert_spatial_to_bev(
    points: np.ndarray,
    bev_resolution: float = 0.5,
    bev_size: Tuple[int, int] = (200, 200),
    bev_range: Tuple[float, float, float, float] = (-50, -50, 50, 50),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert spatial coordinates to BEV grid indices.

    Args:
        points: (N, 2) or (N, 3) points in world coordinates
        bev_resolution: resolution in meters per cell
        bev_size: (H, W) BEV grid size
        bev_range: (x_min, y_min, x_max, y_max) in meters

    Returns:
        Tuple of (grid_x, grid_y) indices
    """
    x_min, y_min, x_max, y_max = bev_range

    grid_x = ((points[:, 0] - x_min) / (x_max - x_min) * bev_size[1]).astype(np.int32)
    grid_y = ((points[:, 1] - y_min) / (y_max - y_min) * bev_size[0]).astype(np.int32)

    # Clip to valid range
    grid_x = np.clip(grid_x, 0, bev_size[1] - 1)
    grid_y = np.clip(grid_y, 0, bev_size[0] - 1)

    return grid_x, grid_y


def compute_trajectory_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """
    Compute trajectory metrics.

    Args:
        predicted: (T, 2) predicted trajectory
        ground_truth: (T, 2) ground truth trajectory

    Returns:
        Dictionary of metrics
    """
    # Endpoint error
    endpoint_error = np.linalg.norm(predicted[-1] - ground_truth[-1])

    # ADE (Average Displacement Error)
    ade = np.linalg.norm(predicted - ground_truth, axis=1).mean()

    # FDE (Final Displacement Error)
    fde = endpoint_error

    return {
        "endpoint_error": endpoint_error,
        "ade": ade,
        "fde": fde,
    }
