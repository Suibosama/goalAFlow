"""Evaluation metrics for GoalFlow."""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


def compute_snc(
    predicted_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
) -> float:
    """
    Scene Completeness (SNC): How well the predicted trajectory covers the scene.

    Higher is better.
    """
    # Simple version: overlap between predicted and ground truth
    gt_length = np.linalg.norm(ground_truth_trajectory[-1] - ground_truth_trajectory[0])
    pred_length = np.linalg.norm(predicted_trajectory[-1] - predicted_trajectory[0])

    # Ratio of lengths
    snc = min(gt_length / (pred_length + 1e-6), 1.0)
    return snc


def compute_sdac(
    predicted_trajectory: np.ndarray,
    drivability_map: np.ndarray,
) -> float:
    """
    Drivability Area Classification (SDAC): Whether trajectory is in drivable area.

    Higher is better (1.0 = fully drivable).
    """
    # Sample points along trajectory
    num_points = min(len(predicted_trajectory), 20)
    indices = np.linspace(0, len(predicted_trajectory) - 1, num_points).astype(int)

    in_drivable = 0
    for idx in indices:
        x, y = predicted_trajectory[idx]
        if drivability_map[int(y), int(x)] > 0:
            in_drivable += 1

    return in_drivable / num_points


def compute_sttc(
    predicted_trajectory: np.ndarray,
    agent_trajectories: np.ndarray,
    ego_position: np.ndarray,
    threshold: float = 3.0,
) -> float:
    """
    Time to Collision (STTC): Time until closest approach to any agent.

    Higher is better.
    """
    if len(agent_trajectories) == 0:
        return float("inf")

    min_ttc = float("inf")

    for agent_traj in agent_trajectories:
        # Compute closest approach
        for t in range(min(len(predicted_trajectory), len(agent_traj))):
            dist = np.linalg.norm(predicted_trajectory[t] - agent_traj[t])
            if dist < threshold:
                ttc = t * 0.5  # Assuming 2Hz
                min_ttc = min(min_ttc, ttc)

    return min_ttc if min_ttc != float("inf") else 0.0


def compute_scf(
    predicted_trajectory: np.ndarray,
    agent_trajectories: np.ndarray,
    threshold: float = 2.0,
) -> float:
    """
    Scene Collision Rate (SCF): Rate of collisions with agents.

    Lower is better.
    """
    if len(agent_trajectories) == 0:
        return 0.0

    collision = 0
    total_checks = 0

    for agent_traj in agent_trajectories:
        for t in range(min(len(predicted_trajectory), len(agent_traj))):
            dist = np.linalg.norm(predicted_trajectory[t] - agent_traj[t])
            if dist < threshold:
                collision += 1
            total_checks += 1

    return collision / max(total_checks, 1)


def compute_sep(
    predicted_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
) -> float:
    """
    Endpoint Error (SEP): Distance between predicted and GT endpoints.

    Lower is better.
    """
    return np.linalg.norm(predicted_trajectory[-1] - ground_truth_trajectory[-1])


def compute_spdm(
    predicted_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
) -> float:
    """
    Perceptual Distance Metric (SPDM): Average displacement error.

    Lower is better.
    """
    return np.linalg.norm(predicted_trajectory - ground_truth_trajectory, axis=1).mean()


def compute_all_metrics(
    predicted_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
    agent_trajectories: Optional[np.ndarray] = None,
    drivability_map: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all Navsim metrics.

    Args:
        predicted_trajectory: (T, 2) predicted trajectory
        ground_truth_trajectory: (T, 2) ground truth trajectory
        agent_trajectories: (N, T, 2) other agent trajectories
        drivability_map: (H, W) binary drivability map

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        "snc": compute_snc(predicted_trajectory, ground_truth_trajectory),
        "sep": compute_sep(predicted_trajectory, ground_truth_trajectory),
        "spdm": compute_spdm(predicted_trajectory, ground_truth_trajectory),
    }

    if drivability_map is not None:
        metrics["sdac"] = compute_sdac(predicted_trajectory, drivability_map)
    else:
        metrics["sdac"] = 1.0

    if agent_trajectories is not None:
        metrics["scf"] = compute_scf(predicted_trajectory, agent_trajectories)
        metrics["sttc"] = compute_sttc(
            predicted_trajectory, agent_trajectories,
            ground_truth_trajectory[0]
        )
    else:
        metrics["scf"] = 0.0
        metrics["sttc"] = float("inf")

    return metrics


class MetricsComputer:
    """Computes metrics for a batch of predictions."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.metrics = {
            "snc": [],
            "sdac": [],
            "sttc": [],
            "scf": [],
            "sep": [],
            "spdm": [],
        }

    def update(
        self,
        predicted_trajectories: np.ndarray,
        ground_truth_trajectories: np.ndarray,
        agent_trajectories: Optional[np.ndarray] = None,
        drivability_maps: Optional[np.ndarray] = None,
    ):
        """Update metrics for a batch."""
        B = predicted_trajectories.shape[0]

        for i in range(B):
            pred = predicted_trajectories[i]
            gt = ground_truth_trajectories[i]

            agents = agent_trajectories[i] if agent_trajectories is not None else None
            drivable = drivability_maps[i] if drivability_maps is not None else None

            m = compute_all_metrics(pred, gt, agents, drivable)

            for key, value in m.items():
                if not np.isnan(value) and not np.isinf(value):
                    self.metrics[key].append(value)

    def compute(self) -> Dict[str, float]:
        """Compute mean metrics."""
        result = {}
        for key, values in self.metrics.items():
            if values:
                result[key] = np.mean(values)
            else:
                result[key] = 0.0
        return result
