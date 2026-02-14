"""Evaluation script for GoalFlow."""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goalflow.config import load_config
from goalflow.models.goal_flow_model import create_model
from goalflow.data import SimulatedGoalFlowDataset
from goalflow.utils.metrics import MetricsComputer


def evaluate(config_path: str = None, checkpoint_path: str = None):
    """Main evaluation function."""
    # Load config
    if config_path is None:
        config = load_config()
    else:
        config = load_config(config_path)

    # Create model
    print("Creating model...")
    model = create_model(config)
    model.eval()

    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

    # Create dataset
    print("Creating dataset...")
    dataset = SimulatedGoalFlowDataset(
        num_samples=100,
        num_cameras=config.get("data", {}).get("num_cameras", 3),
    )

    # Metrics computer
    metrics_computer = MetricsComputer()

    # Evaluate
    print("Evaluating...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]

            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].unsqueeze(0).to(device)

            # Forward
            outputs = model(
                images=batch["images"],
                lidar_points=batch.get("lidar_points"),
                ego_state=batch.get("ego_state"),
            )

            # Get predicted trajectory
            pred_traj = outputs["best_trajectory"].cpu().numpy()[0]
            gt_traj = batch["trajectory"].cpu().numpy()[0]

            # Update metrics
            metrics_computer.update(
                predicted_trajectories=pred_traj.unsqueeze(0),
                ground_truth_trajectories=gt_traj.unsqueeze(0),
            )

            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{len(dataset)}")

    # Compute final metrics
    metrics = metrics_computer.compute()

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint)
