"""Data loader for GoalFlow."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import random


class GoalFlowDataset(Dataset):
    """Dataset for GoalFlow training and evaluation."""

    def __init__(
        self,
        data_root: str = "./data",
        split: str = "train",
        num_cameras: int = 3,
        image_size: Tuple[int, int] = (256, 256),
        history_steps: int = 4,
        future_steps: int = 30,
        augmentation: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        self.num_cameras = num_cameras
        self.image_size = image_size
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.augmentation = augmentation

        # Load data (placeholder - would load from actual data)
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load dataset (placeholder implementation)."""
        # In practice, this would load from nuScenes or other dataset
        # For now, return empty list - will be populated with simulated data
        return []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        return self.data[idx]

    def _augment(self, sample: Dict) -> Dict:
        """Apply data augmentation."""
        if not self.augmentation:
            return sample

        # Random flip
        if random.random() < 0.5:
            sample["images"] = torch.flip(sample["images"], dims=[-1])
            sample["trajectory"][:, 0] *= -1
            sample["goal"][0] *= -1

        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            sample["trajectory"] = sample["trajectory"] @ rot_matrix.T
            sample["goal"] = sample["goal"] @ rot_matrix.T

        return sample


def create_dataloader(
    config: Dict,
    split: str = "train",
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloader from config."""
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    dataset = GoalFlowDataset(
        data_root=data_config.get("data_root", "./data"),
        split=split,
        num_cameras=data_config.get("num_cameras", 3),
        image_size=tuple(data_config.get("image_size", [256, 256])),
        history_steps=data_config.get("history_steps", 4),
        future_steps=data_config.get("future_steps", 30),
        augmentation=(
            data_config.get("augmentation", {}).get("enabled", False)
            if split == "train" else False
        ),
    )

    batch_size = training_config.get("batch_size", 8)
    num_workers = training_config.get("num_workers", 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=training_config.get("pin_memory", True),
    )


class SimulatedGoalFlowDataset(GoalFlowDataset):
    """Simulated dataset for testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_cameras: int = 3,
        image_size: Tuple[int, int] = (256, 256),
        history_steps: int = 4,
        future_steps: int = 30,
    ):
        super().__init__(
            data_root="",
            split="train",
            num_cameras=num_cameras,
            image_size=image_size,
            history_steps=history_steps,
            future_steps=future_steps,
            augmentation=False,
        )
        self.num_samples = num_samples
        self.data = self._generate_simulated_data()

    def _generate_simulated_data(self) -> List[Dict]:
        """Generate simulated data."""
        data = []
        for i in range(self.num_samples):
            # Generate random lidar points (x, y, z, intensity)
            lidar_points = torch.randn(1000, 4) * 10
            lidar_points[:, 2] = torch.rand(1000) * 3  # z between 0-3
            lidar_points[:, 3] = torch.rand(1000)  # intensity 0-1

            # Add voxel coordinates (room_x, room_y, room_z) for 7D input
            # Normalize to point cloud range
            room_x = (lidar_points[:, 0] - (-50)) / 100  # normalized to 0-1
            room_y = (lidar_points[:, 1] - (-50)) / 100
            room_z = (lidar_points[:, 2] - (-3)) / 6
            lidar_points_7d = torch.cat([
                lidar_points,
                room_x.unsqueeze(1),
                room_y.unsqueeze(1),
                room_z.unsqueeze(1),
            ], dim=1)  # (1000, 7)

            sample = {
                "images": torch.randn(
                    self.num_cameras, 3, self.image_size[0], self.image_size[1]
                ),
                "lidar_points": lidar_points_7d,
                "ego_state": torch.tensor([0.0, 0.0, 0.0, 5.0]),  # x, y, heading, speed
                "trajectory": torch.randn(self.future_steps, 2) * 10,
                "goal": torch.randn(2) * 20,
            }
            data.append(sample)
        return data
