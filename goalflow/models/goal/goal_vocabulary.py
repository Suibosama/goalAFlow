"""Goal vocabulary: clustering trajectory endpoints."""

import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple


class GoalVocabulary:
    """
    Goal vocabulary for trajectory endpoints.

    Clusters training trajectory endpoints to create a discrete set of
    candidate goal positions.
    """

    def __init__(
        self,
        num_clusters: int = 256,
        cluster_method: str = "kmeans",
        cluster_dims: int = 3,  # x, y, heading
    ):
        """
        Args:
            num_clusters: Number of cluster centers (vocabulary size)
            cluster_method: Clustering method ("kmeans")
            cluster_dims: Number of dimensions for clustering (2 for x,y, 3 for x,y,theta)
        """
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        self.cluster_dims = cluster_dims

        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_model: Optional[KMeans] = None

    def build(
        self,
        trajectories: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build vocabulary from trajectory endpoints.

        Args:
            trajectories: (N, T, D) or (N, D) - trajectory endpoints
                         D=2 for (x,y), D=3 for (x,y,theta)
            sample_weights: (N,) - weights for each sample
        """
        # Extract endpoints
        if trajectories.ndim == 3:
            # (N, T, D) -> (N, D)
            endpoints = trajectories[:, -1, :]
        else:
            endpoints = trajectories

        # Ensure correct dimensions
        if endpoints.shape[1] < self.cluster_dims:
            # Pad with zeros if needed
            padded = np.zeros((endpoints.shape[0], self.cluster_dims))
            padded[:, :endpoints.shape[1]] = endpoints
            endpoints = padded

        # Fit clustering
        if self.cluster_method == "kmeans":
            self.cluster_model = KMeans(
                n_clusters=self.num_clusters,
                random_state=42,
                n_init=10,
            )
            self.cluster_model.fit(endpoints, sample_weight=sample_weights)
            self.cluster_centers = self.cluster_model.cluster_centers_
        else:
            raise ValueError(f"Unknown cluster method: {self.cluster_method}")

    def encode(self, positions: np.ndarray) -> np.ndarray:
        """
        Encode positions to vocabulary indices.

        Args:
            positions: (N, D) positions

        Returns:
            indices: (N,) cluster indices
        """
        if self.cluster_model is None:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        # Ensure dimensions match
        if positions.shape[1] < self.cluster_dims:
            padded = np.zeros((positions.shape[0], self.cluster_dims))
            padded[:, :positions.shape[1]] = positions
            positions = padded

        return self.cluster_model.predict(positions)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode vocabulary indices to positions.

        Args:
            indices: (N,) cluster indices

        Returns:
            positions: (N, D) positions
        """
        if self.cluster_centers is None:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        return self.cluster_centers[indices]

    def get_all_centers(self) -> np.ndarray:
        """Get all cluster centers."""
        if self.cluster_centers is None:
            raise RuntimeError("Vocabulary not built. Call build() first.")
        return self.cluster_centers

    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        np.savez(
            path,
            cluster_centers=self.cluster_centers,
            num_clusters=self.num_clusters,
            cluster_dims=self.cluster_dims,
        )

    def load(self, path: str) -> None:
        """Load vocabulary from file."""
        data = np.load(path, allow_pickle=True)
        self.cluster_centers = data["cluster_centers"]
        self.num_clusters = int(data["num_clusters"])
        self.cluster_dims = int(data["cluster_dims"])

        # Rebuild model
        self.cluster_model = KMeans(
            n_clusters=self.num_clusters,
            random_state=42,
            n_init=10,
        )
        # We don't refit the model since we just need predict functionality
        # The cluster centers are already set


def build_goal_vocabulary(
    trajectories: List[np.ndarray],
    num_clusters: int = 256,
) -> GoalVocabulary:
    """
    Build goal vocabulary from a list of trajectories.

    Args:
        trajectories: List of trajectories, each (T, 2) or (T, 3)
        num_clusters: Number of clusters

    Returns:
        GoalVocabulary instance
    """
    # Concatenate all endpoints
    endpoints = []
    for traj in trajectories:
        endpoints.append(traj[-1][:2])  # x, y

    endpoints = np.array(endpoints)

    # Build vocabulary
    vocab = GoalVocabulary(num_clusters=num_clusters, cluster_dims=2)
    vocab.build(endpoints)

    return vocab
