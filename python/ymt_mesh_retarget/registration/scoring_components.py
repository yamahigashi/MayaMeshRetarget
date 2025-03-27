"""Scoring components for mesh registration.

This module provides various scoring methods for evaluating correspondence points.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class IScoringComponent(ABC):
    """Abstract base class for scoring components used in correspondence point evaluation."""

    @abstractmethod
    def compute_score(self, context: dict[str, Any]) -> float:
        """Compute a score based on the provided context.

        Args:
            context: Dictionary containing contextual information for scoring

        Returns:
            float: Score value (higher is better)
        """
        pass

    def get_weight(self) -> float:
        """Get the weight of this scoring component.

        Returns:
            float: Weight value for this component
        """
        return 1.0


@dataclass
class DistanceScoring(IScoringComponent):
    """Scoring component based on spatial distance."""

    weight: float = 1.0

    def compute_score(self, context: dict[str, Any]) -> float:
        """Compute score based on distance.

        Args:
            context: Dictionary containing 'distance' key

        Returns:
            float: Score value (higher is better)
        """
        distance = context.get('distance', float('inf'))
        # Convert distance to score (inverse relationship - closer is better)
        if distance <= 0.0 or distance == float('inf'):
            return 0.0

        # Exponential falloff for distance - closer points score higher
        return self.weight * np.exp(-distance)

    def get_weight(self) -> float:
        """Get the weight of this scoring component."""
        return self.weight


@dataclass
class RayQualityScoring(IScoringComponent):
    """Scoring component based on ray quality."""

    weight: float = 0.5

    def compute_score(self, context: dict[str, Any]) -> float:
        """Compute score based on ray quality.

        Args:
            context: Dictionary containing 'raycast_result' key

        Returns:
            float: Score value (higher is better)
        """
        raycast_result = context.get('raycast_result')
        if raycast_result is None:
            return 0.0

        ray_weight = getattr(raycast_result, 'weight', 0.0)
        relate_distance = getattr(raycast_result, 'relate_distance', 1.0)

        # Ray quality score - higher weight and lower relative distance is better
        return self.weight * ray_weight * (1.0 - min(1.0, relate_distance))

    def get_weight(self) -> float:
        """Get the weight of this scoring component."""
        return self.weight


@dataclass
class NormalScoring(IScoringComponent):
    """Scoring component based on normal vector similarity."""

    weight: float = 0.8

    def compute_score(self, context: dict[str, Any]) -> float:
        """Compute score based on normal vector similarity.

        Args:
            context: Dictionary containing 'normal_src' and 'normal_tar' keys

        Returns:
            float: Score value (higher is better)
        """
        normal_src = context.get('normal_src')
        normal_tar = context.get('normal_tar')

        if normal_src is None or normal_tar is None:
            return 0.0

        # Normalize vectors if they aren't already
        normal_src_norm = np.linalg.norm(normal_src)
        normal_tar_norm = np.linalg.norm(normal_tar)

        if normal_src_norm < 1e-6 or normal_tar_norm < 1e-6:
            return 0.0

        normal_src = normal_src / normal_src_norm
        normal_tar = normal_tar / normal_tar_norm

        # Cosine similarity [-1, 1] -> remap to [0, 1]
        cosine_similarity = np.dot(normal_src, normal_tar)
        score = (cosine_similarity + 1.0) * 0.5

        return self.weight * score

    def get_weight(self) -> float:
        """Get the weight of this scoring component."""
        return self.weight


@dataclass
class WeightVectorScoring(IScoringComponent):
    """Scoring component based on weight vector similarity."""

    weight: float = 0.7

    def compute_score(self, context: dict[str, Any]) -> float:
        """Compute score based on weight vector similarity.

        Args:
            context: Dictionary containing 'weights_src' and 'weights_tar' keys

        Returns:
            float: Score value (higher is better)
        """
        weights_src = context.get('weights_src')
        weights_tar = context.get('weights_tar')

        if weights_src is None or weights_tar is None:
            return 0.0

        # Ensure weights have the same dimensions
        if len(weights_src) != len(weights_tar):
            # Could interpolate or pad, but for simplicity we'll return 0
            return 0.0

        # Normalize weight vectors (L2 norm)
        weights_src_norm = np.linalg.norm(weights_src)
        weights_tar_norm = np.linalg.norm(weights_tar)

        if weights_src_norm < 1e-6 or weights_tar_norm < 1e-6:
            return 0.0

        weights_src_normalized = weights_src / weights_src_norm
        weights_tar_normalized = weights_tar / weights_tar_norm

        # Cosine similarity between weight vectors
        cosine_similarity = np.dot(weights_src_normalized, weights_tar_normalized)
        score = max(0.0, cosine_similarity)  # Only positive similarity contributes

        return self.weight * score

    def get_weight(self) -> float:
        """Get the weight of this scoring component."""
        return self.weight


@dataclass
class LaplacianScoring(IScoringComponent):
    """Scoring component based on Laplacian coordinate similarity."""

    weight: float = 0.6

    def compute_score(self, context: dict[str, Any]) -> float:
        """Compute score based on Laplacian similarity.

        Args:
            context: Dictionary containing 'laplacian_src' and 'laplacian_tar' keys

        Returns:
            float: Score value (higher is better)
        """
        laplacian_src = context.get('laplacian_src')
        laplacian_tar = context.get('laplacian_tar')

        if laplacian_src is None or laplacian_tar is None:
            return 0.0

        # Normalize Laplacian vectors
        lap_src_norm = np.linalg.norm(laplacian_src)
        lap_tar_norm = np.linalg.norm(laplacian_tar)

        if lap_src_norm < 1e-6 or lap_tar_norm < 1e-6:
            return 0.0

        lap_src_normalized = laplacian_src / lap_src_norm
        lap_tar_normalized = laplacian_tar / lap_tar_norm

        # Compute similarity between normalized Laplacian vectors
        # Using cosine similarity for direction and magnitude difference
        cosine_similarity = np.dot(lap_src_normalized, lap_tar_normalized)
        # Scale to [0, 1] range
        score = (cosine_similarity + 1.0) * 0.5

        return self.weight * score

    def get_weight(self) -> float:
        """Get the weight of this scoring component."""
        return self.weight


def create_default_scoring_components() -> list[IScoringComponent]:
    """Create a default list of scoring components with standard weights.

    Returns:
        list[IScoringComponent]: List of default scoring components
    """
    return [
        DistanceScoring(weight=1.0),
        RayQualityScoring(weight=0.5),
        NormalScoring(weight=0.8),
        WeightVectorScoring(weight=0.7),
        LaplacianScoring(weight=0.6),
    ]
