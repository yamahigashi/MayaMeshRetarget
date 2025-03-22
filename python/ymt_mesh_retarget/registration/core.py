# -*- coding: utf-8 -*-
"""
Core data structures and common functionality for mesh registration.

This module defines the fundamental data classes used throughout the mesh registration process.
"""

from dataclasses import dataclass, field
from typing import (
    List, 
    Optional, 
    Union, 
    Tuple, 
    Dict,
    TypeVar,
    Protocol,
    Callable,
    Any
)

import numpy as np
from numpy.typing import NDArray
from maya.api import OpenMaya as om


# Type aliases for improved readability
Vector3 = NDArray[np.float64]  # 3D vector (x, y, z)
Matrix4x4 = Tuple[float, float, float, float, 
                 float, float, float, float, 
                 float, float, float, float, 
                 float, float, float, float]  # 4x4 transformation matrix

VertexIndex = int
BoneIndex = int
TriangleIndex = int

# Identity matrix constant
IDENTITY_MATRIX: Matrix4x4 = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
)


class Kernel(Protocol):
    """Protocol for kernel functions used in RBF interpolation"""
    def __call__(self, r: Union[float, NDArray[np.float64]], eps: float = 1.0) -> Union[float, NDArray[np.float64]]:
        """Calculate kernel value based on radius"""
        ...


@dataclass
class CorrespondencePoint:
    """Data class for storing correspondence point information
    
    This data class stores vertex indices instead of positions to allow for
    more efficient coordinate access and transformation through the mesh function set.
    
    Attributes:
        source_index: Vertex index in source mesh
        target_index: Vertex index in target mesh
        weight: Weight (confidence) of the correspondence point, between 0.0 and 1.0
    """
    source_index: VertexIndex  # Vertex index in source mesh
    target_index: VertexIndex  # Vertex index in target mesh
    weight: float = 1.0  # Weight (confidence) of the correspondence point

    def __post_init__(self) -> None:
        """Validate data after initialization"""
        if not isinstance(self.source_index, int) or self.source_index < 0:
            raise ValueError(f"Invalid source_index: {self.source_index}. Must be a non-negative integer.")
        if not isinstance(self.target_index, int) or self.target_index < 0:
            raise ValueError(f"Invalid target_index: {self.target_index}. Must be a non-negative integer.")
        if not 0.0 <= self.weight <= 1.0:
            # Clamp weight to valid range
            self.weight = max(0.0, min(1.0, self.weight))


@dataclass
class MappingNode:
    """Data class for storing mapping node information
    
    Attributes:
        point: 3D position of the point
        bone_index: Index of the associated bone (-1 if none)
        distance: Distance metric
        weight: Weight/influence value (0.0-1.0)
    """
    point: Vector3  # Position of the point
    bone_index: BoneIndex = -1  # Bone index
    distance: float = 0.0  # Distance
    weight: float = 0.0  # Weight

    def __post_init__(self) -> None:
        """Validate data after initialization"""
        if not isinstance(self.point, np.ndarray) or self.point.shape != (3,):
            raise ValueError(f"Invalid point: {self.point}. Must be a numpy array with shape (3,).")
        
        # Ensure numeric types
        self.distance = float(self.distance)
        self.weight = float(self.weight)
        
        # Clamp weight to valid range
        if not 0.0 <= self.weight <= 1.0:
            self.weight = max(0.0, min(1.0, self.weight))


@dataclass
class MappingResult:
    """Data class for storing mapping results for a vertex
    
    Attributes:
        vertex_index: Index of the vertex this mapping refers to
        node_array: List of mapping nodes associated with this vertex
    """
    vertex_index: VertexIndex  # Vertex index
    node_array: List[MappingNode] = field(default_factory=list)  # List of mapping nodes

    def add_node(self, node: MappingNode) -> None:
        """Add a mapping node to this result
        
        Args:
            node: The mapping node to add
        """
        self.node_array.append(node)
    
    def get_best_node(self) -> Optional[MappingNode]:
        """Get the mapping node with the highest weight
        
        Returns:
            The mapping node with the highest weight, or None if no nodes
        """
        if not self.node_array:
            return None
        return max(self.node_array, key=lambda node: node.weight)


@dataclass
class JointNode:
    """Data class for storing joint information
    
    Attributes:
        path: DAG path to the joint in Maya
        index: Index in the joint array
        detail_name: Full name of the joint
        position: World position of the joint
        matrix: World matrix of the joint
    """
    path: om.MDagPath  # DAG path to the joint
    index: int = 0  # Index in the joint array
    detail_name: str = ""  # Full name of the joint
    position: Vector3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # Position of the joint
    matrix: Matrix4x4 = IDENTITY_MATRIX  # World matrix of the joint
    
    @property
    def short_name(self) -> str:
        """Get the short name of the joint (without namespace or path)"""
        return self.detail_name.split("|")[-1].split(":")[-1]
    
    def get_matrix_as_om(self) -> om.MMatrix:
        """Convert the matrix tuple to an OpenMaya MMatrix
        
        Returns:
            OpenMaya MMatrix representation of the joint's transformation
        """
        return om.MMatrix(self.matrix)


@dataclass
class BoneNode:
    """Data class for storing bone information
    
    A bone connects two joints and is used for skinning weights and deformation.
    
    Attributes:
        start_joint_index: Index of the start joint in the joint array
        end_joint_index: Index of the end joint in the joint array
    """
    start_joint_index: int = -1  # Index of the start joint
    end_joint_index: int = -1  # Index of the end joint

    def is_valid(self) -> bool:
        """Check if this bone has valid joint indices
        
        Returns:
            True if both joint indices are valid (>= 0), False otherwise
        """
        return self.start_joint_index >= 0 and self.end_joint_index >= 0
    
    def get_direction(self, joints: List[JointNode]) -> Optional[Vector3]:
        """Calculate the direction vector of this bone
        
        Args:
            joints: List of joint nodes
            
        Returns:
            Normalized direction vector from start to end joint,
            or None if the bone is invalid or has zero length
        """
        if not self.is_valid():
            return None
            
        start_pos = joints[self.start_joint_index].position
        end_pos = joints[self.end_joint_index].position
        
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        
        if length < 1e-6:  # Avoid division by zero
            return None
            
        return direction / length


@dataclass
class TriangleWeightIndex:
    """Data class for triangle weight index information
    
    Used for weighted triangle calculations in mesh mapping.
    
    Attributes:
        weight_distance: Weight distance metric (lower is better)
        triangle_index: Index of the triangle
    """
    weight_distance: float = 10000.0  # Weight distance
    triangle_index: TriangleIndex = -1  # Triangle index


@dataclass
class RaycastResult:
    """Data class for storing ray hit information
    
    Records the result of a raycast operation, including hit point and related metadata.
    
    Attributes:
        from_point: Origin point of the ray
        point: Hit point where ray intersects geometry
        triangle_index: Index of the intersected triangle
        weight: Weight/confidence value for this hit
        relate_distance: Relative distance metric for ranking hits
    """
    from_point: Vector3  # (3, ) - Ray origin
    point: Vector3  # (3, ) - Hit point
    triangle_index: TriangleIndex  # Triangle index
    weight: float  # Weight
    relate_distance: float  # Relative distance metric
    
    def distance(self) -> float:
        """Calculate the distance between ray origin and hit point
        
        Returns:
            Euclidean distance from ray origin to hit point
        """
        return float(np.linalg.norm(self.point - self.from_point))


@dataclass
class RegistrationOptions:
    """Options for mesh registration process
    
    Centralizes configuration parameters for the registration process
    to improve maintainability and consistency.
    
    Attributes:
        sample_rate: Percentage of vertices to sample (0.0-1.0)
        sample_number: Number of sample rays per point
        sample_degree: Angular range for sampling (degrees)
        weight_decay: Weight decay coefficient
        align_spaces: Whether to align source and target spaces
        max_points_per_target: Maximum correspondence points per target vertex
        min_weight_threshold: Minimum weight threshold for correspondence
        distance_weight: Weight coefficient for distance in scoring
        ray_weight: Weight coefficient for ray quality in scoring
        max_triangles: Maximum triangles to process (-1 for unlimited)
        batch_size: Batch size for ray processing
        num_threads: Number of threads to use for parallel processing
        use_bvh: Whether to use BVH acceleration for standard raycast engine
    """
    sample_rate: float = 1.0
    sample_number: int = 32
    sample_degree: float = 45.0
    weight_decay: float = 2.0
    align_spaces: bool = True
    max_points_per_target: int = 3
    min_weight_threshold: float = 0.01
    distance_weight: float = 1.0
    ray_weight: float = 0.5
    max_triangles: int = -1
    batch_size: int = 1024
    num_threads: int = 4
    use_bvh: bool = True