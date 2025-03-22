# -*- coding: utf-8 -*-
"""
Core data structures and common functionality for mesh registration.

This module defines the fundamental data classes used throughout the mesh registration process.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from maya.api import OpenMaya as om


@dataclass
class CorrespondencePoint:
    """Data class for storing correspondence point information
    
    This data class stores vertex indices instead of positions to allow for
    more efficient coordinate access and transformation through the mesh function set.
    """
    source_index: int  # Vertex index in source mesh
    target_index: int  # Vertex index in target mesh
    weight: float = 1.0  # Weight (confidence) of the correspondence point


@dataclass
class MappingNode:
    """Data class for storing mapping node information"""
    point: np.ndarray  # Position of the point
    bone_index: int = -1  # Bone index
    distance: np.float64 = 0.0  # Distance
    weight: float = 0.0  # Weight


@dataclass
class MappingResult:
    """Data class for storing mapping results"""
    vertex_index: int = -1  # Vertex index
    node_array: List[MappingNode] = None  # List of mapping nodes

    def __post_init__(self):
        if self.node_array is None:
            self.node_array = []


@dataclass
class JointNode:
    """Data class for storing joint information"""
    path: om.MDagPath  # DAG path to the joint
    index: int = 0  # Index in the joint array
    detail_name: str = ""  # Short name of the joint
    position: np.ndarray = np.zeros(3)  # Position of the joint
    matrix: tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0,
                                                                                                                                     0.0, 1.0, 0.0, 0.0,
                                                                                                                                     0.0, 0.0, 1.0, 0.0,
                                                                                                                                     0.0, 0.0, 0.0, 1.0)


@dataclass
class BoneNode:
    """Data class for storing bone information"""
    start_joint_index: int = -1  # Index of the start joint
    end_joint_index: int = -1  # Index of the end joint


@dataclass
class TriangleWeightIndex:
    """Data class for triangle weight index information"""
    weight_distance: float = 10000.0  # Weight distance
    triangle_index: int = -1  # Triangle index


@dataclass
class RaycastResult:
    """Data class for storing ray hit information"""
    from_point: np.ndarray  # (3, ) - Ray origin
    point: np.ndarray  # (3, ) - Hit point
    triangle_index: int  # Triangle index
    weight: float  # Weight
    relate_distance: float
