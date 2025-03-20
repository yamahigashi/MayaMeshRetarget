# -*- coding: utf-8 -*-
"""
Mesh Registration Module

This module implements functionality to find corresponding points between meshes with different topologies.
Based on the techniques from the "Skeleton-Aware Skin Weight Transfer" paper,
it generates correspondence point pairs for RBF interpolation between source and target meshes
with different vertex counts.

This file is a compatibility layer for the refactored registration package.
It reexports the main functionality to maintain backward compatibility.
"""

from .registration.core import (
    CorrespondencePoint,
    MappingNode,
    MappingResult,
    JointNode,
    BoneNode,
    TriangleWeightIndex
)

from .registration.main import (
    MeshRegistration,
    find_correspondence_pairs
)

# Re-export utility functions for backward compatibility
from .registration.geometry import (
    rand_cone_vector,
    ray_triangle_intersection,
    triangle_interpolation
)

from .registration.raycast import (
    build_embree_scene_from_source
)

__all__ = [
    # Core data structures
    "CorrespondencePoint", "MappingNode", "MappingResult", 
    "JointNode", "BoneNode", "TriangleWeightIndex",
    
    # Main functionality
    "MeshRegistration", "find_correspondence_pairs",
    
    # Utility functions
    "rand_cone_vector", "ray_triangle_intersection", "triangle_interpolation",
    "build_embree_scene_from_source"
]