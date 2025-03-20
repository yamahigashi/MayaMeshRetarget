# -*- coding: utf-8 -*-
"""
Mesh Registration Package

This package implements functionality to find corresponding points between meshes with different topologies.
Based on the techniques from the "Skeleton-Aware Skin Weight Transfer" paper,
it generates correspondence point pairs for RBF interpolation between source and target meshes
with different vertex counts.

Main features:
- Correspondence search using skeletal information
- Vertex sampling and reduction
- Visualization of mesh registration results
"""

from .core import (
    CorrespondencePoint,
    MappingNode,
    MappingResult,
    JointNode,
    BoneNode,
    TriangleWeightIndex
)

from .geometry import (
    rand_cone_vector,
    ray_triangle_intersection,
    triangle_interpolation
)

from .raycast import (
    build_embree_scene_from_source
)

from .mapping import (
    create_optimized_correspondence_points
)

from .weights import (
    weight_transform,
    get_weight_distance
)

from .alignment import (
    calculate_alignment_transform,
    calculate_alignment_transform_rbf,
    match_joint_trees
)

# Reexport main class and functions from main module
from .main import (
    MeshRegistration,
    find_correspondence_pairs
)

__all__ = [
    # Core data structures
    "CorrespondencePoint", "MappingNode", "MappingResult", 
    "JointNode", "BoneNode", "TriangleWeightIndex",
    
    # Main functionality
    "MeshRegistration", "find_correspondence_pairs",
    
    # Utility functions
    "rand_cone_vector", "ray_triangle_intersection", "triangle_interpolation",
    "build_embree_scene_from_source", "create_optimized_correspondence_points",
    "weight_transform", "get_weight_distance",
    "calculate_alignment_transform", "calculate_alignment_transform_rbf", "match_joint_trees"
]