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
- Raycast-based point matching
- Joint-based alignment
"""

from .core import (
    CorrespondencePoint,
    MappingNode,
    MappingResult,
    JointNode,
    BoneNode,
    TriangleWeightIndex,
    RaycastResult,
    RegistrationOptions
)

from .geometry import (
    rand_cone_vector,
    ray_triangle_intersection,
    triangle_interpolation
)

from .alignment import (
    calculate_alignment_transform,
    calculate_alignment_transform_rbf,
    get_joint_tree,
    match_joint_trees
)

from .mapping import (
    get_mapping_points,
    create_optimized_correspondence_points,
    find_correspondence_using_skeleton
)

from .raycast import (
    build_embree_scene_from_source,
    perform_raycast
)

from .utils import (
    get_matched_info,
    find_root_joints,
    scale_joint_hierarchy_to_mesh,
    match_joint_positions,
    get_default_registration_options,
    validate_registration_options
)

from .weights import (
    get_weight_distance
)

# Reexport main class and functions from main module
from .main import (
    MeshRegistration,
    find_correspondence_pairs
)

__all__ = [
    # Core data classes
    'CorrespondencePoint',
    'MappingNode',
    'MappingResult',
    'JointNode',
    'BoneNode',
    'TriangleWeightIndex',
    'RaycastResult',
    'RegistrationOptions',
    
    # Main class and functions
    'MeshRegistration',
    'find_correspondence_pairs',
    
    # Alignment functions
    'calculate_alignment_transform',
    'calculate_alignment_transform_rbf',
    'get_joint_tree',
    'match_joint_trees',
    
    # Mapping functions
    'get_mapping_points',
    'create_optimized_correspondence_points',
    'find_correspondence_using_skeleton',
    
    # Geometry functions
    'rand_cone_vector',
    'ray_triangle_intersection', 
    'triangle_interpolation',
    
    # Raycast functions
    'build_embree_scene_from_source',
    'perform_raycast',
    
    # Utility functions
    'get_matched_info',
    'find_root_joints',
    'scale_joint_hierarchy_to_mesh',
    'match_joint_positions',
    'get_default_registration_options',
    'validate_registration_options',
    'get_weight_distance'
]