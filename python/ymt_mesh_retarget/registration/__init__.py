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

Usage Examples:
--------------
Basic usage:

    from ymt_mesh_retarget.registration import find_correspondence_pairs
    
    # Find correspondence points between two meshes
    source_points, target_points = find_correspondence_pairs(
        source_mesh="sourceModel",
        target_mesh="targetModel"
    )

Advanced usage with custom options:

    from ymt_mesh_retarget.registration import (
        MeshRegistration, 
        get_default_registration_options,
        RegistrationOptions
    )
    
    # Create and customize options
    options = get_default_registration_options()
    options.sample_rate = 0.3  # Use 30% of vertices
    options.num_threads = 8    # Use 8 threads for processing
    
    # Create registration object
    registration = MeshRegistration("sourceModel", "targetModel", options)
    
    # Find correspondence pairs
    source_points, target_points = registration.find_correspondence_pairs()
    
    # Visualize the results
    registration.visualize_correspondences()
"""

# ---------------------------------------------------------------------------
# Public API - Primary classes and functions for external use
# ---------------------------------------------------------------------------
from .core import (
    CorrespondencePoint,
    RegistrationOptions
)

from .utils import (
    get_default_registration_options,
    validate_registration_options
)

from .main import (
    MeshRegistration,
    find_correspondence_pairs,
    visualize_correspondences
)

# ---------------------------------------------------------------------------
# Internal API - Classes and functions used internally
# These are exported for advanced users and custom implementation needs
# ---------------------------------------------------------------------------
# Core data classes
from .core import (
    MappingNode,
    MappingResult,
    JointNode,
    BoneNode,
    TriangleWeightIndex,
    RaycastResult
)

# Alignment functions
from .alignment import (
    calculate_alignment_transform,
    calculate_alignment_transform_rbf,
    get_joint_tree,
    match_joint_trees
)

# Mapping functions
from .mapping import (
    get_mapping_points,
    create_optimized_correspondence_points,
    find_correspondence_using_skeleton
)

# Geometry functions
from .geometry import (
    rand_cone_vector,
    ray_triangle_intersection, 
    triangle_interpolation
)

# Raycast functions
from .raycast import (
    build_embree_scene_from_source,
    perform_raycast
)

# Utility functions
from .utils import (
    get_matched_info,
    find_root_joints,
    scale_joint_hierarchy_to_mesh,
    match_joint_positions
)

from .weights import (
    get_weight_distance
)

# List of public API elements
__all__ = [
    # Public API - Primary classes and functions for most users
    'MeshRegistration',
    'find_correspondence_pairs',
    'visualize_correspondences',
    'CorrespondencePoint',
    'RegistrationOptions',
    'get_default_registration_options',
    'validate_registration_options',
    
    # Internal API - For advanced users and custom implementations
    # Core data classes
    'MappingNode',
    'MappingResult',
    'JointNode',
    'BoneNode',
    'TriangleWeightIndex',
    'RaycastResult',
    
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
    'get_weight_distance'
]