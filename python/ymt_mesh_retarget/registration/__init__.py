"""Mesh Registration Package.

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
# Alignment functions
from .alignment import (
    calculate_alignment_transform,
    calculate_alignment_transform_rbf,
    get_joint_tree,
    match_joint_trees,
)

# ---------------------------------------------------------------------------
# Internal API - Classes and functions used internally
# These are exported for advanced users and custom implementation needs
# ---------------------------------------------------------------------------
# Core data classes
from .core import (
    BoneNode,
    CorrespondencePoint,
    JointNode,
    MappingNode,
    MappingResult,
    RaycastResult,
    RegistrationOptions,
    TriangleWeightIndex,
)

# Geometry functions
from .geometry import rand_cone_vector, ray_triangle_intersection, triangle_interpolation
from .main import MeshRegistration, find_correspondence_pairs, visualize_correspondences

# Mapping functions
from .mapping import create_optimized_correspondence_points, get_mapping_points

# Raycast functions
from .raycast import perform_raycast

# Utility functions
from .utils import (
    find_root_joints,
    get_default_registration_options,
    get_matched_info,
    match_joint_positions,
    scale_joint_hierarchy_to_mesh,
    validate_registration_options,
)
from .weights import get_weight_distance


# List of public API elements
__all__ = [
    "BoneNode",
    "CorrespondencePoint",
    "JointNode",
    # Internal API - For advanced users and custom implementations
    # Core data classes
    "MappingNode",
    "MappingResult",
    # Public API - Primary classes and functions for most users
    "MeshRegistration",
    "RaycastResult",
    "RegistrationOptions",
    "TriangleWeightIndex",
    # Alignment functions
    "calculate_alignment_transform",
    "calculate_alignment_transform_rbf",
    "create_optimized_correspondence_points",
    "find_correspondence_pairs",
    "find_correspondence_using_skeleton",
    "find_root_joints",
    "get_default_registration_options",
    "get_joint_tree",
    # Mapping functions
    "get_mapping_points",
    # Utility functions
    "get_matched_info",
    "get_weight_distance",
    "match_joint_positions",
    "match_joint_trees",
    # Raycast functions
    "perform_raycast",
    # Geometry functions
    "rand_cone_vector",
    "ray_triangle_intersection",
    "scale_joint_hierarchy_to_mesh",
    "triangle_interpolation",
    "validate_registration_options",
    "visualize_correspondences",
]
