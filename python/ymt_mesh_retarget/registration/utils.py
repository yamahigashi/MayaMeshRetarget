"""Utility functions for mesh registration.

This module provides common utility functions used throughout the registration process.
"""

import multiprocessing
import typing

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from numpy.typing import NDArray

from ..logger import logger
from ..types import ensure_list
from ..util import get_short_name, timeit
from .core import JointNode, RegistrationOptions, Vector3


if typing.TYPE_CHECKING:
    from ..objects import MeshObject


@timeit
def get_matched_info(
    src_joint_group: list[JointNode],
    tar_joint_group: list[JointNode],
) -> tuple[list[int], list[int], list[str]]:
    """Get matched joint indices between source and target joint groups.

    Finds joints with matching names between source and target hierarchies.

    Args:
        src_joint_group: Source joint group
        tar_joint_group: Target joint group

    Returns:
        A tuple containing:
        - List of matched source joint indices
        - List of matched target joint indices
        - List of matched joint short names
    """
    matched_pairs = []
    matched_joint_names = []

    for i, src_joint in enumerate(src_joint_group):
        src_name = get_short_name(src_joint.detail_name)
        for j, tar_joint in enumerate(tar_joint_group):
            tar_name = get_short_name(tar_joint.detail_name)
            if src_name == tar_name:
                matched_pairs.append((i, j))
                matched_joint_names.append(tar_name)
                break

    src_indices = [pair[0] for pair in matched_pairs]
    tar_indices = [pair[1] for pair in matched_pairs]

    return ensure_list(src_indices), ensure_list(tar_indices), matched_joint_names


def find_root_joints(joint_paths: typing.Union[list[om.MDagPath], list[JointNode]]) -> list[om.MDagPath]:
    """Find root joints in a joint hierarchy.

    Determines which joints are at the root level (no joint parents).
    If no true root joints are found, it returns the joints with the shallowest hierarchy depth.

    Args:
        joint_paths: List of joint DAG paths or JointNode objects

    Returns:
        List of root joint DAG paths
    """
    import math

    root_joints = []
    for _i, path in enumerate(joint_paths):
        if isinstance(path, JointNode):
            pp = path.path
            if pp is not None:
                path = pp
            else:
                path = om.MDagPath.getAPathTo(path.detail_name)

        if path.length() == 1 or cmds.listRelatives(path.fullPathName(), parent=True, type="joint") is None:
            root_joints.append(path)

    if not root_joints:
        min_depth = math.inf
        root_joints = []
        for path in joint_paths:
            if isinstance(path, JointNode):
                pp = path.path
                path = pp if pp is not None else om.MDagPath.getAPathTo(path.detail_name)

            min_depth = min(min_depth, path.fullPathName().count("|"))
            if path.fullPathName().count("|") == min_depth:
                root_joints.append(path)

    return root_joints


def scale_joint_hierarchy_to_mesh(
    joint_group: list[JointNode],
    source_mesh: "MeshObject",
    target_mesh: "MeshObject",
) -> None:
    """Scale a joint hierarchy to match a target mesh.

    Scales the source joint hierarchy to match the proportions of the target mesh
    based on bounding box comparison.

    Args:
        joint_group: List of source joint nodes to scale
        source_mesh: Source mesh object
        target_mesh: Target mesh object to match scale with
    """
    # Get bounding boxes
    src_bbox = cmds.exactWorldBoundingBox(source_mesh.name)  # minx, miny, minz, maxx, maxy, maxz
    tar_bbox = cmds.exactWorldBoundingBox(target_mesh.name)

    # Calculate scale factors
    sx = abs(tar_bbox[3] - tar_bbox[0]) / abs(src_bbox[3] - src_bbox[0])
    sy = abs(tar_bbox[4] - tar_bbox[1]) / abs(src_bbox[4] - src_bbox[1])
    sz = abs(tar_bbox[5] - tar_bbox[2]) / abs(src_bbox[5] - src_bbox[2])

    # Find root joints
    root_joints = find_root_joints(joint_group)

    # Apply scale to root joints and disable segment scale compensate
    for root_path in root_joints:
        current_scale = cmds.xform(root_path.fullPathName(), query=True, scale=True)
        new_scale = (sx * current_scale[0], sy * current_scale[1], sz * current_scale[2])
        cmds.xform(root_path.fullPathName(), scale=new_scale)

        # Disable segment scale compensate on child joints
        for kid in cmds.listRelatives(root_path.fullPathName(), children=True, type="joint", fullPath=True) or []:
            cmds.setAttr(f"{kid}.segmentScaleCompensate", 0)


def match_joint_positions(
    src_joint_group: list[JointNode],
    tar_joint_group: list[JointNode],
) -> list[int]:
    """Match joint positions between source and target.

    Translates matching source joints to the positions of their target counterparts.

    Args:
        src_joint_group: Source joint group
        tar_joint_group: Target joint group

    Returns:
        List of source joint indices that were matched
    """
    # Get matched joints
    src_indices, tar_indices, _ = get_matched_info(src_joint_group, tar_joint_group)

    # Match joint translations
    for s_i, t_i in zip(src_indices, tar_indices):
        src_joint = src_joint_group[s_i]
        tar_joint = tar_joint_group[t_i]

        tar_pos = cmds.xform(tar_joint.path.fullPathName(), query=True, translation=True, worldSpace=True)
        cmds.xform(src_joint.path.fullPathName(), translation=tar_pos, worldSpace=True)

    return src_indices


def get_default_registration_options() -> RegistrationOptions:
    """Get default registration options.

    Returns a RegistrationOptions object initialized with sensible defaults
    for most common use cases.

    Returns:
        Default RegistrationOptions
    """

    # Determine optimal number of threads based on CPU count
    # Use a reasonable default based on available cores, but limit to avoid system overload
    cpu_count = multiprocessing.cpu_count()
    default_threads = max(2, min(cpu_count - 1, 8))

    return RegistrationOptions(
        sample_count=3000,  # Sample 3000 vertieces
        sample_number=32,  # 32 rays per point
        sample_degree=45.0,  # 45-degree sampling angle
        weight_decay=2.0,  # Standard weight decay
        align_spaces=True,  # Align source and target spaces
        max_points_per_target=3,  # 3 correspondence points per target vertex
        min_weight_threshold=0.01,  # Minimum weight threshold
        distance_weight=1.0,  # Distance weight
        ray_weight=0.5,  # Ray weight
        use_scoring_components=False,  # Use legacy scoring by default
        scoring_components=[],  # Empty scoring components list (initialized on demand)
        use_normal_scoring=True,  # Enable normal scoring by default
        use_weight_scoring=False,  # Disable weight scoring by default (requires skinning)
        use_laplacian_scoring=False,  # Disable laplacian scoring by default
        precompute_mesh_data=True,  # Precompute mesh data for better performance
        max_triangles=-1,  # No triangle limit
        batch_size=1024,  # Process 1024 rays at a time
        num_threads=default_threads,  # Use CPU core count-based threading
        use_bvh=True,  # Use BVH acceleration by default
    )


def validate_registration_options(options: RegistrationOptions) -> RegistrationOptions:
    """Validate and normalize registration options.

    Ensures all options are within valid ranges and normalizes values.
    Also initializes scoring components if requested.

    Args:
        options: RegistrationOptions to validate

    Returns:
        Validated and normalized RegistrationOptions
    """
    from .scoring_components import create_default_scoring_components

    options.sample_count = max(100, min(10000, options.sample_count))
    options.sample_number = max(4, min(128, options.sample_number))
    options.sample_degree = max(1.0, min(180.0, options.sample_degree))
    options.weight_decay = max(0.1, options.weight_decay)
    options.max_points_per_target = max(1, min(10, options.max_points_per_target))
    options.min_weight_threshold = max(0.0, min(1.0, options.min_weight_threshold))
    options.batch_size = max(64, min(4096, options.batch_size))

    # Validate thread count (1 to max available CPUs)
    cpu_count = multiprocessing.cpu_count()
    options.num_threads = max(1, min(cpu_count, options.num_threads))

    # Initialize scoring components if requested
    if options.use_scoring_components and not options.scoring_components:
        options.scoring_components = create_default_scoring_components()

        # Filter scoring components based on user preferences
        if not options.use_normal_scoring:
            options.scoring_components = [
                comp for comp in options.scoring_components
                if comp.__class__.__name__ != 'NormalScoring'
            ]

    return options


def mesh_bounding_box_center(mesh: "MeshObject") -> Vector3:
    """Get the center point of a mesh's bounding box.

    Args:
        mesh: Mesh object

    Returns:
        Center point of the bounding box as a 3D vector
    """
    bbox = cmds.exactWorldBoundingBox(mesh.name)  # minx, miny, minz, maxx, maxy, maxz
    center = np.array(
        [
            (bbox[0] + bbox[3]) / 2.0,  # x center
            (bbox[1] + bbox[4]) / 2.0,  # y center
            (bbox[2] + bbox[5]) / 2.0,  # z center
        ],
        dtype=np.float64,
    )

    return center


def mesh_bounding_box_size(mesh: "MeshObject") -> Vector3:
    """Get the size of a mesh's bounding box.

    Args:
        mesh: Mesh object

    Returns:
        Size of the bounding box (width, height, depth) as a 3D vector
    """
    bbox = cmds.exactWorldBoundingBox(mesh.name)  # minx, miny, minz, maxx, maxy, maxz
    size = np.array(
        [
            abs(bbox[3] - bbox[0]),  # width (x)
            abs(bbox[4] - bbox[1]),  # height (y)
            abs(bbox[5] - bbox[2]),  # depth (z)
        ],
        dtype=np.float64,
    )

    return size


def calculate_average_nearest_distance(points: NDArray[np.float64]) -> float:
    """Calculate the average nearest neighbor distance for a point cloud.

    Args:
        points: Array of points (N, 3)

    Returns:
        Average nearest neighbor distance
    """
    from scipy.spatial import cKDTree

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(points)

    # Find the distance to the nearest neighbor for each point
    # We query with k=2 because the nearest point to any point is itself (dist=0)
    distances, _ = tree.query(points, k=2)

    # Take the second column (distances to the actual nearest neighbor)
    nearest_distances = distances[:, 1]

    # Return the average
    return float(np.mean(nearest_distances))
