"""Alignment utilities for mesh registration.

This module provides functions for aligning meshes and joints in space.
"""

import typing
from typing import Callable, Optional

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from numpy.typing import NDArray

from ..logic import RBF, calculate_rbf_weight_matrix, get_distance_matrix
from ..util import timeit
from .core import BoneNode, JointNode, Kernel
from .utils import find_root_joints, get_matched_info, match_joint_positions, scale_joint_hierarchy_to_mesh


if typing.TYPE_CHECKING:
    from ..objects import MeshObject


@timeit
def calculate_alignment_transform_rbf(
    src_points: NDArray[np.float64],
    tar_points: NDArray[np.float64],
    kernel: "Kernel" = RBF.linear,
    radius: float = 1.0,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Calculate alignment transform using RBF.

    Use RBF to calculate the alignment transform of source space to target space.

    Args:
        src_points: Source points (M, 3)
        tar_points: Target points (M, 3)
        kernel: Radial basis function kernel
        radius: RBF radius

    Returns:
        RBF transform function that maps points from source to target space
    """
    # Calculate RBF weights to map source points to target points
    weights = calculate_rbf_weight_matrix(
        source_points=src_points,
        target_points=tar_points,
        kernel=kernel,
        radius=radius,
    )

    def rbf_transform(query_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform query points from source space to target space using RBF.

        Args:
            query_points: Points to transform, shape=(K,3)

        Returns:
            Transformed points, shape=(K,3)
        """
        # Calculate distance matrix (K,M)
        dist_mat = get_distance_matrix(query_points, tar_points, kernel, radius)

        # Build matrix according to RBF formula: [dist_mat, 1, query_points]
        K = query_points.shape[0]  # noqa: N806
        ones = np.ones((K, 1), dtype=np.float64)
        # dist_mat: (K,M), ones: (K,1), query_points: (K,3) => (K, M+1+3)
        h_combined = np.hstack([dist_mat, ones, query_points])

        # weights is (M+1+3, 3) so result is (K,3)
        deformed = h_combined @ weights
        return deformed

    return rbf_transform


@timeit
def calculate_alignment_transform(
    src_joint_group: list[JointNode],
    tar_joint_group: list[JointNode],
    kernel: "Kernel" = RBF.linear,
    radius: float = 1.0,
) -> Optional[NDArray[np.float64]]:
    """Calculate alignment transform between joint hierarchies.

    This function aligns the source joint hierarchy to the target joint hierarchy.

    Args:
        src_joint_group: Source joint group
        tar_joint_group: Target joint group
        kernel: RBF kernel
        radius: RBF radius

    Returns:
        Transformed source points or None if matching failed
    """
    # (1) Get all joint coordinates
    src_points = np.array([joint.position for joint in src_joint_group])  # (Ns,3)
    tar_points = np.array([joint.position for joint in tar_joint_group])  # (Nt,3)

    # (2) Find matching joints by name
    matched_src_indices, matched_tar_indices, _ = get_matched_info(src_joint_group, tar_joint_group)

    # (3) Need at least 3 matching joints for RBF
    if len(matched_src_indices) < 3:
        print(f"Not enough matched joints to build RBF. Found {len(matched_src_indices)}, need at least 3.")
        return None

    # (4) Create RBF transform function (source to target)
    matched_src_points = src_points[matched_src_indices]  # shape=(M,3)
    matched_tar_points = tar_points[matched_tar_indices]  # shape=(M,3)
    rbf_func = calculate_alignment_transform_rbf(
        matched_src_points,
        matched_tar_points,
        kernel=kernel,
        radius=radius,
    )

    # (5) Transform all source joints
    new_src_points = rbf_func(src_points)  # shape=(Ns,3)

    # (6) Force matched joints to exactly match target positions
    for i, src_id in enumerate(matched_src_indices):
        new_src_points[src_id] = tar_points[matched_tar_indices[i]]

    return new_src_points


def get_joint_tree(joint_names: list[str]) -> tuple[list[om.MDagPath], list[JointNode], list[BoneNode]]:
    """Get joint tree information from joint names.

    Args:
        joint_names: List of joint names

    Returns:
        Tuple containing:
        - List of joint DAG paths
        - List of JointNode objects
        - List of BoneNode objects
    """
    # Get joint DAG paths
    joint_paths = []
    for joint_name in joint_names:
        # Check if joint exists
        if not cmds.objExists(joint_name):
            continue

        # Get DAG path
        selection = om.MSelectionList()
        selection.add(joint_name)
        dag_path = selection.getDagPath(0)
        joint_paths.append(dag_path)

    if not joint_paths:
        return [], [], []

    # --------------------------------------------------------------------
    # 2. Find root joints among those DAG paths
    # --------------------------------------------------------------------
    root_joints = find_root_joints(joint_paths)

    # --------------------------------------------------------------------
    # 3. Perform a BFS from each root to build the JointNode/BoneNode lists
    # --------------------------------------------------------------------
    joint_group: list[JointNode] = []
    bone_group: list[BoneNode] = []
    visited = set()

    # We will re-populate joint_paths in BFS order
    bfs_ordered_paths: list[om.MDagPath] = []

    # A queue of (dag_path, parent_index)
    # parent_index = -1 indicates a root (no parent)
    queue = []
    for root in root_joints:
        queue.append((root, -1))

    while queue:
        current_path, parent_index = queue.pop(0)
        current_full_name = current_path.fullPathName()
        if current_full_name in visited:
            continue

        visited.add(current_full_name)

        # ----------------------------------------------------------------
        # Create a JointNode for the current DAG path
        # ----------------------------------------------------------------
        name = current_full_name
        pos = cmds.xform(name, query=True, translation=True, worldSpace=True)
        position = np.array(pos, dtype=np.float64)
        matrix = cmds.xform(name, query=True, matrix=True, worldSpace=True)

        current_index = len(joint_group)
        joint_node = JointNode(
            path=current_path,
            index=current_index,
            detail_name=name,
            position=position,
            matrix=matrix,
        )

        joint_group.append(joint_node)
        bfs_ordered_paths.append(current_path)

        # ----------------------------------------------------------------
        # Create a BoneNode if this is not a root
        # ----------------------------------------------------------------
        if parent_index != -1:
            bone_node = BoneNode(
                start_joint_index=parent_index,
                end_joint_index=current_index,
            )
            bone_group.append(bone_node)

        # ----------------------------------------------------------------
        # Enqueue child joints
        # ----------------------------------------------------------------
        children = cmds.listRelatives(name, children=True, type="joint", fullPath=True) or []
        for child_name in children:
            child_sel = om.MSelectionList()
            child_sel.add(child_name)
            child_path = child_sel.getDagPath(0)

            queue.append((child_path, current_index))

    return joint_paths, joint_group, bone_group


def match_joint_trees(
    src_mesh: "MeshObject",
    tar_mesh: "MeshObject",
    src_joint_group: list[JointNode],
    tar_joint_group: list[JointNode],
) -> list[int]:
    """Match joint trees between target and source.

    1. Calculate the bounding box of the source and target meshes
    2. Scale the source joint tree to match the target mesh proportions
    3. Match joint translations between source and target

    Args:
        src_mesh: Source mesh object
        tar_mesh: Target mesh object
        src_joint_group: Source joint group
        tar_joint_group: Target joint group

    Returns:
        List of matched source joint indices
    """
    # Scale source joint hierarchy to match target mesh proportions
    scale_joint_hierarchy_to_mesh(src_joint_group, src_mesh, tar_mesh)

    # Match joint translations between source and target
    src_indices = match_joint_positions(src_joint_group, tar_joint_group)

    return src_indices
