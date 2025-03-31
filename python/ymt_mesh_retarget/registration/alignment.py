"""Alignment utilities for mesh registration.

This module provides functions for aligning meshes and joints in space.
"""
import typing
from typing import Callable, Optional

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from ..logger import logger
from ..logic import (
    RBF,
    calculate_rbf_weight_matrix,
    get_distance_matrix,
)
from ..util import timeit
from .core import BoneNode, CorrespondencePoint, JointNode, Kernel
from .utils import (
    find_root_joints,
    get_matched_info,
    match_joint_positions,
    scale_joint_hierarchy_to_mesh,
)


if typing.TYPE_CHECKING:
    from ..objects import MeshObject
    from ..types import VertexArray


@timeit
def calculate_alignment_transform_rbf(
    src_points: "VertexArray",
    tar_points: "VertexArray",
    kernel: "Kernel" = RBF.linear,
    radius: float = 1.0,
) -> Callable[["VertexArray"], "VertexArray"]:
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

    def rbf_transform(query_points: "VertexArray") -> "VertexArray":
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
) -> Optional["VertexArray"]:
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
        logger.warning(f"Not enough matched joints to build RBF. Found {len(matched_src_indices)}, need at least 3.")
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
    joint_paths_input = []
    for joint_name in joint_names:
        if cmds.objExists(joint_name):
            selection = om.MSelectionList()
            selection.add(joint_name)
            dag_path = selection.getDagPath(0)
            joint_paths_input.append(dag_path)

    if not joint_paths_input:
        return [], [], []

    # 2) Find root joints among the DAG paths
    root_joints = find_root_joints(joint_paths_input)

    # 3) BFS gather: build BFS-ordered data
    bfs_ordered_joint_group = []
    bfs_ordered_bone_group = []
    bfs_ordered_paths = []

    visited = set()
    queue = [(root, -1) for root in root_joints]

    while queue:
        current_path, parent_index = queue.pop(0)
        current_full_name = current_path.fullPathName()
        if current_full_name in visited:
            continue

        visited.add(current_full_name)

        # Create a JointNode
        pos = cmds.xform(current_full_name, query=True, t=True, ws=True)
        matrix = cmds.xform(current_full_name, query=True, m=True, ws=True)

        current_index = len(bfs_ordered_joint_group)
        jnode = JointNode(
            path=current_path,
            index=current_index,
            detail_name=current_full_name,
            position=np.array(pos, dtype=np.float64),
            matrix=matrix,
        )
        bfs_ordered_joint_group.append(jnode)
        bfs_ordered_paths.append(current_path)

        # If not a root, create a bone from the parent to the child
        if parent_index != -1:
            bone = BoneNode(start_joint_index=parent_index, end_joint_index=current_index)
            bfs_ordered_bone_group.append(bone)

        # Enqueue child joints
        children = cmds.listRelatives(current_full_name, c=True, type="joint", f=True) or []
        for child_name in children:
            child_sel = om.MSelectionList()
            child_sel.add(child_name)
            child_path = child_sel.getDagPath(0)
            queue.append((child_path, current_index))

    # ---------------------------------------------------------------------
    # 4) Post-process: reorder the BFS results to match the `joint_names` order
    # ---------------------------------------------------------------------
    # Build map from full name -> BFS index
    name_to_bfs_index = {
        jnode.detail_name: i
        for i, jnode in enumerate(bfs_ordered_joint_group)
    }

    # Create the final joint_group (and a map from old -> new index)
    final_joint_group = []
    old_to_new_index = {}
    for name in joint_names:
        if name in name_to_bfs_index:
            old_index = name_to_bfs_index[name]
            new_index = len(final_joint_group)
            old_to_new_index[old_index] = new_index
            final_joint_group.append(bfs_ordered_joint_group[old_index])

    # Remap bone indices
    final_bone_group = []
    for bone in bfs_ordered_bone_group:
        if (bone.start_joint_index in old_to_new_index and
            bone.end_joint_index   in old_to_new_index):
            new_start = old_to_new_index[bone.start_joint_index]
            new_end   = old_to_new_index[bone.end_joint_index]
            bone = BoneNode(start_joint_index=new_start, end_joint_index=new_end)
            final_bone_group.append(bone)

    # Reorder the joint_paths similarly
    final_joint_paths = [None] * len(final_joint_group)
    for new_idx, jnode in enumerate(final_joint_group):
        final_joint_paths[new_idx] = jnode.path

    # Return results in the original `joint_names` order
    return final_joint_paths, final_joint_group, final_bone_group


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


def shrink_mesh_toward_skeleton(
    vertices: "VertexArray",
    joint_group: list[JointNode],
    bone_group: list[BoneNode],
    weights: list[list[float]],
    shrink_factor: float = 0.5,
) -> "VertexArray":
    """Move each vertex toward its associated bone(s) to simulate skeleton shrinkage.

    Args:
        vertices: (N,3) source vertex positions
        joint_group: Joint hierarchy
        bone_group: List of bones
        weights: skinning weights per vertex
        shrink_factor: shrink amount in [0.0, 1.0]; 1.0 = full projection onto bone

    Returns:
        Deformed vertex array (N,3)
    """
    num_vertices = vertices.shape[0]
    new_vertices = np.copy(vertices)

    for vidx in range(num_vertices):
        vertex = vertices[vidx]

        # ウェイト付きの射影位置をまとめる
        weighted_projection = np.zeros(3, dtype=np.float32)
        total_weight = 0.0

        for bone in bone_group:
            start_idx = bone.start_joint_index
            end_idx   = bone.end_joint_index

            # ウェイトの取り出し (無効なインデックスや極小ウェイトはスキップ)
            w_start = 0.0
            w_end   = 0.0

            # start_joint_index のウェイトを取得
            if 0 <= start_idx < len(weights[vidx]):
                w_start = weights[vidx][start_idx]
            # end_joint_index のウェイトを取得
            if 0 <= end_idx < len(weights[vidx]):
                w_end = weights[vidx][end_idx]

            w = w_start + w_end
            if w <= 1e-6:
                # スキップ(このボーンからの影響は無視)
                continue

            # ボーンの両端ジョイントの位置を取得
            start_joint = joint_group[start_idx]
            end_joint   = joint_group[end_idx]
            a = start_joint.position  # Bone start
            b = end_joint.position    # Bone end
            ab = b - a
            ab_norm = np.linalg.norm(ab)

            # ボーンがゼロ長さに近い場合はスキップ
            if ab_norm < 1e-6:
                continue

            # ボーン方向単位ベクトル
            ab_dir = ab / ab_norm

            # 頂点をボーンの線分上に射影 (最近接点を取得)
            ap = vertex - a
            t = np.dot(ap, ab_dir)
            # 0 ~ ab_norm の範囲にクランプ (線分外に出ないようにする)
            t = np.clip(t, 0.0, ab_norm)
            projected = a + ab_dir * t

            # 重み w を使って加算
            weighted_projection += w * projected
            total_weight += w

        # 1つも有効なボーンがなければスキップ
        if total_weight < 1e-6:
            continue

        # ボーンへの射影点を頂点に反映 (shrink_factor分だけ寄せる)
        final_proj = weighted_projection / total_weight

        # check if final_proj is NaN or infinite
        if np.any(np.isinf(final_proj)) or np.any(np.isnan(final_proj)):
            logger.error("Final projection contains infinite or NaN values")
            logger.error(f"{np.where(np.isnan(final_proj))} count: {np.sum(np.isnan(final_proj))}")
            logger.error(f"{np.where(np.isinf(final_proj))} count: {np.sum(np.isinf(final_proj))}")
            continue

        new_vertices[vidx] = (1.0 - shrink_factor) * vertex + shrink_factor * final_proj

    return new_vertices


@timeit
def project_shrunk_vertices_nearest(
    shrunk_vertices: "VertexArray",
    target_vertices: "VertexArray",
    k: int = 1,
) -> list[CorrespondencePoint]:
    """Project shrunk vertices to their nearest neighbor(s) on the target mesh.

    Args:
        shrunk_vertices: (N,3) array of deformed vertex positions
        target_vertices: (M,3) array of target vertex positions
        k: Number of nearest neighbors to consider per vertex (default: 1)

    Returns:
        List of CorrespondencePoint(source_index=i, target_index=nearest_j, score=1.0)
    """
    from scipy.spatial import cKDTree

    try:
        tree = cKDTree(target_vertices)
    except ValueError:
        # check if target_vertices has infinite or NaN values
        if np.any(np.isinf(target_vertices)) or np.any(np.isnan(target_vertices)):
            logger.error("Target vertices contain infinite or NaN values")
            logger.error(f"{np.where(np.isnan(target_vertices))} count: {np.sum(np.isnan(target_vertices))}")
            logger.error(f"{np.where(np.isinf(target_vertices))} count: {np.sum(np.isinf(target_vertices))}")
        raise

    # Perform nearest neighbor search
    if k == 1:
        distances, indices = tree.query(shrunk_vertices)  # shape (N,)
        return [
            CorrespondencePoint(source_index=i, target_index=int(indices[i]), score=1.0)
            for i in range(len(shrunk_vertices))
        ]
    else:
        distances, indices = tree.query(shrunk_vertices, k=k)  # shape (N,k)
        cps: list[CorrespondencePoint] = []
        for i in range(len(shrunk_vertices)):
            for j in range(k):
                cp = CorrespondencePoint(source_index=i, target_index=int(indices[i][j]), score=1.0)
                cps.append(cp)
        return cps
