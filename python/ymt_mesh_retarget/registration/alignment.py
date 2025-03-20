# -*- coding: utf-8 -*-
"""
Alignment utilities for mesh registration.

This module provides functions for aligning meshes and joints in space.
"""

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from ..util import timeit
from ..logic import RBF, calculate_rbf_weight_matrix, get_distance_matrix


def match_joint_trees(tar_joint_group, src_joint_group):
    """Match joint trees between target and source
    
    Args:
        tar_joint_group (list): Target joint group
        src_joint_group (list): Source joint group
        
    Returns:
        list: Source joint indices
    """
    # Create joint mapping from target to source
    src_joint_index = [-1] * len(tar_joint_group)
    
    for i, tar_joint in enumerate(tar_joint_group):
        tar_short_name = tar_joint.detail_name.split(":")[-1].split("|")[-1]
        
        for j, src_joint in enumerate(src_joint_group):
            src_short_name = src_joint.detail_name.split(":")[-1].split("|")[-1]
            
            if src_short_name == tar_short_name:
                src_joint_index[i] = j
                break
    
    return src_joint_index


@timeit
def calculate_alignment_transform_rbf(
    src_points,
    tar_points,
    kernel=RBF.linear,
    radius=1.0
):
    """Calculate alignment transform using RBF.
    
    Use RBF to calculate the alignment transform of source space to target space.
    
    Args:
        src_points (np.ndarray): Source points (M, 3)
        tar_points (np.ndarray): Target points (M, 3)
        kernel (RBF): Radial basis function kernel
        radius (float): RBF radius
        
    Returns:
        function: RBF transform function
    """
    # Calculate RBF weights to map source points to target points
    weights = calculate_rbf_weight_matrix(
        source_points=src_points,
        target_points=tar_points,
        kernel=kernel,
        radius=radius
    )
    
    def rbf_transform(query_points):
        """
        Transform query points from source space to target space using RBF.
        
        Args:
            query_points: Points to transform, shape=(K,3)
        Returns:
            np.ndarray: Transformed points, shape=(K,3)
        """
        # Calculate distance matrix (K,M)
        dist_mat = get_distance_matrix(query_points, tar_points, kernel, radius)
        
        # Build matrix according to RBF formula: [dist_mat, 1, query_points]
        K = query_points.shape[0]
        ones = np.ones((K, 1), dtype=np.float64)
        # dist_mat: (K,M), ones: (K,1), query_points: (K,3) => (K, M+1+3)
        h_combined = np.hstack([dist_mat, ones, query_points])
        
        # weights is (M+1+3, 3) so result is (K,3)
        deformed = h_combined @ weights
        return deformed
    
    return rbf_transform


@timeit
def calculate_alignment_transform(
    src_joint_group,
    tar_joint_group,
    kernel=RBF.linear,
    radius=1.0
):
    """Calculate alignment transform between joint hierarchies.
    
    This function aligns the source joint hierarchy to the target joint hierarchy.
    
    Args:
        src_joint_group (list): Source joint group
        tar_joint_group (list): Target joint group
        kernel (RBF): RBF kernel
        radius (float): RBF radius
        
    Returns:
        np.ndarray or None: Transformed source points or None if matching failed
    """
    # (1) Get all joint coordinates
    src_points = np.array([joint.position for joint in src_joint_group])  # (Ns,3)
    tar_points = np.array([joint.position for joint in tar_joint_group])  # (Nt,3)
    
    # (2) Find matching joints by name
    src_matched_flags = np.zeros(len(src_joint_group), dtype=bool)
    tar_matched_flags = np.zeros(len(tar_joint_group), dtype=bool)
    
    for i, tar_joint in enumerate(tar_joint_group):
        tar_name = tar_joint.detail_name  # e.g. "joint1"
        for j, src_joint in enumerate(src_joint_group):
            src_name = src_joint.detail_name
            if src_name == tar_name:
                tar_matched_flags[i] = True
                src_matched_flags[j] = True
                break
    
    matched_src_indices = np.where(src_matched_flags)[0]
    matched_tar_indices = np.where(tar_matched_flags)[0]
    
    # (3) Need at least 3 matching joints for RBF
    if len(matched_src_indices) < 3:
        print("Not enough matched joints to build RBF. Skipping alignment.")
        return None
    
    # (4) Create RBF transform function (source to target)
    matched_src_points = src_points[matched_src_indices]  # shape=(M,3)
    matched_tar_points = tar_points[matched_tar_indices]  # shape=(M,3)
    rbf_func = calculate_alignment_transform_rbf(
        matched_src_points,
        matched_tar_points,
        kernel=kernel,
        radius=radius
    )
    
    # (5) Transform all source joints
    new_src_points = rbf_func(src_points)  # shape=(Ns,3)
    
    # (6) Force matched joints to exactly match target positions
    for i, src_id in enumerate(matched_src_indices):
        new_src_points[src_id] = tar_points[matched_tar_indices[i]]
    
    return new_src_points


def get_joint_tree(joint_names):
    """Get joint tree information from joint names
    
    Args:
        joint_names (list): List of joint names
        
    Returns:
        tuple: (joint_paths, joint_group, bone_group)
    """
    from .core import JointNode, BoneNode
    
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
    
    # Build joint tree
    joint_group = []
    bone_group = []
    
    # Find root joints (joints without parents)
    root_joints = []
    for i, path in enumerate(joint_paths):
        if path.length() == 1 or cmds.listRelatives(path.fullPathName(), parent=True, type="joint") is None:
            root_joints.append(path)
    
    if not root_joints:
        # Use joints with shallowest hierarchy as roots
        min_depth = min(path.fullPathName().count("|") for path in joint_paths)
        root_joints = [path for path in joint_paths if path.fullPathName().count("|") == min_depth]
    
    # Build tree from each root joint
    for root_path in root_joints:
        queue = [root_path]
        visited = set()
        
        while queue:
            current_path = queue.pop(0)
            if current_path.fullPathName() in visited:
                continue
            
            visited.add(current_path.fullPathName())
            
            # Create joint node
            name = current_path.fullPathName().split("|")[-1].split(":")[-1].split("|")[-1]
            pos = cmds.xform(current_path.fullPathName(), query=True, translation=True, worldSpace=True)
            position = np.array(pos, dtype=np.float64)
            
            joint_node = JointNode(
                path=current_path,
                index=len(joint_group),
                detail_name=name,
                position=position
            )
            joint_group.append(joint_node)
            
            # Get child joints
            children = cmds.listRelatives(current_path.fullPathName(), children=True, type="joint", fullPath=True) or []
            
            for child in children:
                # Skip helper joints
                if "helper" in child:
                    continue
                
                child_sel = om.MSelectionList()
                child_sel.add(child)
                child_path = child_sel.getDagPath(0)
                
                # Create bone node
                bone_node = BoneNode(
                    start_joint_index=joint_node.index,
                    end_joint_index=len(joint_group)  # Index of child joint to be added
                )
                bone_group.append(bone_node)
                
                queue.append(child_path)
    
    return joint_paths, joint_group, bone_group