# -*- coding: utf-8 -*-
"""
Weight transfer utilities for mesh registration.

This module provides functions for transferring skinning weights between meshes.
"""

import numpy as np
from maya import cmds, mel

from ..util import timeit
from .core import TriangleWeightIndex
from .geometry import triangle_interpolation


@timeit
def get_weight_distance(
    src_triangles,
    src_num_tri,
    src_weight,         # shape=(num_vertices, num_inf)
    tar_weight,         # shape=(tar_num_vertex, num_inf)
    tar_joints_retarget,
    top_n=100
):
    """
    Calculate weight-based distance between source triangles and target vertices.
    
    This function:
    1. Extracts "valid influences"
    2. Creates array s_w_valid of shape (src_num_tri, len(valid_l)) from average source triangle weights
    3. For each target vertex, calculates distances using vector operations and sorts results
    
    Args:
        src_triangles (np.ndarray): Source mesh triangle indices, shape=(3*src_num_tri,)
        src_num_tri (int): Number of source triangles
        src_weight (list[list[float]]): Source mesh vertex weights [vertex][inf]
        tar_weight (list[list[float]]): Target mesh vertex weights [vertex][inf]
        tar_joints_retarget (list[int]): Target to source joint index mapping (or -1)
        top_n (int): Number of top results to keep
    
    Returns:
        list[list[TriangleWeightIndex]]:
            tar_best_weight_triangle_map, shape=(tar_num_vertex,),
            each element is a list of TriangleWeightIndex sorted by distance
    """
    # 1) Extract valid influences "valid_l"
    #    Skip elements where tar_joints_retarget[l] = -1
    num_inf = len(src_weight[0])
    valid_l = [l for l in range(num_inf) if tar_joints_retarget[l] != -1]
    
    # 2) Create s_w_valid of shape (src_num_tri, len(valid_l))
    #    from average source triangle weights
    src_triangles_np = np.array(src_triangles, dtype=np.int32).reshape(-1, 3)
    src_weight_np = np.array(src_weight, dtype=np.float32)
    tar_weight_np = np.array(tar_weight, dtype=np.float32)
    triangle_weights = src_weight_np[src_triangles_np]  # shape=(src_num_tri,3,num_inf)
    src_triangle_weight = np.mean(triangle_weights, axis=1)  # shape=(src_num_tri, num_inf)
    
    # Extract valid columns => s_w_valid shape=(src_num_tri, len(valid_l))
    s_w_valid = np.zeros((src_num_tri, len(valid_l)), dtype=np.float32)
    for col, l in enumerate(valid_l):
        # tar_joints_retarget[l] = source joint ID
        src_jid = tar_joints_retarget[l]
        # No need to check if src_jid is -1 as we already filtered those out
        s_w_valid[:, col] = src_triangle_weight[:, src_jid]
    
    # 3) Calculate distances for each target vertex
    tar_num_vertex = len(tar_weight)
    tar_best_weight_triangle_map = []
    
    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                isInterruptable=False,
                maxValue=tar_num_vertex,
                status="Calculating weight-based triangle mapping..."
        )
    
    for i in range(tar_num_vertex):
        t_w_valid = tar_weight_np[i, valid_l]
        
        # Calculate all distances at once using vector operations
        # diff shape=(src_num_tri, len(valid_l))
        diff = s_w_valid - t_w_valid
        dist2 = np.sum(diff**2, axis=1)  # Sum of squares => shape=(src_num_tri,)
        dist = np.sqrt(dist2)            # shape=(src_num_tri,)
        
        # Sort distances and convert to TriangleWeightIndex
        idx_sorted = np.argsort(dist)
        if top_n > 0:
            idx_sorted = idx_sorted[:top_n]
        
        triangle_weight_indices = []
        for j in idx_sorted:
            twi = TriangleWeightIndex(
                weight_distance=dist[j],
                triangle_index=j
            )
            triangle_weight_indices.append(twi)
        
        tar_best_weight_triangle_map.append(triangle_weight_indices)
        
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1)
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)
    
    return tar_best_weight_triangle_map
