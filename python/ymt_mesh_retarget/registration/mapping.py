# -*- coding: utf-8 -*-
"""
Mapping utility functions for mesh registration.

This module provides functions for creating and managing mapping points between meshes.
"""

import numpy as np
from maya import cmds, mel
from scipy.spatial import cKDTree

from ..util import timeit, get_short_name
from .core import CorrespondencePoint, MappingNode, MappingResult


def find_nearest_vertex_index(position, triangle_idx, raycast_data):
    """Find the nearest vertex index to the given position
    
    This function determines the closest vertex to a point (e.g., ray intersection point)
    using triangle information. It either:
    1. Uses the triangle vertices directly
    2. Performs a local search around the triangle
    
    Args:
        position (np.ndarray): The position to find nearest vertex for
        triangle_idx (int): The triangle index
        raycast_data (dict): Raycast result data containing additional info
        
    Returns:
        int: The index of the nearest vertex
    """
    # If raycast_data contains triangle vertices info, use it
    if "triangle_vertex_indices" in raycast_data:
        indices = raycast_data["triangle_vertex_indices"]
        best_idx = indices[0]  # Default to first vertex
        
        # If the raycast has vertex positions, find closest
        if "triangle_vertices" in raycast_data:
            vertices = raycast_data["triangle_vertices"]
            min_dist = float('inf')
            
            for i, vertex in enumerate(vertices):
                dist = np.linalg.norm(vertex - position)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = indices[i]
                    
            return best_idx
    
    # Default implementation - for improved version, we would use
    # a spatial search structure like KD-tree to find nearest vertex
    # For now, we'll return a placeholder that should be resolved later
    # This would be improved when MeshObject is passed to the function
    return int(raycast_data.get("nearest_vertex_index", -1))


@timeit
def get_mapping_points(
        target_points,
        target_joint_group,
        target_bone_group,
        target_weights,
        target_joint_names,
        max_distance=0.0
):
    """Get mapping points for target mesh vertices
    
    Generate mapping points for each target mesh vertex based on skeleton information.
    
    Args:
        target_points (np.ndarray): Target mesh vertex positions
        target_joint_group (list): Target joint group
        target_bone_group (list): Target bone group
        target_weights (list): Target mesh skinning weights
        target_joint_names (list): Target joint names
        max_distance (float): Maximum distance (0 means no limit)
        
    Returns:
        list: Array of mapping results
    """
    # Joint weight mapping
    bones_weight_index = [-1] * len(target_joint_group)
    
    for i, joint in enumerate(target_joint_group):
        for j, joint_name in enumerate(target_joint_names):
            jname = get_short_name(joint_name)
            lname = get_short_name(joint.detail_name)
            print(f"joint_name: {joint_name}, jname: {jname}, detail_name: {joint.detail_name}, lname: {lname}")
            if jname == lname:
                bones_weight_index[i] = j
                break
    
    # Initialize results array
    mapping_results = []
    
    # Process each vertex
    for vert_idx, vertex in enumerate(target_points):
        mapping_result = MappingResult(vertex_index=vert_idx)
        
        # Process each bone
        for bone_idx, bone in enumerate(target_bone_group):
            start_joint = target_joint_group[bone.start_joint_index]
            end_joint = target_joint_group[bone.end_joint_index]
            
            # Check weight
            weight_index = bones_weight_index[bone.start_joint_index]
            if weight_index < 0 or target_weights[vert_idx][weight_index] < 1e-5:
                continue
            
            # Calculate bone vector
            start_point = start_joint.position
            end_point = end_joint.position
            bone_vector = end_point - start_point
            bone_length = np.linalg.norm(bone_vector)
            
            if bone_length < 1e-10:
                continue
            
            normalize_bone_vector = bone_vector / bone_length
            
            # Calculate projection of vertex onto bone
            w = vertex - start_point
            projection_length = np.dot(w, normalize_bone_vector)
            p = projection_length * normalize_bone_vector + start_point
            
            # Check if projection point is valid
            check_direction = p - start_point
            left_is_legal = np.dot(check_direction, normalize_bone_vector) > 0
            left_is_legal |= max_distance > np.linalg.norm(check_direction)
            
            check_direction = p - end_point
            right_is_legal = np.dot(check_direction, normalize_bone_vector) < 0
            right_is_legal |= max_distance > np.linalg.norm(check_direction)
            
            # Distance from vertex to projection point
            distance = np.linalg.norm(p - vertex)
            
            if right_is_legal and left_is_legal:
                # Valid projection point on bone
                mapping_node = MappingNode(
                    point=p,
                    bone_index=bone_idx,
                    distance=distance,
                    weight=target_weights[vert_idx][weight_index]
                )
                mapping_result.node_array.append(mapping_node)
            else:
                # Use bone endpoint
                start_distance = np.linalg.norm(start_point - vertex)
                end_distance = np.linalg.norm(end_point - vertex)
                
                if start_distance < end_distance:
                    mapping_node = MappingNode(
                        point=start_point,
                        bone_index=bone_idx,
                        distance=start_distance,
                        weight=target_weights[vert_idx][weight_index]
                    )
                    mapping_result.node_array.append(mapping_node)
                else:
                    mapping_node = MappingNode(
                        point=end_point,
                        bone_index=bone_idx,
                        distance=end_distance,
                        weight=target_weights[vert_idx][weight_index]
                    )
                    mapping_result.node_array.append(mapping_node)
        
        mapping_results.append(mapping_result)
    
    return mapping_results


def create_optimized_correspondence_points(
        raycast_result_array,
        tar_mapping_points,
        target_points,
        max_points_per_target=1,
        min_weight_threshold=0.01,
        distance_weight=1.0,
        ray_weight=0.5
):
    """Create optimized correspondence points from raycast results
    
    Keep only the most significant correspondence points for each target vertex
    to reduce the total number.
    
    Args:
        raycast_result_array (list): Raycast result array
        tar_mapping_points (list): Target mapping points
        target_points (np.ndarray): Target vertex positions
        max_points_per_target (int): Maximum number of correspondence points per target vertex
        min_weight_threshold (float): Minimum weight threshold
        distance_weight (float): Weight coefficient for distance score
        ray_weight (float): Weight coefficient for ray information score
        
    Returns:
        list: Optimized correspondence points list
    """
    # Dictionary to store correspondence points (key: target vertex index)
    correspondence_dict = {}
    
    # Process raycast results
    for i, raycast_results in enumerate(raycast_result_array):
        if not raycast_results:
            continue
            
        target_idx = tar_mapping_points[i].vertex_index
        target_pos = target_points[target_idx]
        
        # Candidates list for this target vertex (with scores)
        candidates = []
        
        for raycast in raycast_results:
            # Validate triangle index
            triangle_idx = raycast["triangle_index"]
            if triangle_idx < 0:
                continue
                
            # Intersection point
            src_pos = raycast["point"]
            
            # Basic distance and weight
            distance = np.linalg.norm(src_pos - target_pos)
            basic_weight = 1.0 / (1.0 + distance)
            
            # Use ray information to calculate quality score
            ray_quality = 1.0
            if "relate_distance" in raycast:
                # Smaller relate_distance is better
                ray_quality = 1.0 / (1.0 + raycast["relate_distance"])
            
            # Consider node weight
            node_weight = raycast.get("weight", 1.0)
            
            # Calculate total score
            # Consider distance-based score, ray quality, and node weight
            total_score = (
                distance_weight * basic_weight +  # Distance-based score
                ray_weight * ray_quality * node_weight  # Ray quality and node weight
            ) / (distance_weight + ray_weight)  # Normalize
            
            # Add to candidates if score exceeds threshold
            if total_score >= min_weight_threshold:
                # Find nearest source vertex to the intersection point
                source_vertex_index = find_nearest_vertex_index(src_pos, triangle_idx, raycast)
                
                candidates.append({
                    "source_index": source_vertex_index,
                    "target_index": target_idx,
                    "weight": basic_weight,  # Keep original weight calculation
                    "score": total_score,    # Total score for sorting
                    "triangle_index": triangle_idx
                })
        
        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Keep top N candidates
        top_candidates = candidates[:max_points_per_target]
        
        # Add final correspondence points to dictionary
        if target_idx not in correspondence_dict:
            correspondence_dict[target_idx] = []
            
        correspondence_dict[target_idx].extend(top_candidates)
    
    # Create final correspondence points list
    optimized_correspondence_points = []
    
    for target_idx, candidates in correspondence_dict.items():
        # Create correspondence point object for each candidate
        for candidate in candidates:
            source_idx = candidate["source_index"]
            if source_idx < 0:
                # Skip invalid source indices
                continue
                
            correspondence_point = CorrespondencePoint(
                source_index=source_idx,
                target_index=candidate["target_index"],
                weight=candidate["weight"]
            )
            optimized_correspondence_points.append(correspondence_point)
    
    return optimized_correspondence_points


def find_correspondence_using_skeleton(
        source_points,
        target_points,
        source_weights,
        target_weights,
        source_joints,
        target_joints,
        sample_rate=1.0,
        weight_decay=2.0
):
    """Find correspondence points using skeleton information
    
    This is a fallback method for finding correspondence points when
    advanced methods fail.
    
    Args:
        source_points (np.ndarray): Source mesh vertex positions
        target_points (np.ndarray): Target mesh vertex positions
        source_weights (list): Source mesh skinning weights
        target_weights (list): Target mesh skinning weights
        source_joints (list): Source joint names
        target_joints (list): Target joint names
        sample_rate (float): Vertex sampling rate (0.0-1.0)
        weight_decay (float): Weight decay coefficient
        
    Returns:
        list: Correspondence points
    """
    # Create joint name to index mapping
    source_joint_map = {j.split(":")[-1]: i for i, j in enumerate(source_joints)}
    _target_joint_map = {j.split(":")[-1]: i for i, j in enumerate(target_joints)}
    
    # Reduce vertex count for sampling
    if sample_rate < 1.0:
        num_samples = max(10, int(len(target_points) * sample_rate))
        sample_indices = np.linspace(0, len(target_points) - 1, num_samples).astype(int)
    else:
        sample_indices = range(len(target_points))
    
    # Initialize correspondence points list
    correspondence_points = []
    
    # Find correspondence for each target vertex
    for idx in sample_indices:
        target_pos = target_points[idx]
        
        # Find joints that influence this vertex
        influential_joints = []
        for joint_idx, weight in enumerate(target_weights[idx]):
            if weight > 0.01:  # Consider only joints with significant influence
                influential_joints.append((joint_idx, weight))
        
        # Sort by influence weight (descending)
        influential_joints.sort(key=lambda x: x[1], reverse=True)
        
        best_match = None
        min_distance = float("inf")
        
        # Find appropriate correspondence point
        for joint_idx, weight in influential_joints:
            target_joint_name = target_joints[joint_idx].split(":")[-1]
            
            # Check if source mesh has the same joint
            if target_joint_name in source_joint_map:
                source_joint_idx = source_joint_map[target_joint_name]
                
                # Find vertices in source mesh influenced by this joint
                candidates = []
                for src_idx, src_weights in enumerate(source_weights):
                    src_weight = src_weights[source_joint_idx]
                    if src_weight > 0.01:
                        candidates.append((src_idx, src_weight))
                
                # Find the closest candidate
                for src_idx, src_weight in candidates:
                    src_pos = source_points[src_idx]
                    distance = np.linalg.norm(src_pos - target_pos)
                    
                    # Adjust distance by weights
                    adjusted_distance = distance / (src_weight * weight)
                    
                    if adjusted_distance < min_distance:
                        min_distance = adjusted_distance
                        best_match = (src_idx, src_pos)
        
        # Add to correspondence points if a match was found
        if best_match:
            src_idx, _ = best_match
            correspondence_points.append(CorrespondencePoint(
                source_index=src_idx,
                target_index=idx,
                weight=1.0 / (1.0 + min_distance)  # Weight based on distance
            ))
    
    return correspondence_points
