"""Mapping utility functions for mesh registration.

This module provides functions for creating and managing mapping points between meshes.
"""

import typing
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..logger import logger
from ..util import get_short_name, timeit
from .core import CorrespondencePoint, MappingNode, MappingResult, RegistrationOptions, Vector3


if typing.TYPE_CHECKING:
    from ..objects.mesh import MeshObject
    from .core import BoneNode, JointNode, RaycastResult


def find_nearest_vertex_index(position: Vector3, triangle_idx: int, raycast_data: Any) -> int:  # noqa: ARG001
    """Find the nearest vertex index to the given position.

    This function determines the closest vertex to a point (e.g., ray intersection point)
    using triangle information.

    Args:
        position: The position to find nearest vertex for
        triangle_idx: The triangle index
        raycast_data: Additional information about the raycast

    Returns:
        The index of the nearest vertex
    """
    # This is a placeholder implementation that should be improved
    # In a real implementation, we would use mesh topology information

    # For now, return -1 to indicate not implemented
    return -1


@timeit
def get_mapping_points(
    target_points: NDArray[np.float64],
    target_joint_group: list["JointNode"],
    target_bone_group: list["BoneNode"],
    target_weights: list[list[float]],
    target_joint_names: list[str],
    max_distance: float = 0.0,
    # sample_count: int = 1500,
) -> list[MappingResult]:
    """Get mapping points for target mesh vertices.

    Generate mapping points for each target mesh vertex based on skeleton information.
    These mapping points are used to establish correspondences between meshes.

    Args:
        target_points: Target mesh vertex positions (N, 3)
        target_joint_group: Target joint group
        target_bone_group: Target bone group
        target_weights: Target mesh skinning weights
        target_joint_names: Target joint names
        max_distance: Maximum distance (0 means no limit)
        sample_count: Number of samples to use for mapping

    Returns:
        Array of mapping results for each target vertex
    """
    # Joint weight mapping
    bones_weight_index = [-1] * len(target_joint_group)

    # Map joints to weight indices
    for i, joint in enumerate(target_joint_group):
        for j, joint_name in enumerate(target_joint_names):
            jname = get_short_name(joint_name)
            lname = get_short_name(joint.detail_name)
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
                    weight=target_weights[vert_idx][weight_index],
                )
                mapping_result.add_node(mapping_node)

            else:
                # Use bone endpoint
                start_distance = np.linalg.norm(start_point - vertex)
                end_distance = np.linalg.norm(end_point - vertex)

                if start_distance < end_distance:
                    mapping_node = MappingNode(
                        point=start_point,
                        bone_index=bone_idx,
                        distance=start_distance,
                        weight=target_weights[vert_idx][weight_index],
                    )
                    mapping_result.add_node(mapping_node)

                else:
                    mapping_node = MappingNode(
                        point=end_point,
                        bone_index=bone_idx,
                        distance=end_distance,
                        weight=target_weights[vert_idx][weight_index],
                    )
                    mapping_result.add_node(mapping_node)

        mapping_results.append(mapping_result)

    return mapping_results


def create_optimized_correspondence_points(
    raycast_result_array: list[list["RaycastResult"]],
    src_mapping_points: list[MappingResult],
    source_points: NDArray[np.float64],
    max_points_per_target: int = 1,
    min_weight_threshold: float = 0.01,
    source_mesh: Optional["MeshObject"] = None,
    target_mesh: Optional["MeshObject"] = None,
    options: Optional["RegistrationOptions"] = None,
) -> list[CorrespondencePoint]:
    """Create optimized correspondence points from raycast results.

    Keep only the most significant correspondence points for each source vertex
    to reduce the total number and improve quality.

    Args:
        raycast_result_array: Array of raycast results per source vertex
        src_mapping_points: source mapping points
        source_points: source vertex positions
        max_points_per_target: Maximum number of correspondence points per source vertex
        min_weight_threshold: Minimum weight threshold
        source_mesh: Optional source mesh for advanced scoring
        target_mesh: Optional target mesh for advanced scoring
        options: Optional registration options for advanced configuration

    Returns:
        Optimized list of correspondence points
    """
    # Dictionary to store correspondence points (key: source vertex index)
    correspondence_dict: dict[int, list[dict[str, Any]]] = {}

    # Check if we're using advanced scoring components
    scoring_components = options.scoring_components

    # If we're using advanced scoring but don't have the meshes, issue a warning
    if not source_mesh or not target_mesh:
        logger.warning("Advanced scoring enabled but meshes not provided. Falling back to basic scoring.")
        raise NotImplementedError("Advanced scoring requires source and target meshes")

    logger.debug(f"Using {len(scoring_components)} scoring components for correspondence optimization")
    logger.debug(f"Starting correspondence optimization with {len(raycast_result_array)} source vertices")
    # Process raycast results
    for i, raycast_results in enumerate(raycast_result_array):
        if not raycast_results:
            continue

        src_vtx_id = src_mapping_points[i].vertex_index
        src_pos = source_points[src_vtx_id]

        # Candidates list for this source vertex (with scores)
        candidates = []

        for raycast in raycast_results:
            # Validate triangle index
            triangle_idx = raycast.triangle_index
            if triangle_idx < 0:
                continue

            # Intersection point
            tar_pos = raycast.point

            # Basic distance and weight
            distance = np.linalg.norm(tar_pos - src_pos)

            # Create context for scoring components
            context = {
                "src_pos": src_pos,
                "tar_pos": tar_pos,
                "distance": distance,
                "raycast_result": raycast,
            }

            # Add mesh-specific data if available
            if source_mesh and target_mesh:
                # Add source normal if available
                try:
                    context["normal_src"] = source_mesh.get_vertex_normal(src_vtx_id)
                except Exception as e:
                    logger.debug(f"Failed to get source normal: {e}")
                    raise

                # For target normals, use vertex normals from hit triangle
                # We'll use the first vertex index as a simplified approach
                if raycast.vertex_indices:
                    try:
                        tar_vtx_id = raycast.vertex_indices[0]
                        context["normal_tar"] = target_mesh.get_vertex_normal(tar_vtx_id)
                    except Exception as e:
                        logger.debug(f"Failed to get target normal: {e}")
                        raise

                # Add laplacians if available and enabled
                if options and options.use_laplacian_scoring:
                    try:
                        context["laplacian_src"] = source_mesh.compute_laplacian_for_vertex(src_vtx_id)
                        if raycast.vertex_indices:
                            tar_vtx_id = raycast.vertex_indices[0]
                            context["laplacian_tar"] = target_mesh.compute_laplacian_for_vertex(tar_vtx_id)
                    except Exception as e:
                        logger.debug(f"Failed to get laplacians: {e}")
                        raise

                # Add weight vectors if available and enabled
                if options and options.use_weight_scoring:
                    try:
                        context["weights_src"] = source_mesh.get_vertex_weight_vector(src_vtx_id)
                        if raycast.vertex_indices:
                            tar_vtx_id = raycast.vertex_indices[0]
                            context["weights_tar"] = target_mesh.get_vertex_weight_vector(tar_vtx_id)
                    except Exception as e:
                        logger.debug(f"Failed to get weight vectors: {e}")
                        raise

            # Calculate total score using all scoring components
            total_score = 0.0
            total_weight = 0.0

            for component in scoring_components:
                component_weight = component.get_weight()
                component_score = component.compute_score(context)
                total_score += component_weight * component_score
                total_weight += component_weight

            # Normalize the score
            if total_weight > 0.0:
                total_score /= total_weight

            # Add to candidates if score exceeds threshold
            if total_score >= min_weight_threshold:
                # Find nearest source vertex to the intersection point
                for tar_vtx_id in raycast.vertex_indices:
                    candidates.append(
                        {
                            "source_index": src_vtx_id,
                            "target_index": tar_vtx_id,
                            "score": total_score,  # Total score for sorting
                            "triangle_index": triangle_idx,
                        },
                    )

        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        logger.debug(f"Found {len(candidates)} candidates for source vertex {src_vtx_id}")

        # Keep top N candidates
        top_candidates = candidates[:max_points_per_target]

        # Add final correspondence points to dictionary
        if src_vtx_id not in correspondence_dict:
            correspondence_dict[src_vtx_id] = []

        correspondence_dict[src_vtx_id].extend(top_candidates)

    # Create final correspondence points list
    optimized_correspondence_points = []

    for _source_idx, candidates in correspondence_dict.items():
        # Create correspondence point object for each candidate
        for candidate in candidates:
            target_idx = candidate["target_index"]
            if target_idx < 0:
                # Skip invalid source indices
                continue

            correspondence_point = CorrespondencePoint(
                source_index=candidate["source_index"],
                target_index=target_idx,
                score=candidate["score"],
            )
            optimized_correspondence_points.append(correspondence_point)

    return optimized_correspondence_points
