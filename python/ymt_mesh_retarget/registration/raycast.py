# -*- coding: utf-8 -*-
"""
Raycast utilities for mesh registration.

This module provides functions for raycasting operations using Embree or fallback methods.
"""

import numpy as np
from maya import cmds, mel

try:
    import embreex
    from embreex import rtcore_scene as rtcs
    from embreex.mesh_construction import TriangleMesh
    EMBREE_AVAILABLE = True
except ImportError:
    EMBREE_AVAILABLE = False
    cmds.warning("embreex library not found. Using standard raycasting instead.")


def build_embree_scene_from_source(src_triangles):
    """
    Build an Embree scene from source mesh triangle array.
    
    Args:
        src_triangles (np.ndarray): shape = (num_tri, 3, 3)
            Number of triangles = num_tri
            Each triangle has 3 vertices, each vertex has xyz (3D) coordinates

    Returns:
        tuple: (EmbreeScene, TriangleMesh) - scene and mesh
    """
    if not EMBREE_AVAILABLE:
        raise ImportError("Embree library is not available. Cannot build Embree scene.")
    
    scene = rtcs.EmbreeScene()
    mesh = TriangleMesh(scene, src_triangles)  # This builds the BVH
    return scene, mesh


def perform_raycast(
        tar_mapping_points,
        src_triangles,
        src_triangle_indices,
        sample_number,
        sample_degree,
        src_joint_group=None,
        tar_joint_group=None,
        src_bone_group=None,
        tar_bone_group=None,
        batch_size=1024,
        max_triangles=0,
        match_joint_trees_func=None
):
    """
    Perform raycasting to find correspondence points between meshes.
    
    This is a high-level function that chooses between Embree-based raycasting
    or standard raycasting based on availability.
    
    Args:
        tar_mapping_points (list): Target mapping points
        src_triangles (np.ndarray): Source mesh vertices
        src_triangle_indices (np.ndarray): Source mesh triangle indices
        sample_number (int): Number of sample rays
        sample_degree (float): Angle range for sampling (degrees)
        src_joint_group (list, optional): Source joint group
        tar_joint_group (list, optional): Target joint group
        src_bone_group (list, optional): Source bone group
        tar_bone_group (list, optional): Target bone group
        batch_size (int, optional): Batch size for ray processing
        max_triangles (int, optional): Maximum number of triangles to process (0=unlimited)
        match_joint_trees_func (callable, optional): Function to match joint trees
        
    Returns:
        list: Raycast result array
    """
    if EMBREE_AVAILABLE:
        return perform_embree_raycast(
            tar_mapping_points=tar_mapping_points,
            src_triangles=src_triangles,
            src_triangle_indices=src_triangle_indices,
            sample_number=sample_number,
            sample_degree=sample_degree,
            src_joint_group=src_joint_group,
            tar_joint_group=tar_joint_group,
            src_bone_group=src_bone_group,
            tar_bone_group=tar_bone_group,
            batch_size=batch_size,
            max_triangles=max_triangles,
            match_joint_trees_func=match_joint_trees_func
        )
    else:
        return perform_standard_raycast(
            tar_mapping_points=tar_mapping_points,
            src_triangles=src_triangles,
            src_triangle_indices=src_triangle_indices,
            sample_number=sample_number,
            sample_degree=sample_degree,
            src_joint_group=src_joint_group,
            tar_joint_group=tar_joint_group,
            src_bone_group=src_bone_group,
            tar_bone_group=tar_bone_group,
            match_joint_trees_func=match_joint_trees_func
        )


def perform_embree_raycast(
        tar_mapping_points,
        src_triangles,
        src_triangle_indices,
        sample_number,
        sample_degree,
        src_joint_group=None,
        tar_joint_group=None,
        src_bone_group=None,
        tar_bone_group=None,
        batch_size=1024,
        max_triangles=0,
        match_joint_trees_func=None
):
    """
    Perform high-speed raycast using Intel Embree for correspondence point search.
    
    This function uses the Embree library for batch processing of multiple rays
    to efficiently find correspondence points between meshes.
    
    Args:
        tar_mapping_points (list): Target mapping points
        src_triangles (np.ndarray): Source mesh vertices
        src_triangle_indices (np.ndarray): Source mesh triangle indices
        sample_number (int): Number of sample rays
        sample_degree (float): Angle range for sampling (degrees)
        src_joint_group (list, optional): Source joint group
        tar_joint_group (list, optional): Target joint group
        src_bone_group (list, optional): Source bone group
        tar_bone_group (list, optional): Target bone group
        batch_size (int, optional): Batch size for ray processing
        max_triangles (int, optional): Maximum number of triangles to process (0=unlimited)
        match_joint_trees_func (callable, optional): Function to match joint trees
        
    Returns:
        list: Raycast result array
    """
    from .geometry import rand_cone_vector
    
    if not EMBREE_AVAILABLE:
        raise ImportError("Embree library is not available. Cannot perform Embree raycast.")
    
    # Convert input data to NumPy arrays
    src_triangles_np = np.asarray(src_triangles, dtype=np.float32)
    src_triangle_indices_np = np.asarray(src_triangle_indices, dtype=np.int32)
    
    # Create joint mapping from target to source
    if match_joint_trees_func is None:
        from .alignment import match_joint_trees
        match_joint_trees_func = match_joint_trees
    
    src_joint_index = match_joint_trees_func(tar_joint_group, src_joint_group)
    
    # Get number of source triangles
    src_num_triangles = len(src_triangle_indices) // 3
    
    # Limit number of triangles (optional)
    if max_triangles > 0 and max_triangles < src_num_triangles:
        src_num_triangles = max_triangles
    
    # Set up progress bar
    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
            bar,
            edit=True,
            beginProgress=True,
            status="Setting up Embree acceleration structure...",
            maxValue=100
        )
        cmds.progressBar(bar, edit=True, step=10)
    
    # Convert triangles to Embree format
    # Triangle array shape=(src_num_triangles, 3, 3)
    embree_triangles = np.zeros((src_num_triangles, 3, 3), dtype=np.float32)
    
    for i in range(src_num_triangles):
        v0_idx = src_triangle_indices_np[i * 3 + 0]
        v1_idx = src_triangle_indices_np[i * 3 + 1]
        v2_idx = src_triangle_indices_np[i * 3 + 2]
        
        embree_triangles[i, 0] = src_triangles_np[v0_idx]
        embree_triangles[i, 1] = src_triangles_np[v1_idx]
        embree_triangles[i, 2] = src_triangles_np[v2_idx]
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, step=40)
    
    # Create Embree scene and mesh
    scene = rtcs.EmbreeScene()
    _mesh = TriangleMesh(scene, embree_triangles)
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, step=50)
        cmds.progressBar(bar, edit=True, endProgress=True)
        
        # Set up new progress bar for correspondence point calculation
        cmds.progressBar(
            bar,
            edit=True,
            beginProgress=True,
            status="Calculating correspondence points with Embree...",
            maxValue=len(tar_mapping_points)
        )
    
    # Initialize results array
    raycast_result_array = [[] for _ in range(len(tar_mapping_points))]
    
    # Temporary buffers for ray data
    ray_origins = np.zeros((batch_size, 3), dtype=np.float32)
    ray_directions = np.zeros((batch_size, 3), dtype=np.float32)
    ray_data = np.zeros(batch_size, dtype=[
        ("vertex_idx", np.int32),
        ("from_point", np.float32, (3,)),
        ("node_weight", np.float32),
        ("target_distance", np.float32)
    ])
    
    # Process each target vertex
    for current_vert, mapping_result in enumerate(tar_mapping_points):
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1)
        
        # Skip empty mapping points
        if not mapping_result.node_array:
            continue
        
        target_vertex_idx = mapping_result.vertex_index
        
        # Get target vertex position
        if hasattr(mapping_result, "target_position"):
            target_vertex_pos = mapping_result.target_position
        else:
            raise ValueError("Target position not available in mapping result")
        
        # Process each mapping point
        for current_node in mapping_result.node_array:
            # Current bone index and weight
            current_tar_bone_index = current_node.bone_index
            current_node_weight = current_node.weight
            
            # Skip nodes with very small weights
            if current_node_weight < 0.001:
                continue
            
            # Vector from mapping point to vertex
            current_tar_p = current_node.point
            current_tar_pv = target_vertex_pos - current_tar_p
            
            # Normalize direction vector
            current_tar_pv_norm = np.linalg.norm(current_tar_pv)
            if current_tar_pv_norm < 1e-10:
                continue
                
            current_tar_normal_pv = current_tar_pv / current_tar_pv_norm
            
            # Target bone information
            tar_start_joint_index = tar_bone_group[current_tar_bone_index].start_joint_index
            tar_end_joint_index = tar_bone_group[current_tar_bone_index].end_joint_index
            
            current_tar_bone_start_point = tar_joint_group[tar_start_joint_index].position
            current_tar_bone_end_point = tar_joint_group[tar_end_joint_index].position
            current_tar_bone_v = current_tar_bone_end_point - current_tar_bone_start_point
            
            # Calculate distance ratio along bone
            current_tar_bone_v_norm = np.linalg.norm(current_tar_bone_v)
            if current_tar_bone_v_norm < 1e-10:
                continue
                
            tar_distance = np.linalg.norm(current_tar_p - current_tar_bone_start_point) / current_tar_bone_v_norm
            
            # Source bone information
            src_start_joint_index = src_joint_index[tar_start_joint_index]
            src_end_joint_index = src_joint_index[tar_end_joint_index]
            
            if src_start_joint_index == -1 or src_end_joint_index == -1:
                continue
            
            current_src_bone_start_point = src_joint_group[src_start_joint_index].position
            current_src_bone_end_point = src_joint_group[src_end_joint_index].position
            current_src_bone_v = current_src_bone_end_point - current_src_bone_start_point
            
            # Corresponding point on source bone (using same distance ratio)
            p = current_src_bone_start_point + current_src_bone_v * tar_distance
            p = np.array(p).squeeze()
            
            # Direction vector
            d = np.array(current_tar_normal_pv).squeeze()
            
            # Generate sample directions
            sample_directions = rand_cone_vector(d, sample_degree, sample_number)
            
            # Process rays in batches
            n_rays = len(sample_directions)
            n_batches = (n_rays + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_rays)
                current_batch_size = end_idx - start_idx
                
                # Prepare batch data
                ray_origins[:current_batch_size] = p
                ray_directions[:current_batch_size] = sample_directions[start_idx:end_idx]
                ray_data["vertex_idx"][:current_batch_size] = current_vert
                ray_data["from_point"][:current_batch_size] = p
                ray_data["node_weight"][:current_batch_size] = current_node_weight
                ray_data["target_distance"][:current_batch_size] = current_tar_pv_norm
                
                # Run Embree raycast
                res = scene.run(ray_origins[:current_batch_size], ray_directions[:current_batch_size], output=1)
                
                # Process hit rays
                hit_mask = res["geomID"] >= 0
                if np.any(hit_mask):
                    # Extract hit data
                    hit_indices = np.where(hit_mask)[0]
                    primIDs = res["primID"][hit_mask]
                    ts = res["tfar"][hit_mask]
                    us = res["u"][hit_mask]
                    vs = res["v"][hit_mask]
                    
                    # Create result for each hit
                    for i, hit_idx in enumerate(hit_indices):
                        primID = primIDs[i]
                        t = ts[i]
                        u = us[i]
                        v = vs[i]
                        w = 1.0 - u - v
                        
                        # Triangle vertex coordinates
                        v0 = embree_triangles[primID, 0]
                        v1 = embree_triangles[primID, 1]
                        v2 = embree_triangles[primID, 2]
                        
                        # Calculate intersection point (barycentric coordinates)
                        intersection_point = w * v0 + u * v1 + v * v2
                        
                        # Get original data
                        idx = hit_idx
                        vertex_idx = ray_data["vertex_idx"][idx]
                        from_point = ray_data["from_point"][idx]
                        node_weight = ray_data["node_weight"][idx]
                        target_distance = ray_data["target_distance"][idx]
                        
                        # Create result node
                        result_node = {
                            "from_point": from_point,
                            "point": intersection_point,
                            "triangle_index": int(primID),
                            "weight": float(node_weight),
                            "relate_distance": float(target_distance / t)
                        }
                        
                        raycast_result_array[vertex_idx].append(result_node)
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)
    
    return raycast_result_array


def perform_standard_raycast(
        tar_mapping_points,
        src_triangles,
        src_triangle_indices,
        sample_number,
        sample_degree,
        src_joint_group=None,
        tar_joint_group=None,
        src_bone_group=None,
        tar_bone_group=None,
        match_joint_trees_func=None
):
    """
    Perform standard raycast for correspondence point search.
    
    This is a fallback method when Embree is not available. It uses standard
    ray-triangle intersection tests.
    
    Args:
        tar_mapping_points (list): Target mapping points
        src_triangles (np.ndarray): Source mesh vertices
        src_triangle_indices (np.ndarray): Source mesh triangle indices
        sample_number (int): Number of sample rays
        sample_degree (float): Angle range for sampling (degrees)
        src_joint_group (list, optional): Source joint group
        tar_joint_group (list, optional): Target joint group
        src_bone_group (list, optional): Source bone group
        tar_bone_group (list, optional): Target bone group
        match_joint_trees_func (callable, optional): Function to match joint trees
        
    Returns:
        list: Raycast result array
    """
    from .geometry import rand_cone_vector, ray_triangle_intersection
    
    # Create joint mapping from target to source
    if match_joint_trees_func is None:
        from .alignment import match_joint_trees
        match_joint_trees_func = match_joint_trees
    
    src_joint_index = match_joint_trees_func(tar_joint_group, src_joint_group)
    
    # Set up progress bar
    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
            bar,
            edit=True,
            beginProgress=True,
            status="Calculating correspondence points...",
            maxValue=len(tar_mapping_points)
        )
    
    # Initialize results array
    raycast_result_array = [[] for _ in range(len(tar_mapping_points))]
    
    # Convert triangles to numpy array
    src_triangles_np = np.asarray(src_triangles, dtype=np.float32)
    src_triangle_indices_np = np.asarray(src_triangle_indices, dtype=np.int32)
    
    # Get number of source triangles
    src_num_triangles = len(src_triangle_indices) // 3
    
    # Process each target vertex
    for current_vert, mapping_result in enumerate(tar_mapping_points):
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1)
        
        # Skip empty mapping points
        if not mapping_result.node_array:
            continue
        
        target_vertex_idx = mapping_result.vertex_index
        
        # Get target vertex position
        if hasattr(mapping_result, "target_position"):
            target_vertex_pos = mapping_result.target_position
        else:
            raise ValueError("Target position not available in mapping result")
        
        # Process each mapping point
        for current_node in mapping_result.node_array:
            # Current bone index and weight
            current_tar_bone_index = current_node.bone_index
            current_node_weight = current_node.weight
            
            # Skip nodes with very small weights
            if current_node_weight < 0.001:
                continue
            
            # Vector from mapping point to vertex
            current_tar_p = current_node.point
            current_tar_pv = target_vertex_pos - current_tar_p
            
            # Normalize direction vector
            current_tar_pv_norm = np.linalg.norm(current_tar_pv)
            if current_tar_pv_norm < 1e-10:
                continue
                
            current_tar_normal_pv = current_tar_pv / current_tar_pv_norm
            
            # Target bone information
            tar_start_joint_index = tar_bone_group[current_tar_bone_index].start_joint_index
            tar_end_joint_index = tar_bone_group[current_tar_bone_index].end_joint_index
            
            current_tar_bone_start_point = tar_joint_group[tar_start_joint_index].position
            current_tar_bone_end_point = tar_joint_group[tar_end_joint_index].position
            current_tar_bone_v = current_tar_bone_end_point - current_tar_bone_start_point
            
            # Calculate distance ratio along bone
            current_tar_bone_v_norm = np.linalg.norm(current_tar_bone_v)
            if current_tar_bone_v_norm < 1e-10:
                continue
                
            tar_distance = np.linalg.norm(current_tar_p - current_tar_bone_start_point) / current_tar_bone_v_norm
            
            # Source bone information
            src_start_joint_index = src_joint_index[tar_start_joint_index]
            src_end_joint_index = src_joint_index[tar_end_joint_index]
            
            if src_start_joint_index == -1 or src_end_joint_index == -1:
                continue
            
            current_src_bone_start_point = src_joint_group[src_start_joint_index].position
            current_src_bone_end_point = src_joint_group[src_end_joint_index].position
            current_src_bone_v = current_src_bone_end_point - current_src_bone_start_point
            
            # Corresponding point on source bone (using same distance ratio)
            p = current_src_bone_start_point + current_src_bone_v * tar_distance
            p = np.array(p).squeeze()
            
            # Direction vector
            d = np.array(current_tar_normal_pv).squeeze()
            
            # Generate sample directions
            sample_directions = rand_cone_vector(d, sample_degree, sample_number)
            
            # Process each ray
            for ray_dir in sample_directions:
                closest_hit = None
                closest_t = float('inf')
                closest_tri_idx = -1
                
                # Test intersection with each triangle
                for tri_idx in range(src_num_triangles):
                    v0_idx = src_triangle_indices_np[tri_idx * 3 + 0]
                    v1_idx = src_triangle_indices_np[tri_idx * 3 + 1]
                    v2_idx = src_triangle_indices_np[tri_idx * 3 + 2]
                    
                    v0 = src_triangles_np[v0_idx]
                    v1 = src_triangles_np[v1_idx]
                    v2 = src_triangles_np[v2_idx]
                    
                    hit, hit_point, t = ray_triangle_intersection(p, ray_dir, v0, v1, v2)
                    
                    if hit and t < closest_t:
                        closest_hit = hit_point
                        closest_t = t
                        closest_tri_idx = tri_idx
                
                # If a hit was found, add it to the results
                if closest_hit is not None:
                    result_node = {
                        "from_point": p,
                        "point": closest_hit,
                        "triangle_index": int(closest_tri_idx),
                        "weight": float(current_node_weight),
                        "relate_distance": float(current_tar_pv_norm / closest_t)
                    }
                    
                    raycast_result_array[current_vert].append(result_node)
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)
    
    return raycast_result_array