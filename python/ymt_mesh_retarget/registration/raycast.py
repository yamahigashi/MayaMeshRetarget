# -*- coding: utf-8 -*-
"""
Raycast utilities for mesh registration.

This module provides functions for raycasting operations using Embree or fallback methods.
"""
import typing
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import math

import numpy as np
from numpy.typing import NDArray
from maya import cmds, mel
from maya.api import OpenMaya as om

# Try to import Embree
try:
    from embreex import rtcore_scene as rtcs  # type: ignore
    from embreex.mesh_construction import TriangleMesh  # type: ignore
    EMBREE_AVAILABLE = True
except ImportError:
    EMBREE_AVAILABLE = False
    logger.warning("embreex library not found. Using standard raycasting instead.")

from . import geometry
from .core import (
    RaycastResult,
    Vector3,
    TriangleIndex,
    RegistrationOptions
)
from .utils import (
    get_matched_info
)

if typing.TYPE_CHECKING:
    from ..objects import (
        MeshObject
    )
    from .core import (
        MappingResult,
        JointNode,
        BoneNode,
    )

# Use centralized logger
from ..logger import logger


class RaycastEngine:
    """Base class for raycast engines
    
    This class provides a common interface for different raycast implementations.
    """
    
    def __init__(self, triangles: NDArray[np.float32], triangle_indices: NDArray[np.int32]):
        """Initialize the raycast engine
        
        Args:
            triangles: Triangle vertex positions
            triangle_indices: Triangle indices
        """
        self.triangles = triangles
        self.triangle_indices = triangle_indices
        self.num_triangles = len(triangle_indices) // 3
        self._prepare_scene()
    
    def _prepare_scene(self) -> None:
        """Prepare the scene for raycasting
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _prepare_scene()")
    
    def cast_ray(self, origin: Vector3, direction: Vector3) -> Optional[Dict[str, Any]]:
        """Cast a ray through the scene
        
        Args:
            origin: Ray origin point
            direction: Ray direction vector
        
        Returns:
            Dict containing hit information or None if no hit
        """
        raise NotImplementedError("Subclasses must implement cast_ray()")
    
    def cast_rays(self, 
                  origins: NDArray[np.float32], 
                  directions: NDArray[np.float32]
                 ) -> Dict[str, NDArray]:
        """Cast multiple rays through the scene
        
        Args:
            origins: Ray origin points (N, 3)
            directions: Ray direction vectors (N, 3)
        
        Returns:
            Dict containing hit information for all rays
        """
        raise NotImplementedError("Subclasses must implement cast_rays()")
    
    def cleanup(self) -> None:
        """Clean up resources
        
        This method should be implemented by subclasses.
        """
        pass


class EmbreeRaycastEngine(RaycastEngine):
    """Embree-based raycast engine
    
    This class uses Intel's Embree library for high-performance raycasting.
    """
    
    def __init__(self, triangles: NDArray[np.float32], triangle_indices: NDArray[np.int32]):
        """Initialize the Embree raycast engine
        
        Args:
            triangles: Triangle vertex positions
            triangle_indices: Triangle indices
        """
        if not EMBREE_AVAILABLE:
            raise ImportError("Embree library is not available")
        
        super().__init__(triangles, triangle_indices)
    
    def _prepare_scene(self) -> None:
        """Prepare the Embree scene for raycasting
        
        Converts triangle data to Embree's format and builds BVH acceleration structure.
        """
        # Convert triangles to Embree's format (num_triangles, 3, 3)
        embree_triangles = np.zeros((self.num_triangles, 3, 3), dtype=np.float32)
        
        for i in range(self.num_triangles):
            v0_idx = self.triangle_indices[i * 3 + 0]
            v1_idx = self.triangle_indices[i * 3 + 1]
            v2_idx = self.triangle_indices[i * 3 + 2]
            
            embree_triangles[i, 0] = self.triangles[v0_idx]
            embree_triangles[i, 1] = self.triangles[v1_idx]
            embree_triangles[i, 2] = self.triangles[v2_idx]
        
        # Create Embree scene and mesh
        self.scene = rtcs.EmbreeScene()
        self.mesh = TriangleMesh(self.scene, embree_triangles)
        self.embree_triangles = embree_triangles
    
    def cast_ray(self, origin: Vector3, direction: Vector3) -> Optional[Dict[str, Any]]:
        """Cast a single ray through the Embree scene
        
        Args:
            origin: Ray origin point
            direction: Ray direction vector
        
        Returns:
            Dict containing hit information or None if no hit
        """
        # Convert to numpy arrays
        origin_np = np.asarray(origin, dtype=np.float32).reshape(1, 3)
        direction_np = np.asarray(direction, dtype=np.float32).reshape(1, 3)
        
        # Cast ray
        hit = self.scene.run(origin_np, direction_np, output=1)
        
        # Check if hit
        if hit["geomID"][0] >= 0:
            result = {
                "primID": hit["primID"][0],
                "tfar": hit["tfar"][0],
                "u": hit["u"][0],
                "v": hit["v"][0]
            }
            return result
        
        return None
    
    def cast_rays(self, 
                  origins: NDArray[np.float32], 
                  directions: NDArray[np.float32]
                 ) -> Dict[str, NDArray]:
        """Cast multiple rays through the Embree scene
        
        Args:
            origins: Ray origin points (N, 3)
            directions: Ray direction vectors (N, 3)
        
        Returns:
            Dict containing hit information for all rays
        """
        return self.scene.run(origins, directions, output=1)
    
    def cleanup(self) -> None:
        """Clean up Embree resources
        
        Makes sure to release Embree resources to avoid memory leaks.
        """
        # These are automatically cleaned up by Python's garbage collector
        # but explicitly setting to None helps ensure timely cleanup
        self.scene = None
        self.mesh = None


class StandardRaycastEngine(RaycastEngine):
    """Standard raycast engine using NumPy with BVH acceleration
    
    This class provides a fallback for when Embree is not available, using a
    Bounding Volume Hierarchy (BVH) for accelerated ray-triangle intersection tests.
    """
    
    def __init__(self, triangles: NDArray[np.float32], triangle_indices: NDArray[np.int32]):
        """Initialize the standard raycast engine
        
        Args:
            triangles: Triangle vertex positions
            triangle_indices: Triangle indices
        """
        super().__init__(triangles, triangle_indices)
    
    def _prepare_scene(self) -> None:
        """Prepare data for raycasting using a BVH
        
        Creates a BVH acceleration structure for efficient ray-triangle testing.
        """
        from .spatial import BVH
        
        # Initialize BVH
        self.bvh = BVH(
            self.triangles,
            self.triangle_indices,
            max_triangles_per_leaf=8,
            max_depth=20
        )
        
        # Also store the triangle vertices for direct access if needed
        self.triangle_vertices = self.bvh.triangle_vertices
    
    def cast_ray(self, origin: Vector3, direction: Vector3) -> Optional[Dict[str, Any]]:
        """Cast a ray through the scene using BVH acceleration
        
        Args:
            origin: Ray origin point
            direction: Ray direction vector
        
        Returns:
            Dict containing hit information or None if no hit
        """
        # Use BVH for accelerated intersection testing
        return self.bvh.ray_intersection(origin, direction)
    
    def cast_rays(self, 
                  origins: NDArray[np.float32], 
                  directions: NDArray[np.float32]
                 ) -> Dict[str, NDArray]:
        """Cast multiple rays through the scene using BVH acceleration
        
        Args:
            origins: Ray origin points (N, 3)
            directions: Ray direction vectors (N, 3)
        
        Returns:
            Dict containing hit information for all rays
        """
        # Use BVH for accelerated batch intersection testing
        return self.bvh.ray_intersections_batch(origins, directions)
    
    def cleanup(self) -> None:
        """Clean up resources
        
        Explicitly release BVH and triangle data to help garbage collection.
        """
        self.bvh = None
        self.triangle_vertices = None
        super().cleanup()


def get_raycast_engine(triangles: NDArray[np.float32], 
                       triangle_indices: NDArray[np.int32],
                       force_standard: bool = False) -> RaycastEngine:
    """Factory function to get the appropriate raycast engine
    
    Args:
        triangles: Triangle vertex positions
        triangle_indices: Triangle indices
        force_standard: Force using the standard raycast engine
    
    Returns:
        RaycastEngine implementation
    """
    if EMBREE_AVAILABLE and not force_standard:
        logger.info("Using Embree raycast engine")
        return EmbreeRaycastEngine(triangles, triangle_indices)
    else:
        logger.info("Using standard raycast engine")
        return StandardRaycastEngine(triangles, triangle_indices)


def build_embree_scene_from_source(src_triangles: NDArray[np.float32]) -> Tuple[Any, Any]:
    """
    Build an Embree scene from source mesh triangle array.
    
    Args:
        src_triangles: shape = (num_tri, 3, 3)
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
        src_mesh: "MeshObject",
        tar_mesh: "MeshObject",
        tar_mapping_points: List["MappingResult"],
        src_triangles: NDArray[np.float64],
        src_triangle_indices: NDArray[np.int32],
        sample_number: int,
        sample_degree: float,
        src_joint_group: List["JointNode"],
        tar_joint_group: List["JointNode"],
        src_bone_group: List["BoneNode"],
        tar_bone_group: List["BoneNode"],
        batch_size: int = 1024,
        max_triangles: int = -1,
        force_standard_raycast: bool = False,
        num_threads: int = 4
) -> List[List[RaycastResult]]:
    """
    Perform raycasting to find correspondence points between meshes.
    
    This is a high-level function that chooses between Embree-based raycasting
    or standard raycasting based on availability and user preference.
    
    Args:
        src_mesh: Source mesh object
        tar_mesh: Target mesh object
        tar_mapping_points: Target mapping points
        src_triangles: Source mesh vertices (N, 3)
        src_triangle_indices: Source mesh triangle indices (T*3)
        sample_number: Number of sample rays to cast
        sample_degree: Cone angle for sample rays (degrees)
        src_joint_group: Source joint group
        tar_joint_group: Target joint group
        src_bone_group: Source bone group
        tar_bone_group: Target bone group
        batch_size: Batch size for processing rays
        max_triangles: Maximum number of triangles to process (-1 for all)
        force_standard_raycast: Force using standard raycasting instead of Embree
        num_threads: Number of threads to use for parallel processing
        
    Returns:
        List of raycast results for each target vertex
    """
    import concurrent.futures
    from functools import partial
    
    # Validate inputs
    if not EMBREE_AVAILABLE and not force_standard_raycast:
        logger.warning("Embree library is not available. Using standard raycasting.")
        force_standard_raycast = True

    # Create joint mapping from target to source
    src_indices, _, _ = get_matched_info(src_joint_group, tar_joint_group)
    
    # Convert input data to NumPy arrays
    src_triangles_np = np.asarray(src_triangles, dtype=np.float32)
    src_triangle_indices_np = np.asarray(src_triangle_indices, dtype=np.int32)
    
    # Get number of source triangles
    src_num_triangles = len(src_triangle_indices_np) // 3
    
    # Limit triangles if requested
    if max_triangles > 0 and max_triangles < src_num_triangles:
        src_num_triangles = max_triangles
        src_triangle_indices_np = src_triangle_indices_np[:max_triangles * 3]
    
    # Set up progress bar
    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
            bar,
            edit=True,
            beginProgress=True,
            status="Setting up raycast acceleration structure...",
            maxValue=100
        )
        cmds.progressBar(bar, edit=True, step=10)
    
    # Initialize raycast engine
    engine = get_raycast_engine(
        src_triangles_np, 
        src_triangle_indices_np,
        force_standard=force_standard_raycast
    )
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, step=40)
    
    # Initialize result array
    raycast_result_array = [[] for _ in range(len(tar_mapping_points))]
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, step=50)
        cmds.progressBar(bar, edit=True, endProgress=True)
        
        # Set up new progress bar for processing
        cmds.progressBar(
            bar,
            edit=True,
            beginProgress=True,
            status="Calculating correspondence points...",
            maxValue=len(tar_mapping_points)
        )
    
    # Pre-compute bone mapping
    bone_mapping = {}
    for i, bone in enumerate(tar_bone_group):
        tar_start_idx = bone.start_joint_index
        tar_end_idx = bone.end_joint_index
        
        if tar_start_idx < 0 or tar_end_idx < 0:
            continue
            
        src_start_idx = src_indices[tar_start_idx] if tar_start_idx < len(src_indices) else -1
        src_end_idx = src_indices[tar_end_idx] if tar_end_idx < len(src_indices) else -1
        
        if src_start_idx < 0 or src_end_idx < 0:
            continue
            
        bone_mapping[i] = {
            'tar_start_idx': tar_start_idx,
            'tar_end_idx': tar_end_idx,
            'src_start_idx': src_start_idx,
            'src_end_idx': src_end_idx
        }
    
    # Prepare target mesh points for faster access
    target_points = tar_mesh.get_points()
    
    # Define worker function for parallel processing
    def process_vertex_chunk(vertex_indices, engine, target_points):
        results = [[] for _ in range(len(tar_mapping_points))]
        
        for current_vert in vertex_indices:
            mapping_result = tar_mapping_points[current_vert]
            
            # Skip if no mapping points
            if not mapping_result.node_array:
                continue
            
            target_vertex_idx = mapping_result.vertex_index
            target_vertex_pos = target_points[target_vertex_idx]
            
            # Process each mapping point
            for current_node in mapping_result.node_array:
                # Get current bone index and weight
                current_tar_bone_index = current_node.bone_index
                current_node_weight = current_node.weight
                
                # Skip if weight is very small
                if current_node_weight < 0.001:
                    continue
                
                # Skip if bone mapping not found
                if current_tar_bone_index not in bone_mapping:
                    continue
                    
                # Get bone mapping
                bone_map = bone_mapping[current_tar_bone_index]
                
                # Get vector from mapping point to vertex
                current_tar_p = current_node.point
                current_tar_pv = target_vertex_pos - current_tar_p
                
                # Normalize direction vector
                current_tar_pv_norm = np.linalg.norm(current_tar_pv)
                if current_tar_pv_norm < 1e-10:
                    continue
                current_tar_normal_pv = current_tar_pv / current_tar_pv_norm
                
                # Get target bone information
                tar_start_joint_index = bone_map['tar_start_idx']
                tar_end_joint_index = bone_map['tar_end_idx']
                
                current_tar_bone_start_point = tar_joint_group[tar_start_joint_index].position
                current_tar_bone_end_point = tar_joint_group[tar_end_joint_index].position
                current_tar_bone_v = current_tar_bone_end_point - current_tar_bone_start_point
                
                # Calculate distance ratio along bone
                current_tar_bone_v_norm = np.linalg.norm(current_tar_bone_v)
                if current_tar_bone_v_norm < 1e-10:
                    continue
                tar_distance = np.linalg.norm(current_tar_p - current_tar_bone_start_point) / current_tar_bone_v_norm
                
                # Get source bone information
                src_start_joint_index = bone_map['src_start_idx']
                src_end_joint_index = bone_map['src_end_idx']
                
                if src_start_joint_index == -1 or src_end_joint_index == -1:
                    continue
                
                current_src_bone_start_point = src_joint_group[src_start_joint_index].position
                current_src_bone_end_point = src_joint_group[src_end_joint_index].position
                current_src_bone_v = current_src_bone_end_point - current_src_bone_start_point
                
                # Calculate corresponding point on source bone
                p = current_src_bone_start_point + current_src_bone_v * tar_distance
                p = np.array(p).squeeze()
                
                # Get direction vector
                d = np.array(current_tar_normal_pv).squeeze()
                
                # Generate sample directions
                sample_directions = geometry.rand_cone_vector(d, sample_degree, sample_number)
                
                # Cast rays in batches
                n_rays = len(sample_directions)
                
                # Prepare ray arrays
                ray_origins = np.full((n_rays, 3), p, dtype=np.float32)
                ray_data = np.zeros(n_rays, dtype=[
                    ("vertex_idx", np.int32),
                    ("from_point", np.float32, (3,)),
                    ("node_weight", np.float32),
                    ("target_distance", np.float32)
                ])
                
                # Fill ray data
                ray_data["vertex_idx"][:] = current_vert
                ray_data["from_point"][:] = p
                ray_data["node_weight"][:] = current_node_weight
                ray_data["target_distance"][:] = current_tar_pv_norm
                
                # Process rays in batches
                for batch_start in range(0, n_rays, batch_size):
                    batch_end = min(batch_start + batch_size, n_rays)
                    batch_size_actual = batch_end - batch_start
                    
                    # Cast rays for this batch
                    batch_origins = ray_origins[batch_start:batch_end]
                    batch_directions = sample_directions[batch_start:batch_end]
                    
                    # Cast rays
                    res = engine.cast_rays(batch_origins, batch_directions)
                    
                    # Process hits
                    hit_mask = res["geomID"] >= 0
                    if np.any(hit_mask):
                        # Get hit data
                        hit_indices = np.where(hit_mask)[0]
                        primIDs = res["primID"][hit_mask]
                        ts = res["tfar"][hit_mask]
                        us = res["u"][hit_mask]
                        vs = res["v"][hit_mask]
                        
                        # Process each hit
                        for i, hit_idx in enumerate(hit_indices):
                            # Get global index in the batch
                            global_idx = batch_start + hit_idx
                            
                            primID = primIDs[i]
                            t = ts[i]
                            u = us[i]
                            v = vs[i]
                            w = 1.0 - u - v
                            
                            # Get triangle vertices
                            triangle_idx = primID
                            v0_idx = src_triangle_indices_np[triangle_idx * 3 + 0]
                            v1_idx = src_triangle_indices_np[triangle_idx * 3 + 1]
                            v2_idx = src_triangle_indices_np[triangle_idx * 3 + 2]
                            
                            v0 = src_triangles_np[v0_idx]
                            v1 = src_triangles_np[v1_idx]
                            v2 = src_triangles_np[v2_idx]
                            
                            # Calculate intersection point (barycentric coordinates)
                            intersection_point = w * v0 + u * v1 + v * v2
                            
                            # Get ray data
                            vertex_idx = ray_data["vertex_idx"][global_idx]
                            from_point = ray_data["from_point"][global_idx]
                            node_weight = ray_data["node_weight"][global_idx]
                            target_distance = ray_data["target_distance"][global_idx]
                            
                            # Create result node
                            result_node = RaycastResult(
                                from_point=from_point,
                                point=intersection_point,
                                triangle_index=int(primID),
                                weight=float(node_weight),
                                relate_distance=float(target_distance / t)
                            )
                            
                            results[vertex_idx].append(result_node)
        
        return results
    
    # Split vertices into chunks for parallel processing
    vertex_indices = list(range(len(tar_mapping_points)))
    chunk_size = max(1, len(vertex_indices) // num_threads)
    vertex_chunks = [vertex_indices[i:i+chunk_size] for i in range(0, len(vertex_indices), chunk_size)]
    
    # Use ThreadPoolExecutor for parallel processing
    # Note: We're creating a partial function with engine and target_points already bound
    process_func = partial(process_vertex_chunk, engine=engine, target_points=target_points)
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunk_results = list(executor.map(process_func, vertex_chunks))
    
    # Update progress bar
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, progress=len(tar_mapping_points))
    
    # Merge results from all chunks
    for chunk_result in chunk_results:
        for vertex_idx, results in enumerate(chunk_result):
            if results:  # Only append if there are results
                raycast_result_array[vertex_idx].extend(results)
    
    # Clean up
    engine.cleanup()
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)
    
    return raycast_result_array


def perform_raycast_with_options(
        src_mesh: "MeshObject",
        tar_mesh: "MeshObject",
        tar_mapping_points: List["MappingResult"],
        src_triangles: NDArray[np.float64],
        src_triangle_indices: NDArray[np.int32],
        src_joint_group: List["JointNode"],
        tar_joint_group: List["JointNode"],
        src_bone_group: List["BoneNode"],
        tar_bone_group: List["BoneNode"],
        options: RegistrationOptions,
        num_threads: int = 4
) -> List[List[RaycastResult]]:
    """Perform raycasting with registration options
    
    Convenience function that uses RegistrationOptions to configure raycasting.
    
    Args:
        src_mesh: Source mesh object
        tar_mesh: Target mesh object
        tar_mapping_points: Target mapping points
        src_triangles: Source mesh vertices
        src_triangle_indices: Source mesh triangle indices
        src_joint_group: Source joint group
        tar_joint_group: Target joint group
        src_bone_group: Source bone group
        tar_bone_group: Target bone group
        options: Registration options
        num_threads: Number of threads to use for parallel processing
        
    Returns:
        List of raycast results for each target vertex
    """
    return perform_raycast(
        src_mesh=src_mesh,
        tar_mesh=tar_mesh,
        tar_mapping_points=tar_mapping_points,
        src_triangles=src_triangles,
        src_triangle_indices=src_triangle_indices,
        sample_number=options.sample_number,
        sample_degree=options.sample_degree,
        src_joint_group=src_joint_group,
        tar_joint_group=tar_joint_group,
        src_bone_group=src_bone_group,
        tar_bone_group=tar_bone_group,
        batch_size=options.batch_size,
        max_triangles=options.max_triangles,
        num_threads=num_threads
    )
