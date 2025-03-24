"""Raycast utilities for mesh registration.

This module provides functions for raycasting operations using Embree or fallback methods.
"""

import logging
import random
import typing
from typing import Any, Optional

import numpy as np
from maya import cmds, mel
from numpy.typing import NDArray


# Try to import Embree
try:
    from embreex import rtcore_scene as rtcs  # type: ignore
    from embreex.mesh_construction import TriangleMesh  # type: ignore

    EMBREE_AVAILABLE = True
except ImportError:
    EMBREE_AVAILABLE = False
    cmds.warning("embreex library not found. Using standard raycasting instead.")

from ..util import timeit
from . import geometry
from .core import RaycastResult, RegistrationOptions, Vector3
from .utils import get_matched_info


if typing.TYPE_CHECKING:
    from ..objects import MeshObject
    from .core import (
        BoneNode,
        JointNode,
        MappingResult,
    )

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RaycastEngine:
    """Base class for raycast engines.

    This class provides a common interface for different raycast implementations.
    """

    def __init__(self, triangles: NDArray[np.float32], triangle_indices: NDArray[np.int32]) -> None:
        """Initialize the raycast engine.

        Args:
            triangles: Triangle vertex positions
            triangle_indices: Triangle indices
        """
        self.triangles = triangles
        self.triangle_indices = triangle_indices
        self.num_triangles = len(triangle_indices) // 3
        self._prepare_scene()

    def _prepare_scene(self) -> None:
        """Prepare the scene for raycasting.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _prepare_scene()")

    def cast_ray(self, origin: Vector3, direction: Vector3) -> Optional[dict[str, Any]]:
        """Cast a ray through the scene.

        Args:
            origin: Ray origin point
            direction: Ray direction vector

        Returns:
            Dict containing hit information or None if no hit
        """
        raise NotImplementedError("Subclasses must implement cast_ray()")

    def cast_rays(
        self,
        origins: NDArray[np.float32],
        directions: NDArray[np.float32],
    ) -> dict[str, NDArray]:
        """Cast multiple rays through the scene.

        Args:
            origins: Ray origin points (N, 3)
            directions: Ray direction vectors (N, 3)

        Returns:
            Dict containing hit information for all rays
        """
        raise NotImplementedError("Subclasses must implement cast_rays()")

    def cleanup(self) -> None:
        """Clean up resources.

        This method should be implemented by subclasses.
        """
        pass


class EmbreeRaycastEngine(RaycastEngine):
    """Embree-based raycast engine.

    This class uses Intel's Embree library for high-performance raycasting.
    """

    def __init__(self, triangles: NDArray[np.float32], triangle_indices: NDArray[np.int32]) -> None:
        """Initialize the Embree raycast engine.

        Args:
            triangles: Triangle vertex positions
            triangle_indices: Triangle indices
        """
        if not EMBREE_AVAILABLE:
            raise ImportError("Embree library is not available")

        super().__init__(triangles, triangle_indices)

    def _prepare_scene(self) -> None:
        """Prepare the Embree scene for raycasting.

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

    def cast_ray(self, origin: Vector3, direction: Vector3) -> Optional[dict[str, Any]]:
        """Cast a single ray through the Embree scene.

        Args:
            origin: Ray origin point
            direction: Ray direction vector

        Returns:
            Dict containing hit information or None if no hit
        """
        if self.scene is None:
            raise ValueError("Embree scene is not initialized")

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
                "v": hit["v"][0],
            }
            return result

        return None

    def cast_rays(
        self,
        origins: NDArray[np.float32],
        directions: NDArray[np.float32],
    ) -> dict[str, NDArray]:
        """Cast multiple rays through the Embree scene.

        Args:
            origins: Ray origin points (N, 3)
            directions: Ray direction vectors (N, 3)

        Returns:
            Dict containing hit information for all rays
        """
        if self.scene is None:
            raise ValueError("Embree scene is not initialized")

        return self.scene.run(origins, directions, output=1)

    def cleanup(self) -> None:
        """Clean up Embree resources.

        Makes sure to release Embree resources to avoid memory leaks.
        """
        # These are automatically cleaned up by Python's garbage collector
        # but explicitly setting to None helps ensure timely cleanup
        self.scene = None
        self.mesh = None


class StandardRaycastEngine(RaycastEngine):
    """Standard raycast engine using NumPy.

    This class provides a fallback for when Embree is not available.
    """

    def __init__(self, triangles: NDArray[np.float32], triangle_indices: NDArray[np.int32]) -> None:
        """Initialize the standard raycast engine.

        Args:
            triangles: Triangle vertex positions
            triangle_indices: Triangle indices
        """
        super().__init__(triangles, triangle_indices)

    def _prepare_scene(self) -> None:
        """Prepare data for raycasting.

        Creates a flat array of triangles for efficient access.
        """
        # Convert triangles to a flat array for efficient access
        self.triangle_vertices = np.zeros((self.num_triangles, 3, 3), dtype=np.float32)

        for i in range(self.num_triangles):
            v0_idx = self.triangle_indices[i * 3 + 0]
            v1_idx = self.triangle_indices[i * 3 + 1]
            v2_idx = self.triangle_indices[i * 3 + 2]

            self.triangle_vertices[i, 0] = self.triangles[v0_idx]
            self.triangle_vertices[i, 1] = self.triangles[v1_idx]
            self.triangle_vertices[i, 2] = self.triangles[v2_idx]

    def cast_ray(self, origin: Vector3, direction: Vector3) -> Optional[dict[str, Any]]:
        """Cast a ray through the scene using standard ray-triangle intersection.

        Args:
            origin: Ray origin point
            direction: Ray direction vector

        Returns:
            Dict containing hit information or None if no hit
        """
        closest_hit = None
        closest_t = float("inf")
        closest_prim_id = -1
        closest_u = 0.0
        closest_v = 0.0

        # Normalize direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:
            return None
        direction = direction / direction_norm

        # Check each triangle
        for i in range(self.num_triangles):
            v0 = self.triangle_vertices[i, 0]
            v1 = self.triangle_vertices[i, 1]
            v2 = self.triangle_vertices[i, 2]

            # Ray-triangle intersection using Mテカller窶典rumbore algorithm
            hit, intersection, t, u, v = geometry.ray_triangle_intersection_with_uv(origin, direction, v0, v1, v2)

            if hit and t < closest_t:
                closest_hit = intersection
                closest_t = t
                closest_prim_id = i
                closest_u = u
                closest_v = v

        if closest_hit is not None:
            return {
                "primID": closest_prim_id,
                "tfar": closest_t,
                "u": closest_u,
                "v": closest_v,
            }

        return None

    def cast_rays(
        self,
        origins: NDArray[np.float32],
        directions: NDArray[np.float32],
    ) -> dict[str, NDArray]:
        """Cast multiple rays through the scene.

        Args:
            origins: Ray origin points (N, 3)
            directions: Ray direction vectors (N, 3)

        Returns:
            Dict containing hit information for all rays
        """
        # Initialize result arrays
        num_rays = origins.shape[0]
        prim_id = np.full(num_rays, -1, dtype=np.int32)
        geom_id = np.full(num_rays, -1, dtype=np.int32)
        tfar = np.full(num_rays, np.inf, dtype=np.float32)
        u = np.zeros(num_rays, dtype=np.float32)
        v = np.zeros(num_rays, dtype=np.float32)

        # Process each ray
        for ray_idx in range(num_rays):
            origin = origins[ray_idx]
            direction = directions[ray_idx]

            # Normalize direction
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-10:
                continue
            direction = direction / direction_norm

            closest_t = float("inf")
            closest_prim_id = -1
            closest_u = 0.0
            closest_v = 0.0

            # Check each triangle
            for i in range(self.num_triangles):
                v0 = self.triangle_vertices[i, 0]
                v1 = self.triangle_vertices[i, 1]
                v2 = self.triangle_vertices[i, 2]

                # Ray-triangle intersection
                hit, _, t, u_val, v_val = geometry.ray_triangle_intersection_with_uv(origin, direction, v0, v1, v2)

                if hit and t < closest_t:
                    closest_t = t
                    closest_prim_id = i
                    closest_u = u_val
                    closest_v = v_val

            if closest_prim_id >= 0:
                prim_id[ray_idx] = closest_prim_id
                geom_id[ray_idx] = 0  # We only have one geometry
                tfar[ray_idx] = closest_t
                u[ray_idx] = closest_u
                v[ray_idx] = closest_v

        return {
            "primID": prim_id,
            "geomID": geom_id,
            "tfar": tfar,
            "u": u,
            "v": v,
        }


def get_raycast_engine(
    triangles: NDArray[np.float32], triangle_indices: NDArray[np.int32], force_standard: bool = False,
) -> RaycastEngine:
    """Factory function to get the appropriate raycast engine.

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


def perform_raycast(
    src_mesh: "MeshObject",
    tar_mesh: "MeshObject",
    src_mapping_points: list["MappingResult"],
    tar_triangles: NDArray[np.float64],
    tar_triangle_indices: NDArray[np.int32],
    sample_number: int,
    sample_degree: float,
    src_joint_group: list["JointNode"],
    tar_joint_group: list["JointNode"],
    src_bone_group: list["BoneNode"],
    tar_bone_group: list["BoneNode"],  # noqa: ARG001
    batch_size: int = 1024,
    max_triangles: int = -1,
    sample_vertex_count: int = 1500,
    force_standard_raycast: bool = False,
) -> list[list[RaycastResult]]:
    """Perform raycasting to find correspondence points between meshes.

    This is a high-level function that chooses between Embree-based raycasting
    or standard raycasting based on availability and user preference.

    Args:
        src_mesh: Source mesh object
        tar_mesh: Target mesh object
        src_mapping_points: Source mapping points
        tar_triangles: Target mesh vertices (N, 3)
        tar_triangle_indices: Target mesh triangle indices (T*3)
        sample_number: Number of sample rays to cast
        sample_degree: Cone angle for sample rays (degrees)
        src_joint_group: Source joint group
        tar_joint_group: Target joint group
        src_bone_group: Source bone group
        tar_bone_group: Target bone group
        batch_size: Batch size for processing rays
        max_triangles: Maximum number of triangles to process (-1 for all)
        sample_vertex_count: Number of vertices to sample per target vertex
        force_standard_raycast: Force using standard raycasting instead of Embree

    Returns:
        List of raycast results for each target vertex
    """
    # Validate inputs
    if not EMBREE_AVAILABLE and not force_standard_raycast:
        logger.warning("Embree library is not available. Using standard raycasting.")
        force_standard_raycast = True

    # Create joint mapping from source to target
    src_indices, tar_indices, _names = get_matched_info(src_joint_group, tar_joint_group)

    # Create mapping from source to target
    src2tar_map = [-1] * len(src_joint_group)
    for pair_i, s_idx in enumerate(src_indices):
        t_idx = tar_indices[pair_i]
        src2tar_map[s_idx] = t_idx

    # Convert input data to NumPy arrays
    tar_triangles_np = np.asarray(tar_triangles, dtype=np.float32)
    tar_triangle_indices_np = np.asarray(tar_triangle_indices, dtype=np.int32)

    # Get number of target triangles
    tar_num_triangles = len(tar_triangle_indices_np) // 3

    # Limit triangles if requested
    if max_triangles > 0 and max_triangles < tar_num_triangles:
        tar_num_triangles = max_triangles
        tar_triangle_indices_np = tar_triangle_indices_np[: max_triangles * 3]

    # Set up progress bar
    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
            bar,
            edit=True,
            beginProgress=True,
            status="Setting up raycast acceleration structure...",
            maxValue=100,
        )
        cmds.progressBar(bar, edit=True, step=10)

    # Initialize raycast engine
    engine = get_raycast_engine(
        tar_triangles_np,
        tar_triangle_indices_np,
        force_standard=force_standard_raycast,
    )

    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, step=40)

    # Initialize result array
    raycast_result_array = [[] for _ in range(len(src_mapping_points))]

    # Initialize ray buffers
    ray_origins = np.zeros((batch_size, 3), dtype=np.float32)
    ray_directions = np.zeros((batch_size, 3), dtype=np.float32)
    ray_data = np.zeros(
        batch_size,
        dtype=[
            ("vertex_idx", np.int32),
            ("from_point", np.float32, (3,)),
            ("node_weight", np.float32),
            ("target_distance", np.float32),
        ],
    )

    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, step=50)
        cmds.progressBar(bar, edit=True, endProgress=True)

        # Set up new progress bar for processing
        cmds.progressBar(
            bar,
            edit=True,
            beginProgress=True,
            status="Calculating correspondence points...",
            maxValue=len(src_mapping_points),
        )

    seed = hash(tar_mesh.name + src_mesh.name)
    random.seed(seed)
    sample_vertex_count = min(sample_vertex_count, len(src_mapping_points))
    source_vertex_indices = random.sample(range(len(src_mapping_points)), sample_vertex_count)

    # Process each source vertex
    for current_vert, mapping_result in enumerate(src_mapping_points):
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1)

        # Randomly sample vertices
        if current_vert not in source_vertex_indices:
            continue

        # Skip if no mapping points
        if not mapping_result.node_array:
            continue

        vertex_idx = mapping_result.vertex_index
        vertex_pos = src_mesh.get_points()[vertex_idx]
        flag = vertex_idx == 1929
        if flag:
            print(f"vertex_idx: {vertex_idx}, vertex_pos: {vertex_pos}")

        # Process each mapping point
        for current_node in mapping_result.node_array:
            # Get current bone index and weight
            current_src_bone_index = current_node.bone_index
            current_node_weight = current_node.weight

            # Skip if weight is very small
            if current_node_weight < 0.001:
                continue

            # Get vector from mapping point to vertex
            current_src_p = current_node.point
            current_src_pv = vertex_pos - current_src_p

            # Normalize direction vector
            current_src_pv_norm = np.linalg.norm(current_src_pv)
            if current_src_pv_norm < 1e-10:
                continue
            current_src_normal_pv = current_src_pv / current_src_pv_norm

            # Get source bone information
            src_start_joint_index = src_bone_group[current_src_bone_index].start_joint_index
            src_end_joint_index = src_bone_group[current_src_bone_index].end_joint_index

            current_src_bone_start_point = src_joint_group[src_start_joint_index].position
            current_src_bone_end_point = src_joint_group[src_end_joint_index].position
            current_src_bone_v = current_src_bone_end_point - current_src_bone_start_point

            # Calculate distance ratio along bone
            current_src_bone_v_norm = np.linalg.norm(current_src_bone_v)
            if current_src_bone_v_norm < 1e-10:
                continue
            src_distance = np.linalg.norm(current_src_p - current_src_bone_start_point) / current_src_bone_v_norm

            # Get target bone information
            tar_start_joint_index = src2tar_map[src_start_joint_index]
            tar_end_joint_index   = src2tar_map[src_end_joint_index]
            if tar_start_joint_index < 0 or tar_end_joint_index < 0:
                continue

            current_tar_bone_start_point = tar_joint_group[tar_start_joint_index].position
            current_tar_bone_end_point = tar_joint_group[tar_end_joint_index].position
            current_tar_bone_v = current_tar_bone_end_point - current_tar_bone_start_point

            # Calculate corresponding point on target bone
            p = current_tar_bone_start_point + current_tar_bone_v * src_distance
            p = np.array(p).squeeze()

            # Get direction vector
            d = np.array(current_src_normal_pv).squeeze()

            # Generate sample directions
            sample_directions = geometry.rand_cone_vector(d, sample_degree, sample_number, seed)
            if flag:
                print(f"sample_directions: {sample_directions}")

            # Cast rays in batches
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
                ray_data["target_distance"][:current_batch_size] = current_src_pv_norm

                # Cast rays
                res = engine.cast_rays(ray_origins[:current_batch_size], ray_directions[:current_batch_size])

                # Process hits
                hit_mask = res["geomID"] >= 0
                if np.any(hit_mask):
                    # Get hit data
                    hit_indices = np.where(hit_mask)[0]
                    prim_ids = res["primID"][hit_mask]
                    ts = res["tfar"][hit_mask]
                    us = res["u"][hit_mask]
                    vs = res["v"][hit_mask]

                    if flag:
                        print(f"hit_indices: {hit_indices}, prim_ids: {prim_ids}, ts: {ts}, us: {us}, vs: {vs}")

                    # Process each hit
                    for i, hit_idx in enumerate(hit_indices):
                        prim_id = prim_ids[i]
                        t = ts[i]
                        u = us[i]
                        v = vs[i]
                        w = 1.0 - u - v

                        # Get triangle vertices
                        triangle_idx = prim_id
                        v0_idx = tar_triangle_indices_np[triangle_idx * 3 + 0]
                        v1_idx = tar_triangle_indices_np[triangle_idx * 3 + 1]
                        v2_idx = tar_triangle_indices_np[triangle_idx * 3 + 2]

                        v0 = tar_triangles_np[v0_idx]
                        v1 = tar_triangles_np[v1_idx]
                        v2 = tar_triangles_np[v2_idx]

                        # Calculate intersection point (barycentric coordinates)
                        intersection_point = w * v0 + u * v1 + v * v2

                        # Get ray data
                        idx = hit_idx
                        vertex_idx = ray_data["vertex_idx"][idx]
                        from_point = ray_data["from_point"][idx]
                        node_weight = ray_data["node_weight"][idx]
                        target_distance = ray_data["target_distance"][idx]

                        # Create result node
                        result_node = RaycastResult(
                            from_point=from_point,
                            point=intersection_point,
                            triangle_index=int(prim_id),
                            vertex_indices=(v0_idx, v1_idx, v2_idx),
                            weight=float(node_weight),
                            relate_distance=float(target_distance / t),
                        )

                        raycast_result_array[vertex_idx].append(result_node)

    # Clean up
    engine.cleanup()

    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)

    return raycast_result_array


@timeit
def perform_raycast_with_options(
    src_mesh: "MeshObject",
    tar_mesh: "MeshObject",
    src_mapping_points: list["MappingResult"],
    tar_triangles: NDArray[np.float64],
    tar_triangle_indices: NDArray[np.int32],
    src_joint_group: list["JointNode"],
    tar_joint_group: list["JointNode"],
    src_bone_group: list["BoneNode"],
    tar_bone_group: list["BoneNode"],
    options: RegistrationOptions,
) -> list[list[RaycastResult]]:
    """Perform raycasting with registration options.

    Convenience function that uses RegistrationOptions to configure raycasting.

    Args:
        src_mesh: Target mesh object
        tar_mesh: Source mesh object
        src_mapping_points: Sorce mapping points
        tar_triangles: Target mesh vertices
        tar_triangle_indices: Target mesh triangle indices
        src_joint_group: Source joint group
        tar_joint_group: Target joint group
        src_bone_group: Source bone group
        tar_bone_group: Target bone group
        options: Registration options

    Returns:
        List of raycast results for each target vertex
    """
    return perform_raycast(
        src_mesh=src_mesh,
        tar_mesh=tar_mesh,
        src_mapping_points=src_mapping_points,
        tar_triangles=tar_triangles,
        tar_triangle_indices=tar_triangle_indices,
        sample_number=options.sample_number,
        sample_degree=options.sample_degree,
        src_joint_group=src_joint_group,
        tar_joint_group=tar_joint_group,
        src_bone_group=src_bone_group,
        tar_bone_group=tar_bone_group,
        batch_size=options.batch_size,
        max_triangles=options.max_triangles,
        sample_vertex_count=options.sample_count,
    )
