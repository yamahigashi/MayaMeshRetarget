"""Spatial acceleration structures for mesh registration.

This module provides spatial data structures for accelerating ray-mesh intersection
and other geometric queries.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


# Type aliases
Vector3 = NDArray[np.float64]  # 3D vector (x, y, z)
AABB = tuple[Vector3, Vector3]  # Axis-aligned bounding box (min, max)


class BVHNode:
    """Node in a Bounding Volume Hierarchy tree.

    Attributes:
        aabb: Axis-aligned bounding box for this node (min_point, max_point)
        triangle_indices: Indices of triangles in this node (leaf nodes only)
        left: Left child node
        right: Right child node
    """

    def __init__(self, aabb: AABB) -> None:
        """Initialize a BVH node.

        Args:
            aabb: Axis-aligned bounding box for this node (min_point, max_point)
        """
        self.aabb = aabb
        self.triangle_indices: list[int] = []
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node.

        Returns:
            True if this is a leaf node, False otherwise
        """
        return self.left is None and self.right is None


class BVH:
    """Bounding Volume Hierarchy for accelerating ray-triangle intersections.

    This is a simplified BVH implementation focused on efficient ray casting.
    """

    def __init__(
        self,
        triangles: NDArray[np.float32],
        triangle_indices: NDArray[np.int32],
        max_triangles_per_leaf: int = 4,
        max_depth: int = 20,
    ) -> None:
        """Initialize and build a BVH.

        Args:
            triangles: Vertex positions
            triangle_indices: Triangle indices
            max_triangles_per_leaf: Maximum triangles per leaf node
            max_depth: Maximum tree depth
        """
        self.triangles = triangles
        self.triangle_indices = triangle_indices
        self.max_triangles_per_leaf = max_triangles_per_leaf
        self.max_depth = max_depth

        # Triangle to vertex mapping
        self.triangle_vertices = np.zeros((len(triangle_indices) // 3, 3, 3), dtype=np.float32)
        self._prepare_triangle_vertices()

        # Build the BVH tree
        self.root = self._build_bvh()

    def _prepare_triangle_vertices(self) -> None:
        """Prepare triangle vertices for efficient access."""
        num_triangles = len(self.triangle_indices) // 3

        for i in range(num_triangles):
            v0_idx = self.triangle_indices[i * 3 + 0]
            v1_idx = self.triangle_indices[i * 3 + 1]
            v2_idx = self.triangle_indices[i * 3 + 2]

            self.triangle_vertices[i, 0] = self.triangles[v0_idx]
            self.triangle_vertices[i, 1] = self.triangles[v1_idx]
            self.triangle_vertices[i, 2] = self.triangles[v2_idx]

    def _build_bvh(self) -> BVHNode:
        """Build the BVH tree.

        Returns:
            Root node of the BVH tree
        """
        # Create root node with all triangles
        num_triangles = len(self.triangle_vertices)
        all_triangle_indices = list(range(num_triangles))

        # Calculate initial AABB for all triangles
        min_point = np.min(self.triangles, axis=0)
        max_point = np.max(self.triangles, axis=0)
        root_aabb = (min_point, max_point)

        # Create root node
        root = BVHNode(root_aabb)

        # Build tree recursively
        self._subdivide(root, all_triangle_indices, 0)

        return root

    def _subdivide(self, node: BVHNode, triangle_indices: list[int], depth: int) -> None:
        """Recursively subdivide a BVH node.

        Args:
            node: Node to subdivide
            triangle_indices: Indices of triangles in this node
            depth: Current tree depth
        """
        # Stop if we've reached the maximum depth or have few enough triangles
        if depth >= self.max_depth or len(triangle_indices) <= self.max_triangles_per_leaf:
            node.triangle_indices = triangle_indices
            return

        # Calculate centroids for all triangles
        centroids = np.zeros((len(triangle_indices), 3), dtype=np.float32)
        for i, tri_idx in enumerate(triangle_indices):
            v0 = self.triangle_vertices[tri_idx, 0]
            v1 = self.triangle_vertices[tri_idx, 1]
            v2 = self.triangle_vertices[tri_idx, 2]
            centroids[i] = (v0 + v1 + v2) / 3.0

        # Find the axis with the greatest extent
        min_point, max_point = node.aabb
        extents = max_point - min_point
        axis = np.argmax(extents)

        # Sort triangles by centroid along the chosen axis
        sorted_indices = sorted(
            range(len(triangle_indices)),
            key=lambda i: centroids[i, axis],
        )

        # Split at median
        median = len(sorted_indices) // 2
        left_indices = [triangle_indices[sorted_indices[i]] for i in range(median)]
        right_indices = [triangle_indices[sorted_indices[i]] for i in range(median, len(sorted_indices))]

        # If one side is empty, just create a leaf node
        if not left_indices or not right_indices:
            node.triangle_indices = triangle_indices
            return

        # Calculate AABBs for children
        left_aabb = self._calculate_aabb(left_indices)
        right_aabb = self._calculate_aabb(right_indices)

        # Create child nodes
        node.left = BVHNode(left_aabb)
        node.right = BVHNode(right_aabb)

        # Recursively subdivide children
        self._subdivide(node.left, left_indices, depth + 1)
        self._subdivide(node.right, right_indices, depth + 1)

    def _calculate_aabb(self, triangle_indices: list[int]) -> AABB:
        """Calculate the AABB for a set of triangles.

        Args:
            triangle_indices: Indices of triangles

        Returns:
            AABB as (min_point, max_point)
        """
        # Initialize min/max with first vertex of first triangle
        v0 = self.triangle_vertices[triangle_indices[0], 0]
        min_point = v0.copy()
        max_point = v0.copy()

        # Update min/max with all vertices of all triangles
        for tri_idx in triangle_indices:
            for i in range(3):  # For each vertex in triangle
                vertex = self.triangle_vertices[tri_idx, i]
                min_point = np.minimum(min_point, vertex)
                max_point = np.maximum(max_point, vertex)

        return min_point, max_point

    def ray_intersection(
        self,
        origin: Vector3,
        direction: Vector3,
    ) -> Optional[dict[str, Any]]:
        """Cast a ray through the BVH.

        Args:
            origin: Ray origin
            direction: Ray direction (normalized)

        Returns:
            Dictionary with hit information or None if no hit
        """
        # Normalize direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:
            return None

        direction = direction / direction_norm

        # Initialize closest hit
        closest_t = float("inf")
        closest_prim_id = -1
        closest_u = 0.0
        closest_v = 0.0

        # Traverse BVH
        from .geometry import ray_triangle_intersection_with_uv

        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()

            # Skip if ray doesn't intersect node AABB
            if not self._ray_aabb_intersection(origin, direction, node.aabb):
                continue

            if node.is_leaf():
                # Test all triangles in leaf node
                for tri_idx in node.triangle_indices:
                    v0 = self.triangle_vertices[tri_idx, 0]
                    v1 = self.triangle_vertices[tri_idx, 1]
                    v2 = self.triangle_vertices[tri_idx, 2]

                    hit, _, t, u, v = ray_triangle_intersection_with_uv(
                        origin,
                        direction,
                        v0,
                        v1,
                        v2,
                    )

                    if hit and t < closest_t:
                        closest_t = t
                        closest_prim_id = tri_idx
                        closest_u = u
                        closest_v = v
            else:
                # Add children to visit queue (prioritize closer nodes)
                if node.left:
                    nodes_to_visit.append(node.left)
                if node.right:
                    nodes_to_visit.append(node.right)

        if closest_prim_id >= 0:
            return {
                "prim_id": closest_prim_id,
                "tfar": closest_t,
                "u": closest_u,
                "v": closest_v,
            }

        return None

    def ray_intersections_batch(
        self,
        origins: NDArray[np.float32],
        directions: NDArray[np.float32],
    ) -> dict[str, NDArray]:
        """Cast multiple rays through the BVH.

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
            hit = self.ray_intersection(origins[ray_idx], directions[ray_idx])

            if hit:
                prim_id[ray_idx] = hit["prim_id"]
                geom_id[ray_idx] = 0  # We only have one geometry
                tfar[ray_idx] = hit["tfar"]
                u[ray_idx] = hit["u"]
                v[ray_idx] = hit["v"]

        return {
            "prim_id": prim_id,
            "geom_id": geom_id,
            "tfar": tfar,
            "u": u,
            "v": v,
        }

    def _ray_aabb_intersection(self, origin: Vector3, direction: Vector3, aabb: AABB) -> bool:
        """Check if a ray intersects an AABB.

        Uses the slab method for ray-box intersection.

        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            aabb: Axis-aligned bounding box as (min_point, max_point)

        Returns:
            True if ray intersects AABB, False otherwise
        """
        min_point, max_point = aabb

        # Calculate intersection with each slab
        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            # Handle case where direction component is close to zero
            if abs(direction[i]) < 1e-10:
                if origin[i] < min_point[i] or origin[i] > max_point[i]:
                    return False
            else:
                t1 = (min_point[i] - origin[i]) / direction[i]
                t2 = (max_point[i] - origin[i]) / direction[i]

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    return False

        # t_max >= t_min is implied from the above check
        return t_max >= 0  # Box is in front of the ray origin
