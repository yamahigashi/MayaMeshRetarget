"""Geometric calculation functions for mesh registration.

This module provides functions for geometric operations such as ray-triangle intersection,
random vector generation within cones, and barycentric coordinates calculation.
"""

import math
import random
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .core import Vector3


def rand_cone_vector(
    direction: Vector3,
    angle_degree: float,
    num_samples: int,
    seed: Optional[int] = None,
) -> NDArray[np.float32]:
    """Generate random vectors within a cone around a direction vector.

    Args:
        direction: Center direction vector of the cone (doesn't need to be normalized)
        angle_degree: Angle of the cone in degrees
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of random direction vectors (num_samples, 3) as float32
    """
    # Normalize direction vector
    direction = np.array(direction, dtype=np.float32)
    direction_norm = np.linalg.norm(direction)
    direction = np.array([0, 0, 1], dtype=np.float32) if direction_norm < 1e-10 else direction / direction_norm

    # Convert degrees to radians
    cone_angle = angle_degree * math.pi / 180.0

    # Initialize result array as float32
    try:
        result = np.zeros((num_samples, 3), dtype=np.float32)
    except TypeError:
        print(f"type of num_samples: {type(num_samples)}")
        raise

    random.seed(seed)

    for i in range(num_samples):
        # Generate random z coordinate (cone height direction)
        # z ∈ [cos(cone_angle), 1]
        z = (random.random() * (1.0 - math.cos(cone_angle))) + math.cos(cone_angle)

        # Random polar angle
        phi = random.random() * 2.0 * math.pi

        # Generate point on unit sphere
        x = math.sqrt(1.0 - z * z) * math.cos(phi)
        y = math.sqrt(1.0 - z * z) * math.sin(phi)

        # Calculate rotation matrix from standard basis (0,0,1) to direction
        u = np.cross(np.array([0, 0, 1], dtype=np.float32), direction)
        u_norm = np.linalg.norm(u)

        if u_norm < 1e-10:
            # Direction is close to [0,0,1] or [0,0,-1]
            if direction[2] > 0:
                result[i] = np.array([x, y, z], dtype=np.float32)
            else:
                result[i] = np.array([x, y, -z], dtype=np.float32)
            continue

        u = u / u_norm
        rot = math.acos(np.clip(direction[2], -1.0, 1.0))  # Angle with z axis

        # Rotation using Rodrigues' rotation formula
        cos_rot = math.cos(rot)
        sin_rot = math.sin(rot)

        # Calculate rotation matrix
        rot_matrix = np.zeros((3, 3), dtype=np.float32)
        # Diagonal components
        rot_matrix[0, 0] = cos_rot + (1.0 - cos_rot) * u[0] * u[0]
        rot_matrix[1, 1] = cos_rot + (1.0 - cos_rot) * u[1] * u[1]
        rot_matrix[2, 2] = cos_rot + (1.0 - cos_rot) * u[2] * u[2]

        # Off-diagonal components
        rot_matrix[0, 1] = (1.0 - cos_rot) * u[0] * u[1] - sin_rot * u[2]
        rot_matrix[0, 2] = (1.0 - cos_rot) * u[0] * u[2] + sin_rot * u[1]
        rot_matrix[1, 0] = (1.0 - cos_rot) * u[1] * u[0] + sin_rot * u[2]
        rot_matrix[1, 2] = (1.0 - cos_rot) * u[1] * u[2] - sin_rot * u[0]
        rot_matrix[2, 0] = (1.0 - cos_rot) * u[2] * u[0] - sin_rot * u[1]
        rot_matrix[2, 1] = (1.0 - cos_rot) * u[2] * u[1] + sin_rot * u[0]

        # Rotate direction vector
        rand_dir = rot_matrix @ np.array([x, y, z], dtype=np.float32)
        result[i] = rand_dir

    return result


def ray_triangle_intersection(
    orig: Vector3,
    dir_vec: Vector3,
    v0: Vector3,
    v1: Vector3,
    v2: Vector3,
    scale: float = 1.0,
) -> tuple[bool, Optional[Vector3], float]:
    """Ray-triangle intersection test.

    Python implementation of the ray_triangle_intersection function from C++ version

    Args:
        orig: Ray origin point
        dir_vec: Ray direction vector
        v0: Triangle vertex 0
        v1: Triangle vertex 1
        v2: Triangle vertex 2
        scale: Triangle scale factor (scaling from center)

    Returns:
        Tuple containing:
        - bool: True if intersection found
        - Vector3 or None: Intersection point, None if no intersection
        - float: Ray parameter t, -1 if no intersection
    """
    # Type and shape checking
    EPSILON = 1e-12

    # Scaling
    center_tri = (v0 + v1 + v2) / 3.0
    v0_scaled = (v0 - center_tri) * scale + center_tri
    v1_scaled = (v1 - center_tri) * scale + center_tri
    v2_scaled = (v2 - center_tri) * scale + center_tri

    # Triangle edge vectors
    e1 = v1_scaled - v0_scaled
    e2 = v2_scaled - v0_scaled

    # Normal vector
    n = np.cross(e1, e2)
    ndd = np.dot(dir_vec, n)

    # Check if ray is directed toward the triangle
    if ndd < 0:
        return False, None, -1

    # Möller–Trumbore algorithm
    h = np.cross(dir_vec, e2)
    a = np.dot(e1, h)

    if -EPSILON < a < EPSILON:
        return False, None, -1  # Ray is parallel to triangle

    f = 1.0 / a
    s = orig - v0_scaled
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False, None, -1

    q = np.cross(s, e1)
    v = f * np.dot(dir_vec, q)

    if v < 0.0 or u + v > 1.0:
        return False, None, -1

    t = f * np.dot(e2, q)

    if t > EPSILON:
        intersection_point = orig + dir_vec * t
        return True, intersection_point, t

    return False, None, -1


def ray_triangle_intersection_with_uv(
    orig: Vector3,
    dir_vec: Vector3,
    v0: Vector3,
    v1: Vector3,
    v2: Vector3,
    scale: float = 1.0,
) -> tuple[bool, Optional[Vector3], float, float, float]:
    """Ray-triangle intersection test with barycentric coordinates.

    Extends the ray_triangle_intersection function to also return
    the barycentric coordinates (u, v).

    Args:
        orig: Ray origin point
        dir_vec: Ray direction vector
        v0: Triangle vertex 0
        v1: Triangle vertex 1
        v2: Triangle vertex 2
        scale: Triangle scale factor (scaling from center)

    Returns:
        Tuple containing:
        - bool: True if intersection found
        - Vector3 or None: Intersection point, None if no intersection
        - float: Ray parameter t, -1 if no intersection
        - float: Barycentric coordinate u
        - float: Barycentric coordinate v
    """
    # Type and shape checking
    EPSILON = 1e-12

    # Scaling
    center_tri = (v0 + v1 + v2) / 3.0
    v0_scaled = (v0 - center_tri) * scale + center_tri
    v1_scaled = (v1 - center_tri) * scale + center_tri
    v2_scaled = (v2 - center_tri) * scale + center_tri

    # Triangle edge vectors
    e1 = v1_scaled - v0_scaled
    e2 = v2_scaled - v0_scaled

    # Normal vector
    n = np.cross(e1, e2)
    ndd = np.dot(dir_vec, n)

    # Check if ray is directed toward the triangle
    if ndd < 0:
        return False, None, -1, 0.0, 0.0

    # Möller–Trumbore algorithm
    h = np.cross(dir_vec, e2)
    a = np.dot(e1, h)

    if -EPSILON < a < EPSILON:
        return False, None, -1, 0.0, 0.0  # Ray is parallel to triangle

    f = 1.0 / a
    s = orig - v0_scaled
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False, None, -1, 0.0, 0.0

    q = np.cross(s, e1)
    v = f * np.dot(dir_vec, q)

    if v < 0.0 or u + v > 1.0:
        return False, None, -1, 0.0, 0.0

    t = f * np.dot(e2, q)

    if t > EPSILON:
        intersection_point = orig + dir_vec * t
        return True, intersection_point, t, u, v

    return False, None, -1, 0.0, 0.0


def triangle_interpolation(
    v1: Vector3,
    v2: Vector3,
    v3: Vector3,
    p: Vector3,
) -> tuple[float, float, float]:
    """Calculate barycentric coordinates of a point in a triangle.

    Python implementation of the triangle_interpolation function from C++ version

    Args:
        v1: Triangle vertex 1
        v2: Triangle vertex 2
        v3: Triangle vertex 3
        p: Point to calculate barycentric coordinates for

    Returns:
        Tuple containing the barycentric coordinates (w1, w2, w3)
    """
    # Build local coordinate system for the triangle
    x = v3 - v1
    y = np.cross(v2 - v1, x)
    z = np.cross(y, x)

    # Build transformation matrix
    triangle_local_matrix = np.zeros((4, 4))
    triangle_local_matrix[0, 0] = x[0]
    triangle_local_matrix[0, 1] = y[0]
    triangle_local_matrix[0, 2] = z[0]

    triangle_local_matrix[1, 0] = x[1]
    triangle_local_matrix[1, 1] = y[1]
    triangle_local_matrix[1, 2] = z[1]

    triangle_local_matrix[2, 0] = x[2]
    triangle_local_matrix[2, 1] = y[2]
    triangle_local_matrix[2, 2] = z[2]

    triangle_local_matrix[3, 0] = v1[0]
    triangle_local_matrix[3, 1] = v1[1]
    triangle_local_matrix[3, 2] = v1[2]
    triangle_local_matrix[3, 3] = 1.0

    # Transform points to local coordinate system
    triangle_local_matrix_inv = np.linalg.inv(triangle_local_matrix)

    tv1 = np.append(v1, 1.0) @ triangle_local_matrix_inv
    tv2 = np.append(v2, 1.0) @ triangle_local_matrix_inv
    tv3 = np.append(v3, 1.0) @ triangle_local_matrix_inv
    tp = np.append(p, 1.0) @ triangle_local_matrix_inv

    # Calculate barycentric coordinates
    deno = (tv2[2] - tv3[2]) * (tv1[0] - tv3[0]) + (tv3[0] - tv2[0]) * (tv1[2] - tv3[2])
    w1 = ((tv2[2] - tv3[2]) * (tp[0] - tv3[0]) + (tv3[0] - tv2[0]) * (tp[2] - tv3[2])) / deno
    w2 = ((tv3[2] - tv1[2]) * (tp[0] - tv3[0]) + (tv1[0] - tv3[0]) * (tp[2] - tv3[2])) / deno
    w3 = 1.0 - w1 - w2

    return w1, w2, w3


def calculate_triangle_normal(v0: Vector3, v1: Vector3, v2: Vector3) -> Vector3:
    """Calculate the normal vector of a triangle.

    Args:
        v0: First vertex of the triangle
        v1: Second vertex of the triangle
        v2: Third vertex of the triangle

    Returns:
        Normalized normal vector of the triangle
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)

    # Normalize
    normal_length = np.linalg.norm(normal)
    if normal_length > 1e-10:
        normal = normal / normal_length

    return normal


def create_plane_from_triangle(v0: Vector3, v1: Vector3, v2: Vector3) -> tuple[Vector3, float]:
    """Create a plane equation from a triangle.

    Args:
        v0: First vertex of the triangle
        v1: Second vertex of the triangle
        v2: Third vertex of the triangle

    Returns:
        Tuple containing:
        - Vector3: Normalized plane normal
        - float: Plane distance constant
    """
    normal = calculate_triangle_normal(v0, v1, v2)
    d = -np.dot(normal, v0)

    return normal, d


def point_to_triangle_distance(p: Vector3, v0: Vector3, v1: Vector3, v2: Vector3) -> float:
    """Calculate the minimum distance from a point to a triangle.

    Args:
        p: Point
        v0: First vertex of the triangle
        v1: Second vertex of the triangle
        v2: Third vertex of the triangle

    Returns:
        Minimum distance from the point to the triangle
    """
    # Calculate triangle normal and plane
    normal, d = create_plane_from_triangle(v0, v1, v2)

    # Calculate distance to the plane
    plane_distance = abs(np.dot(normal, p) + d)

    # Project the point onto the plane
    projected_point = p - plane_distance * normal

    # Check if the projected point is inside the triangle
    # using barycentric coordinates
    w1, w2, w3 = triangle_interpolation(v0, v1, v2, projected_point)

    if 0 <= w1 <= 1 and 0 <= w2 <= 1 and 0 <= w3 <= 1:
        # Point is inside the triangle, return plane distance
        return plane_distance

    # Point is outside the triangle, calculate minimum distance to edges
    edge_distances = [
        point_to_line_segment_distance(p, v0, v1),
        point_to_line_segment_distance(p, v1, v2),
        point_to_line_segment_distance(p, v2, v0),
    ]

    return min(edge_distances)


def point_to_line_segment_distance(p: Vector3, a: Vector3, b: Vector3) -> float:
    """Calculate the minimum distance from a point to a line segment.

    Args:
        p: Point
        a: Start of line segment
        b: End of line segment

    Returns:
        Minimum distance from the point to the line segment
    """
    # Vector from a to b
    ab = b - a
    ab_length_squared = np.dot(ab, ab)

    if ab_length_squared < 1e-10:
        # The line segment is degenerate
        return np.linalg.norm(p - a)

    # Calculate projection of p onto line segment
    ap = p - a
    t = np.dot(ap, ab) / ab_length_squared

    if t < 0:
        # Point is beyond point a
        return np.linalg.norm(p - a)
    elif t > 1:
        # Point is beyond point b
        return np.linalg.norm(p - b)
    else:
        # Point projects onto line segment
        projection = a + t * ab
        return np.linalg.norm(p - projection)
