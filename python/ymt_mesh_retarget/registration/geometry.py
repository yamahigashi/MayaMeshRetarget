# -*- coding: utf-8 -*-
"""
Geometric calculation functions for mesh registration.

This module provides functions for geometric operations such as ray-triangle intersection,
random vector generation within cones, and barycentric coordinates calculation.
"""

import math
import random

import numpy as np


def rand_cone_vector(direction, angle_degree, num_samples):
    """Generate random vectors within a cone around a direction vector
    
    Python implementation of the rand_cone_vector function from C++ version

    Args:
        direction (np.ndarray): Center direction vector of the cone (doesn't need to be normalized)
        angle_degree (float): Angle of the cone in degrees
        num_samples (int): Number of samples to generate

    Returns:
        np.ndarray: Array of random direction vectors (num_samples, 3)
    """
    # Normalize direction vector
    direction = np.array(direction, dtype=np.float64)
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-10:
        direction = np.array([0, 0, 1])
    else:
        direction = direction / direction_norm

    # Convert degrees to radians
    cone_angle = angle_degree * math.pi / 180.0

    # Initialize result array
    try:
        result = np.zeros((num_samples, 3))
    except TypeError:
        print(f"type of num_samples: {type(num_samples)}")
        raise

    # Set random seed
    random.seed()

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
        u = np.cross(np.array([0, 0, 1]), direction)
        u_norm = np.linalg.norm(u)

        if u_norm < 1e-10:
            # Direction is close to [0,0,1] or [0,0,-1]
            if direction[2] > 0:
                result[i] = np.array([x, y, z])
            else:
                result[i] = np.array([x, y, -z])
            continue

        u = u / u_norm
        rot = math.acos(np.clip(direction[2], -1.0, 1.0))  # Angle with z axis

        # Rotation using Rodrigues' rotation formula
        cos_rot = math.cos(rot)
        sin_rot = math.sin(rot)

        # Calculate rotation matrix
        rot_matrix = np.zeros((3, 3))
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
        rand_dir = rot_matrix @ np.array([x, y, z])
        result[i] = rand_dir

    return result


def ray_triangle_intersection(orig, dir_vec, v0, v1, v2, scale=1.0):
    """Ray-triangle intersection test
    
    Python implementation of the ray_triangle_intersection function from C++ version

    Args:
        orig (np.ndarray): Ray origin point
        dir_vec (np.ndarray): Ray direction vector
        v0 (np.ndarray): Triangle vertex 0
        v1 (np.ndarray): Triangle vertex 1
        v2 (np.ndarray): Triangle vertex 2
        scale (float): Triangle scale factor (scaling from center)

    Returns:
        tuple: (intersection result, intersection point, ray parameter t)
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


def triangle_interpolation(v1, v2, v3, p):
    """Calculate barycentric coordinates of a point in a triangle
    
    Python implementation of the triangle_interpolation function from C++ version

    Args:
        v1 (np.ndarray): Triangle vertex 1
        v2 (np.ndarray): Triangle vertex 2
        v3 (np.ndarray): Triangle vertex 3
        p (np.ndarray): Point to calculate barycentric coordinates for

    Returns:
        tuple: (w1, w2, w3) Barycentric coordinates
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