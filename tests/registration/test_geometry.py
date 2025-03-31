import math

import numpy as np

from ymt_mesh_retarget.registration.geometry import (
    calculate_triangle_normal,
    point_to_line_segment_distance,
    point_to_triangle_distance,
    rand_cone_vector,
    ray_triangle_intersection,
    ray_triangle_intersection_with_uv,
    triangle_interpolation,
)


def test_rand_cone_vector():
    """Test random vector generation within a cone."""
    # Base direction vector
    direction = np.array([0, 0, 1], dtype=np.float32)

    # Cone angle and sample count
    angle_degree = 45.0
    num_samples = 100

    # Fixed seed for reproducibility
    seed = 42

    # Generate vectors
    vectors = rand_cone_vector(direction, angle_degree, num_samples, seed)

    # Check shape and type of returned vectors
    assert vectors.shape == (num_samples, 3)
    assert vectors.dtype == np.float32

    # Verify all vectors are unit vectors
    for v in vectors:
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)

    # Verify all vectors are within the specified cone
    for v in vectors:
        # Angle between base direction and vector (in radians)
        angle_rad = np.arccos(np.clip(np.dot(direction, v), -1.0, 1.0))
        # Convert to degrees
        angle = np.degrees(angle_rad)
        # Must be within specified angle
        assert angle <= angle_degree + 1e-6  # Add small tolerance for floating point errors

    # Test with a different direction
    direction2 = np.array([1, 1, 1], dtype=np.float32)
    vectors2 = rand_cone_vector(direction2, angle_degree, num_samples, seed)

    # Normalize direction vector
    norm_dir2 = direction2 / np.linalg.norm(direction2)

    # Verify all vectors are within the specified cone
    for v in vectors2:
        angle_rad = np.arccos(np.clip(np.dot(norm_dir2, v), -1.0, 1.0))
        angle = np.degrees(angle_rad)
        assert angle <= angle_degree + 1e-6


def test_ray_triangle_intersection():
    """Test ray-triangle intersection function."""
    # Triangle vertices
    v0 = np.array([0, 0, 0], dtype=np.float64)
    v1 = np.array([1, 0, 0], dtype=np.float64)
    v2 = np.array([0, 1, 0], dtype=np.float64)

    # Test case: Ray intersects triangle
    orig = np.array([0.2, 0.2, 1], dtype=np.float64)
    dir_vec = np.array([0, 0, -1], dtype=np.float64)

    hit, point, t = ray_triangle_intersection(orig, dir_vec, v0, v1, v2)

    assert hit is True
    assert isinstance(point, np.ndarray)
    assert point.shape == (3,)
    # Verify intersection point
    assert np.allclose(point, [0.2, 0.2, 0], atol=1e-10)
    assert t > 0

    # Test case: Ray misses triangle
    orig2 = np.array([2, 2, 1], dtype=np.float64)
    dir_vec2 = np.array([0, 0, -1], dtype=np.float64)

    hit2, point2, t2 = ray_triangle_intersection(orig2, dir_vec2, v0, v1, v2)

    assert hit2 is False
    assert point2 is None
    assert t2 == -1

    # Test case: With triangle scaling
    scale = 2.0
    hit3, point3, t3 = ray_triangle_intersection(orig, dir_vec, v0, v1, v2, scale)

    assert hit3 is True  # Larger triangle should still intersect
    # Intersection point should be the same
    expected_point = orig + dir_vec * t3
    assert np.allclose(point3, expected_point, atol=1e-10)


def test_ray_triangle_intersection_with_uv():
    """Test ray-triangle intersection with UV coordinates."""
    # Triangle vertices
    v0 = np.array([0, 0, 0], dtype=np.float64)
    v1 = np.array([1, 0, 0], dtype=np.float64)
    v2 = np.array([0, 1, 0], dtype=np.float64)

    # Test case: Ray intersects triangle
    orig = np.array([0.25, 0.25, 1], dtype=np.float64)
    dir_vec = np.array([0, 0, -1], dtype=np.float64)

    hit, point, t, u, v = ray_triangle_intersection_with_uv(orig, dir_vec, v0, v1, v2)

    assert hit is True
    assert isinstance(point, np.ndarray)
    assert point.shape == (3,)
    # Verify intersection point
    assert np.allclose(point, [0.25, 0.25, 0], atol=1e-10)
    assert t > 0

    # Verify UV coordinates (barycentric)
    # u, v are weights for v1, v2 respectively
    # Point (0.25, 0.25, 0) should have barycentric coords (0.5, 0.25, 0.25)
    assert np.isclose(u, 0.25, atol=1e-5)
    assert np.isclose(v, 0.25, atol=1e-5)
    assert np.isclose(1 - u - v, 0.5, atol=1e-5)  # Weight for first vertex


def test_triangle_interpolation():
    """Test triangle interpolation (barycentric coordinate calculation)."""
    # Triangle vertices
    v0 = np.array([0, 0, 0], dtype=np.float64)
    v1 = np.array([1, 0, 0], dtype=np.float64)
    v2 = np.array([0, 1, 0], dtype=np.float64)

    # Test cases
    test_points = [
        # point, expected barycentric coords (w1, w2, w3)
        (np.array([0, 0, 0]), (1.0, 0.0, 0.0)),  # Vertex 0
        (np.array([1, 0, 0]), (0.0, 1.0, 0.0)),  # Vertex 1
        (np.array([0, 1, 0]), (0.0, 0.0, 1.0)),  # Vertex 2
        (np.array([1/3, 1/3, 0]), (1/3, 1/3, 1/3)),  # Centroid
        (np.array([0.5, 0, 0]), (0.5, 0.5, 0.0)),  # Edge midpoint (v0-v1)
        (np.array([0, 0.5, 0]), (0.5, 0.0, 0.5)),  # Edge midpoint (v0-v2)
    ]

    for point, expected in test_points:
        w1, w2, w3 = triangle_interpolation(v0, v1, v2, point)
        assert np.allclose([w1, w2, w3], expected, atol=1e-6)


def test_calculate_triangle_normal():
    """Test triangle normal calculation."""
    # Triangle vertices
    v0 = np.array([0, 0, 0], dtype=np.float64)
    v1 = np.array([1, 0, 0], dtype=np.float64)
    v2 = np.array([0, 1, 0], dtype=np.float64)

    normal = calculate_triangle_normal(v0, v1, v2)

    # Normal should be (0, 0, 1)
    expected_normal = np.array([0, 0, 1], dtype=np.float64)
    assert np.allclose(normal, expected_normal, atol=1e-10)

    # Verify normalization
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=1e-10)

    # Test with a different triangle
    v0 = np.array([0, 0, 0], dtype=np.float64)
    v1 = np.array([1, 0, 0], dtype=np.float64)
    v2 = np.array([0, 0, 1], dtype=np.float64)

    normal = calculate_triangle_normal(v0, v1, v2)

    # The normal could be (0, 1, 0) or (0, -1, 0) depending on cross product implementation
    # Just check that it's in the y-axis direction and normalized
    assert np.isclose(abs(normal[1]), 1.0, atol=1e-10)
    assert np.isclose(normal[0], 0.0, atol=1e-10)
    assert np.isclose(normal[2], 0.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=1e-10)


def test_point_to_line_segment_distance():
    """Test distance calculation from point to line segment."""
    # Line segment endpoints
    a = np.array([0, 0, 0], dtype=np.float64)
    b = np.array([1, 0, 0], dtype=np.float64)

    # Test cases
    test_points = [
        # point, expected distance
        (np.array([0.5, 0, 0]), 0.0),  # Point on segment
        (np.array([0.5, 1, 0]), 1.0),  # Point 1 unit perpendicular from segment
        (np.array([-1, 0, 0]), 1.0),   # Point beyond endpoint a
        (np.array([2, 0, 0]), 1.0),    # Point beyond endpoint b
        (np.array([0.5, 0.5, 0]), 0.5),  # Point at angle from segment
    ]

    for point, expected_distance in test_points:
        distance = point_to_line_segment_distance(point, a, b)
        assert np.isclose(distance, expected_distance, atol=1e-10)


def test_point_to_triangle_distance():
    """Test distance calculation from point to triangle."""
    # Triangle vertices
    v0 = np.array([0, 0, 0], dtype=np.float64)
    v1 = np.array([1, 0, 0], dtype=np.float64)
    v2 = np.array([0, 1, 0], dtype=np.float64)

    # Test cases
    test_points = [
        # point, expected distance
        (np.array([0.25, 0.25, 0]), 0.0),  # Point on triangle plane
        (np.array([0.25, 0.25, 1]), 1.0),  # Point 1 unit above triangle
        (np.array([2, 0, 0]), 1.0),        # Point outside triangle edge
        (np.array([-1, -1, 0]), math.sqrt(2)),  # Point outside triangle vertex
    ]

    for point, expected_distance in test_points:
        distance = point_to_triangle_distance(point, v0, v1, v2)
        assert np.isclose(distance, expected_distance, atol=1e-10)
