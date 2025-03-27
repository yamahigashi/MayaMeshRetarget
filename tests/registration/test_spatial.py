import numpy as np
import pytest

from ymt_mesh_retarget.registration.spatial import BVH, BVHNode, AABB


def test_bvh_node_initialization():
    """Test BVHNode initialization and basic properties."""
    # Create a simple AABB
    min_point = np.array([0, 0, 0], dtype=np.float64)
    max_point = np.array([1, 1, 1], dtype=np.float64)
    aabb = (min_point, max_point)
    
    # Create a node
    node = BVHNode(aabb)
    
    # Check initialization
    assert node.aabb == aabb
    assert node.triangle_indices == []
    assert node.left is None
    assert node.right is None
    assert node.is_leaf() == True


def test_bvh_construction():
    """Test BVH construction with simple triangles."""
    # Create a simple cube mesh (8 vertices, 12 triangles)
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    
    # Define triangles for a cube (simplified - just a few triangles)
    triangle_indices = np.array([
        # Front face
        0, 1, 2, 0, 2, 3,
        # Back face
        4, 5, 6, 4, 6, 7,
    ], dtype=np.int32)
    
    # Create BVH
    bvh = BVH(vertices, triangle_indices, max_triangles_per_leaf=2, max_depth=3)
    
    # Check BVH properties
    assert bvh.triangles is not None
    assert bvh.triangle_indices is not None
    assert bvh.max_triangles_per_leaf == 2
    assert bvh.max_depth == 3
    assert bvh.root is not None


def test_ray_aabb_intersection():
    """Test ray-AABB intersection calculation."""
    # Create a BVH with a simple cube
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    
    triangle_indices = np.array([
        0, 1, 2, 0, 2, 3,  # Front face
        4, 5, 6, 4, 6, 7,  # Back face
    ], dtype=np.int32)
    
    bvh = BVH(vertices, triangle_indices)
    
    # Test cases
    test_cases = [
        # Ray origin, ray direction, expected result
        (np.array([0.5, 0.5, -1], dtype=np.float64), np.array([0, 0, 1], dtype=np.float64), True),  # Hit front face
        (np.array([2, 2, 0], dtype=np.float64), np.array([-1, -1, 0], dtype=np.float64), True),    # Hit corner
        (np.array([2, 0, 0], dtype=np.float64), np.array([0, 1, 0], dtype=np.float64), False),     # Miss
    ]
    
    for origin, direction, expected in test_cases:
        aabb = (np.array([0, 0, 0], dtype=np.float64), np.array([1, 1, 1], dtype=np.float64))
        result = bvh._ray_aabb_intersection(origin, direction, aabb)
        assert result == expected


def test_ray_intersection():
    """Test ray intersection with the BVH."""
    # Create a simple cube mesh
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    
    # Define triangles for a cube (just front and back faces for simplicity)
    triangle_indices = np.array([
        # Front face (z=0)
        0, 1, 2, 0, 2, 3,
        # Back face (z=1)
        4, 5, 6, 4, 6, 7,
    ], dtype=np.int32)
    
    # Create BVH
    bvh = BVH(vertices, triangle_indices)
    
    # Test ray that hits the front face
    origin = np.array([0.5, 0.5, -1], dtype=np.float64)
    direction = np.array([0, 0, 1], dtype=np.float64)
    
    hit = bvh.ray_intersection(origin, direction)
    
    # Should hit front face
    assert hit is not None
    assert hit["prim_id"] == 0 or hit["prim_id"] == 1  # Either triangle of front face
    assert hit["tfar"] > 0  # Positive distance
    
    # Test ray that misses
    origin = np.array([2, 2, 2], dtype=np.float64)
    direction = np.array([1, 0, 0], dtype=np.float64)
    
    hit = bvh.ray_intersection(origin, direction)
    assert hit is None


def test_rays_batch_intersection():
    """Test batch ray intersection with the BVH."""
    # Create a simple cube mesh
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    
    # Define triangles for a cube
    triangle_indices = np.array([
        # Front face
        0, 1, 2, 0, 2, 3,
        # Back face
        4, 5, 6, 4, 6, 7,
    ], dtype=np.int32)
    
    # Create BVH
    bvh = BVH(vertices, triangle_indices)
    
    # Create batch of rays
    origins = np.array([
        [0.25, 0.25, -1],  # Hit front face
        [0.75, 0.75, -1],  # Hit front face
        [2, 2, 2],         # Miss
    ], dtype=np.float32)
    
    directions = np.array([
        [0, 0, 1],  # Towards front face
        [0, 0, 1],  # Towards front face
        [1, 0, 0],  # Away from cube
    ], dtype=np.float32)
    
    # Cast rays
    results = bvh.ray_intersections_batch(origins, directions)
    
    # Check results
    assert "prim_id" in results
    assert "geom_id" in results
    assert "tfar" in results
    assert "u" in results
    assert "v" in results
    
    # First two rays should hit, third should miss
    assert results["geom_id"][0] >= 0
    assert results["geom_id"][1] >= 0
    assert results["geom_id"][2] < 0