import numpy as np
import pytest
from unittest.mock import patch

from ymt_mesh_retarget.registration.weights import get_weight_distance
from ymt_mesh_retarget.registration.core import TriangleWeightIndex


@pytest.mark.skip(reason="Errors with mel.eval for progress bar")
def test_get_weight_distance():
    """Test the weight distance calculation function."""
    # Create test data
    # Simple case: 2 triangles, 2 vertices, 2 influences (joints)
    
    # Source triangles (2 triangles, 6 vertex indices)
    src_triangles = np.array([0, 1, 2, 1, 2, 3], dtype=np.int32)
    src_num_tri = 2
    
    # Source mesh vertex weights ([vertex][influence] format)
    # 4 vertices x 2 influences
    src_weight = [
        [1.0, 0.0],  # Vertex 0: 100% from influence 0
        [0.5, 0.5],  # Vertex 1: 50-50 split
        [0.3, 0.7],  # Vertex 2: Mostly influence 1
        [0.0, 1.0],  # Vertex 3: 100% from influence 1
    ]
    
    # Target mesh vertex weights
    # 3 vertices x 2 influences
    tar_weight = [
        [1.0, 0.0],  # Vertex 0: 100% from influence 0
        [0.5, 0.5],  # Vertex 1: 50-50 split
        [0.0, 1.0],  # Vertex 2: 100% from influence 1
    ]
    
    # Target-to-source joint mapping
    # Simple case: direct mapping by index
    tar_joints_retarget = [0, 1]  # Non-negative values are source joint indices
    
    # Call the function with patched Maya environment
    with patch('maya.mel.eval', return_value=None):
        with patch('maya.cmds.about', return_value=True):  # Treat as batch mode
            result = get_weight_distance(
                src_triangles=src_triangles,
                src_num_tri=src_num_tri,
                src_weight=src_weight,
                tar_weight=tar_weight,
                tar_joints_retarget=tar_joints_retarget,
                top_n=2  # Get top 2 triangles for each target vertex
            )
    
    # Basic validation
    assert len(result) == 3  # Number of target vertices
    assert all(len(tri_indices) <= 2 for tri_indices in result)  # Max 2 results per vertex (top_n=2)
    assert all(isinstance(twi, TriangleWeightIndex) for tri_indices in result for twi in tri_indices)
    
    # Detailed validation
    # Vertex 0 should be closest to triangle 0
    # (Triangle 0 avg weights = [0.6, 0.4], Vertex 0 = [1.0, 0.0])
    assert result[0][0].triangle_index == 0
    
    # Vertex 1 should be close to both triangles (small difference)
    assert set([twi.triangle_index for twi in result[1]]) == {0, 1}
    
    # Vertex 2 should be closest to triangle 1
    # (Triangle 1 avg weights = [0.267, 0.733], Vertex 2 = [0.0, 1.0])
    assert result[2][0].triangle_index == 1
    
    # Verify correct distance calculations
    # Triangle 0 avg weights: (1.0+0.5+0.3)/3, (0.0+0.5+0.7)/3 = [0.6, 0.4]
    # Triangle 1 avg weights: (0.5+0.3+0.0)/3, (0.5+0.7+1.0)/3 = [0.267, 0.733]
    
    # Distance between Vertex 0 [1.0, 0.0] and Triangle 0 [0.6, 0.4]
    expected_dist_0_0 = np.sqrt((1.0-0.6)**2 + (0.0-0.4)**2)
    assert np.isclose(result[0][0].weight_distance, expected_dist_0_0, atol=1e-6)
    
    # Distance between Vertex 2 [0.0, 1.0] and Triangle 1 [0.267, 0.733]
    expected_dist_2_1 = np.sqrt((0.0-0.267)**2 + (1.0-0.733)**2)
    assert np.isclose(result[2][0].weight_distance, expected_dist_2_1, atol=1e-6)


@pytest.mark.skip(reason="Errors with mel.eval for progress bar")
def test_get_weight_distance_with_invalid_influences():
    """Test weight distance calculation with invalid influences (unmapped joints)."""
    # Source triangles
    src_triangles = np.array([0, 1, 2], dtype=np.int32)
    src_num_tri = 1
    
    # Source mesh vertex weights - 3 influences
    src_weight = [
        [0.5, 0.3, 0.2],  # Vertex 0
        [0.4, 0.4, 0.2],  # Vertex 1
        [0.2, 0.6, 0.2],  # Vertex 2
    ]
    
    # Target mesh vertex weights - 3 influences
    tar_weight = [
        [0.5, 0.3, 0.2],  # Vertex 0
    ]
    
    # Influence 1 is invalid (-1)
    tar_joints_retarget = [0, -1, 2]
    
    # Call the function with patched Maya environment
    with patch('maya.mel.eval', return_value=None):
        with patch('maya.cmds.about', return_value=True):  # Treat as batch mode
            result = get_weight_distance(
                src_triangles=src_triangles,
                src_num_tri=src_num_tri,
                src_weight=src_weight,
                tar_weight=tar_weight,
                tar_joints_retarget=tar_joints_retarget,
                top_n=1
            )
    
    # Validate results
    assert len(result) == 1  # One target vertex
    assert len(result[0]) == 1  # top_n=1 so one result
    
    # Distance calculation should only use influences 0 and 2
    # Triangle avg weights: (0.5+0.4+0.2)/3 = 0.367 (influence 0), (0.2+0.2+0.2)/3 = 0.2 (influence 2)
    # Distance to Vertex 0 [0.5, 0.2] (only influences 0 and 2)
    expected_dist = np.sqrt((0.5-0.367)**2 + (0.2-0.2)**2)
    assert np.isclose(result[0][0].weight_distance, expected_dist, atol=1e-6)