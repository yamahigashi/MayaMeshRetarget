from unittest.mock import patch

import numpy as np
import pytest

from ymt_mesh_retarget.inpaint import inpaint_distance, segregate_vertices_by_confidence


@pytest.fixture
def mock_vertex_data():
    """Create mock vertex data for testing."""
    # Create array with position, normal, and confidence values
    # Shape: (num_vertices, 7)
    # Where 7 = [x, y, z, nx, ny, nz, confidence]
    return np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # Vertex 0: confident
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # Vertex 1: confident
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5],  # Vertex 2: less confident
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.2],  # Vertex 3: unconvinced
    ])


@patch('ymt_mesh_retarget.inpaint.__create_vertex_data_array')
@patch('ymt_mesh_retarget.inpaint.__get_closest_points_by_kdtree')
@patch('ymt_mesh_retarget.inpaint.__filter_high_confidence_matches')
@patch('ymt_mesh_retarget.util.calculate_threshold_distance')
def test_segregate_vertices_by_confidence(
    mock_calc_threshold,
    mock_filter_matches,
    mock_get_closest,
    mock_create_vertex_data,
    mock_vertex_data,
):
    """Test segregation of vertices by confidence level."""
    # Configure mocks
    mock_calc_threshold.return_value = 0.1
    mock_create_vertex_data.return_value = mock_vertex_data
    mock_get_closest.return_value = np.ones((4, 4))  # Dummy closest points data

    # Mock the filter function to return indices of confident vertices
    confident_indices = np.array([0, 1])  # First two vertices are confident
    mock_filter_matches.return_value = confident_indices

    # Test with a single target mesh
    src_path = "source_mesh"
    dst_path = "target_mesh"

    confident, unconvinced = segregate_vertices_by_confidence(src_path, dst_path)

    # Verify results
    assert np.array_equal(confident, np.array([0, 1]))
    assert np.array_equal(unconvinced, np.array([2, 3]))

    # Test with multiple target meshes
    dst_paths = ["target_mesh1", "target_mesh2"]

    confident, unconvinced = segregate_vertices_by_confidence(src_path, dst_paths)

    # Verify results
    assert np.array_equal(confident, np.array([0, 1]))
    assert np.array_equal(unconvinced, np.array([2, 3]))


@patch('ymt_mesh_retarget.inpaint.__inpaint_distance_matrix')
@patch('ymt_mesh_retarget.inpaint.segregate_vertices_by_confidence')
def test_inpaint_distance(mock_segregate, mock_inpaint_matrix):
    """Test the inpaint_distance function."""
    # Configure mocks
    confident_indices = np.array([0, 1])
    unconvinced_indices = np.array([2, 3])
    mock_segregate.return_value = (confident_indices, unconvinced_indices)

    # Create a simple distance matrix with known and unknown values
    original_distance = np.array([
        [0.0, 1.0, 2.0, 3.0],  # Values for vertex 0 (confident)
        [1.0, 0.0, 1.0, 2.0],  # Values for vertex 1 (confident)
        [2.0, 1.0, 0.0, 1.0],  # Values for vertex 2 (unconvinced)
        [3.0, 2.0, 1.0, 0.0],  # Values for vertex 3 (unconvinced)
    ])

    # Create inpainted distance matrix (dummy result)
    inpainted_distance = np.copy(original_distance)
    inpainted_distance[2, :] = 5.0  # Modified values for unconvinced vertex 2
    inpainted_distance[3, :] = 6.0  # Modified values for unconvinced vertex 3

    mock_inpaint_matrix.return_value = inpainted_distance

    # Create vertex labels (isolated = -1, clusters start from 0)
    labels = np.array([-1, -1, 0, 0])  # Two isolated confident vertices, two clustered unconvinced vertices

    # Call the function
    src_path = "source_mesh"
    dst_paths = ["target_mesh"]

    result = inpaint_distance(
        source_path=src_path,
        target_paths=dst_paths,
        distances=original_distance,
        labels=labels,
    )

    # Verify the result
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
    assert np.array_equal(result, inpainted_distance)


@patch('ymt_mesh_retarget.inpaint.__inpaint_distance_matrix')
@patch('ymt_mesh_retarget.inpaint.segregate_vertices_by_confidence')
def test_inpaint_distance_all_confident(mock_segregate, mock_inpaint_matrix):
    """Test inpaint_distance when all vertices are confident (no inpainting needed)."""
    # Configure mocks - all vertices are confident
    confident_indices = np.array([0, 1, 2, 3])
    unconvinced_indices = np.array([])
    mock_segregate.return_value = (confident_indices, unconvinced_indices)

    # Original distance matrix
    original_distance = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ])

    # Create vertex labels (isolated = -1, clusters start from 0)
    labels = np.array([-1, -1, 0, 0])

    # Call the function
    src_path = "source_mesh"
    dst_paths = ["target_mesh"]

    result = inpaint_distance(
        source_path=src_path,
        target_paths=dst_paths,
        distances=original_distance,
        labels=labels,
    )

    # Verify the result - should be the original distance matrix
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
    assert np.array_equal(result, original_distance)

    # Verify inpaint_matrix was not called
    mock_inpaint_matrix.assert_not_called()
