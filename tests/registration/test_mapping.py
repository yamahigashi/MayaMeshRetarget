from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ymt_mesh_retarget.registration.core import BoneNode, JointNode, MappingResult, RegistrationOptions
from ymt_mesh_retarget.registration.mapping import (
    create_optimized_correspondence_points,
    find_nearest_vertex_index,
    get_mapping_points,
)


def test_find_nearest_vertex_index():
    """Test find_nearest_vertex_index function."""
    # This is placeholder behavior in the actual implementation
    position = np.array([0, 0, 0])
    triangle_idx = 0
    raycast_data = {}

    result = find_nearest_vertex_index(position, triangle_idx, raycast_data)
    assert result == -1  # Current implementation returns -1


def test_get_mapping_points():
    """Test get_mapping_points function."""
    # Create mock target points
    target_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)

    # Create mock joints
    joint1 = JointNode(
        path=None,
        index=0,
        detail_name="joint1",
        position=np.array([0, 0, 0], dtype=np.float64),
        matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    )

    joint2 = JointNode(
        path=None,
        index=1,
        detail_name="joint2",
        position=np.array([1, 0, 0], dtype=np.float64),
        matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1),
    )

    target_joint_group = [joint1, joint2]

    # Create mock bones
    bone = BoneNode(start_joint_index=0, end_joint_index=1)
    target_bone_group = [bone]

    # Create mock weights (vertex, joint) mapping
    target_weights = [
        [0.8, 0.2],  # First vertex weights
        [0.2, 0.8],  # Second vertex weights
        [0.5, 0.5],  # Third vertex weights
    ]

    # Create mock joint names
    target_joint_names = ["joint1", "joint2"]

    # Call the function
    results = get_mapping_points(
        target_points,
        target_joint_group,
        target_bone_group,
        target_weights,
        target_joint_names,
    )

    # Check results
    assert len(results) == 3  # One for each target point
    assert all(isinstance(r, MappingResult) for r in results)

    # Check first vertex result
    assert results[0].vertex_index == 0
    assert len(results[0].node_array) == 1  # One mapping node

    # Check mapping node properties
    node = results[0].node_array[0]
    assert node.bone_index == 0
    assert np.allclose(node.point, np.array([0, 0, 0]))
    assert node.weight == 0.8


@patch('ymt_mesh_retarget.registration.mapping.logger')
def test_create_optimized_correspondence_points(mock_logger):
    """Test create_optimized_correspondence_points function."""
    # This test is more complex due to the many dependencies
    # We'll test for the expected exception when source_mesh or target_mesh is missing

    # Create minimal inputs
    raycast_result_array = [[]]  # Empty results
    src_mapping_points = [MappingResult(vertex_index=0)]
    source_points = np.array([[0, 0, 0]], dtype=np.float64)

    # Options with scoring components
    options = MagicMock(spec=RegistrationOptions)
    options.scoring_components = []

    # Test missing mesh objects should raise an exception
    with pytest.raises(NotImplementedError):
        create_optimized_correspondence_points(
            raycast_result_array=raycast_result_array,
            src_mapping_points=src_mapping_points,
            source_points=source_points,
            options=options,
        )
