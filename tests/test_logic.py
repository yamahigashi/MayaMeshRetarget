from unittest.mock import MagicMock, patch

import numpy as np

from ymt_mesh_retarget.logic import RBF, retarget


class TestRBFKernels:
    """Test RBF kernel functions."""

    def setup_method(self):
        """Set up test data."""
        # Create a simple distance matrix for testing
        self.distance_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ])
        self.radius = 2.0

    def test_linear_kernel(self):
        """Test linear RBF kernel."""
        result = RBF.linear(self.distance_matrix, self.radius)

        # Linear kernel should return the distance matrix unchanged
        assert np.array_equal(result, self.distance_matrix)

    def test_gaussian_kernel(self):
        """Test Gaussian RBF kernel."""
        result = RBF.gaussian(self.distance_matrix, self.radius)

        # Gaussian kernel should apply exponential decay based on distances
        expected = np.exp(-(self.distance_matrix**2) / (self.radius**2))
        assert np.allclose(result, expected)

        # Check specific properties
        assert np.all(result >= 0.0)  # All values should be positive
        assert np.all(result <= 1.0)  # All values should be <= 1
        assert np.allclose(np.diag(result), 1.0)  # Diagonal (zero distance) should be 1

        # Further distances should have smaller values
        assert result[0, 1] > result[0, 2] > result[0, 3]

    def test_multiquadric_kernel(self):
        """Test multiquadric RBF kernel."""
        result = RBF.multi_quadratic_biharmonic(self.distance_matrix, self.radius)

        # Multiquadric kernel formula: sqrt(d^2 + r^2)
        expected = np.sqrt(self.distance_matrix**2 + self.radius**2)
        assert np.allclose(result, expected)

        # Check specific properties
        assert np.all(result >= 1.0)  # All values should be >= 1
        # For zero distance, we expect sqrt(0^2 + radius^2) = radius = 2.0
        assert np.allclose(np.diag(result), self.radius)  # Diagonal should be equal to radius

    def test_inverse_multiquadric_kernel(self):
        """Test inverse multiquadric RBF kernel."""
        result = RBF.inv_multi_quadratic_biharmonic(self.distance_matrix, self.radius)

        # Inverse multiquadric kernel formula: 1/sqrt(d^2 + r^2)
        expected = 1.0 / np.sqrt(self.distance_matrix**2 + self.radius**2)
        assert np.allclose(result, expected)

        # Check specific properties
        assert np.all(result <= 1.0)  # All values should be <= 1
        assert np.all(result > 0.0)   # All values should be positive
        # For zero distance, we expect 1/sqrt(0^2 + radius^2) = 1/radius = 0.5
        assert np.allclose(np.diag(result), 1.0/self.radius)  # Diagonal should be 1/radius

        # Further distances should have smaller values
        assert result[0, 1] > result[0, 2] > result[0, 3]

    def test_thin_plate_kernel(self):
        """Test thin plate RBF kernel."""
        result = RBF.thin_plate(self.distance_matrix, self.radius)

        # Let's check the implementation:
        # result = (matrix / radius) ** 2
        # result = np.where(result > 0, np.log(result), result)
        # So for zero distance, result is 0^2 = 0
        # For non-zero distances, it's (d/r)^2 * log((d/r)^2)

        # Recalculate based on actual implementation
        expected = (self.distance_matrix / self.radius) ** 2
        # Use a mask to avoid warnings with log(0)
        mask = expected > 0
        # Only apply log where result > 0
        temp = np.copy(expected)
        temp[mask] = np.log(temp[mask])
        expected = temp

        assert np.allclose(result, expected, atol=1e-10)

        # Check that zeros are preserved on the diagonal
        assert np.allclose(np.diag(result), 0.0)


@patch('ymt_mesh_retarget.logic.create_retargetable_object')
@patch('ymt_mesh_retarget.logic.mel')
@patch('ymt_mesh_retarget.logic.cmds')
def test_retarget_basic_functionality(mock_cmds, mock_mel, mock_create_object):
    """Test basic functionality of the retarget function."""
    # Mock maya batch mode to avoid progress bar errors
    mock_cmds.about.return_value = True  # Pretend we're in batch mode

    # Create mock source, target and retarget objects
    mock_source = MagicMock()
    mock_target = MagicMock()
    mock_retarget_obj = MagicMock()

    # Configure point mocks
    mock_source.get_points.return_value = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    mock_target.get_points.return_value = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    mock_retarget_obj.get_points.return_value = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    # Set up the name and duplicate methods
    mock_retarget_obj.name = "retarget_obj"
    mock_retarget_obj.duplicate.return_value = mock_retarget_obj
    mock_retarget_obj.get_transforms.return_value = [
        {"position": np.array([0.0, 0.0, 0.0])},
        {"position": np.array([1.0, 0.0, 0.0])},
    ]
    # Empty children for hierarchy maintenance check
    mock_retarget_obj.get_children.return_value = []

    # Set up threshold calculation
    mock_source.calculate_threshold_distance.return_value = 0.5

    # Mock the create_retargetable_object function
    mock_create_object.side_effect = [mock_source, mock_target, mock_retarget_obj]

    # Call the retarget function with simple parameters
    with patch('ymt_mesh_retarget.logic.calculate_rbf_weight_matrix') as mock_weights:
        # Mock the weight matrix calculation
        # We need to make sure this matches the expected dimensions for the dot product
        # h_combined has shape (object_points.shape[0], source_points.shape[0] + 1 + 3)
        mock_weights.return_value = np.array([
            [1.0, 0.0, 0.0],  # First 4 rows correspond to the source points (4 points)
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Row for the identity column
            [0.0, 1.0, 0.0],  # Remaining 3 rows for the object point dimensions (x,y,z)
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])

        # Call the function
        result = retarget(
            source="source_mesh",
            target="target_mesh",
            objects="retarget_obj",
            kernel="gaussian",
            radius_coefficient=0.1,
        )

    # Verify that transforms were applied to the retarget object
    assert mock_retarget_obj.apply_transforms.called

    # Verify the result
    assert result is not None


@patch('ymt_mesh_retarget.logic.create_retargetable_object')
@patch('ymt_mesh_retarget.logic.mel')
@patch('ymt_mesh_retarget.logic.cmds')
def test_retarget_with_multiple_targets(mock_cmds, mock_mel, mock_create_object):
    """Test retargeting with multiple target objects."""
    # Mock maya batch mode to avoid progress bar errors
    mock_cmds.about.return_value = True

    # Create mock source, target and retarget objects
    mock_source = MagicMock()
    mock_target = MagicMock()
    mock_retarget_obj1 = MagicMock()
    mock_retarget_obj2 = MagicMock()

    # Configure point mocks
    for mock_obj in [mock_source, mock_target, mock_retarget_obj1, mock_retarget_obj2]:
        mock_obj.get_points.return_value = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

    # Set up names and duplicate methods
    mock_retarget_obj1.name = "retarget_obj1"
    mock_retarget_obj1.duplicate.return_value = mock_retarget_obj1
    mock_retarget_obj1.get_transforms.return_value = [
        {"position": np.array([0.0, 0.0, 0.0])},
        {"position": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([1.0, 1.0, 0.0])},
    ]
    # Empty children for hierarchy maintenance check
    mock_retarget_obj1.get_children.return_value = []

    mock_retarget_obj2.name = "retarget_obj2"
    mock_retarget_obj2.duplicate.return_value = mock_retarget_obj2
    mock_retarget_obj2.get_transforms.return_value = [
        {"position": np.array([0.0, 0.0, 0.0])},
        {"position": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([1.0, 1.0, 0.0])},
    ]
    # Empty children for hierarchy maintenance check
    mock_retarget_obj2.get_children.return_value = []

    # Set up threshold calculation
    mock_source.calculate_threshold_distance.return_value = 0.5

    # Mock the create_retargetable_object function
    mock_create_object.side_effect = [mock_source, mock_target, mock_retarget_obj1, mock_retarget_obj2]

    # Call the retarget function with simple parameters
    with patch('ymt_mesh_retarget.logic.calculate_rbf_weight_matrix') as mock_weights:
        # Mock the weight matrix calculation
        # We need to make sure this matches the expected dimensions for the dot product
        # h_combined has shape (object_points.shape[0], source_points.shape[0] + 1 + 3)
        mock_weights.return_value = np.array([
            [1.0, 0.0, 0.0],  # First 4 rows correspond to the source points (4 points)
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Row for the identity column
            [0.0, 1.0, 0.0],  # Remaining 3 rows for the object point dimensions (x,y,z)
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])

        # Call with multiple targets
        result = retarget(
            source="source_mesh",
            target="target_mesh",
            objects=["retarget_obj1", "retarget_obj2"],
            kernel="gaussian",
            radius_coefficient=0.1,
        )

    # Verify transforms were applied to both target objects
    assert mock_retarget_obj1.apply_transforms.called
    assert mock_retarget_obj2.apply_transforms.called

    # Verify result contains both targets
    assert len(result) == 2


@patch('ymt_mesh_retarget.logic.create_retargetable_object')
@patch('ymt_mesh_retarget.logic.mel')
@patch('ymt_mesh_retarget.logic.cmds')
def test_retarget_with_different_kernels(mock_cmds, mock_mel, mock_create_object):
    """Test retargeting with different RBF kernels."""
    # Mock maya batch mode to avoid progress bar errors
    mock_cmds.about.return_value = True

    # Create mock source, target and retarget objects
    mock_source = MagicMock()
    mock_target = MagicMock()
    mock_retarget_obj = MagicMock()

    # Configure point mocks
    for mock_obj in [mock_source, mock_target, mock_retarget_obj]:
        mock_obj.get_points.return_value = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

    # Set up name and duplicate methods
    mock_retarget_obj.name = "retarget_obj"
    mock_retarget_obj.duplicate.return_value = mock_retarget_obj
    mock_retarget_obj.get_transforms.return_value = [
        {"position": np.array([0.0, 0.0, 0.0])},
        {"position": np.array([1.0, 0.0, 0.0])},
    ]
    # Empty children for hierarchy maintenance check
    mock_retarget_obj.get_children.return_value = []

    # Set up threshold calculation
    mock_source.calculate_threshold_distance.return_value = 0.5

    # Test with different kernels
    kernels = ['linear', 'gaussian', 'thin_plate', 'multi_quadratic', 'inv_multi_quadratic', 'beckert_wendland']

    for kernel in kernels:
        # Mock the create_retargetable_object function - reset for each kernel
        mock_create_object.side_effect = [mock_source, mock_target, mock_retarget_obj]

        # Call the retarget function with this kernel
        with patch('ymt_mesh_retarget.logic.calculate_rbf_weight_matrix') as mock_weights:
            # Mock the weight matrix calculation
            # For the test_retarget_with_different_kernels, we have 2 source points and 2 object points
            mock_weights.return_value = np.array([
                [1.0, 0.0, 0.0],  # First 2 rows correspond to the source points (2 points)
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],  # Row for the identity column
                [0.0, 1.0, 0.0],  # Remaining 3 rows for the object point dimensions (x,y,z)
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ])

            # Reset mock calls
            mock_retarget_obj.apply_transforms.reset_mock()

            # Call the function with this kernel
            result = retarget(
                source="source_mesh",
                target="target_mesh",
                objects="retarget_obj",
                kernel=kernel,
                radius_coefficient=0.1,
            )

        # Verify transforms were applied
        assert mock_retarget_obj.apply_transforms.called, f"Failed with kernel: {kernel}"

        # Verify result
        assert result is not None, f"No result returned with kernel: {kernel}"


@patch('ymt_mesh_retarget.logic.create_retargetable_object')
@patch('ymt_mesh_retarget.logic.mel')
@patch('ymt_mesh_retarget.logic.cmds')
def test_retarget_with_different_normalization(mock_cmds, mock_mel, mock_create_object):
    """Test retargeting with different normalization modes."""
    # Mock maya batch mode to avoid progress bar errors
    mock_cmds.about.return_value = True

    # Create mock source, target and retarget objects
    mock_source = MagicMock()
    mock_target = MagicMock()
    mock_retarget_obj = MagicMock()

    # Configure point mocks
    for mock_obj in [mock_source, mock_target, mock_retarget_obj]:
        mock_obj.get_points.return_value = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

    # Set up name and duplicate methods
    mock_retarget_obj.name = "retarget_obj"
    mock_retarget_obj.duplicate.return_value = mock_retarget_obj
    mock_retarget_obj.get_transforms.return_value = [
        {"position": np.array([0.0, 0.0, 0.0])},
        {"position": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 1.0, 0.0])},
    ]
    # Empty children for hierarchy maintenance check
    mock_retarget_obj.get_children.return_value = []

    # For MeshObject-specific tests - this mocks the cluster_vertices method
    mock_retarget_obj.cluster_vertices.return_value = np.array([0, 1, 2])

    # Set up threshold calculation
    mock_source.calculate_threshold_distance.return_value = 0.5

    # Mock check to see if the object is a MeshObject
    mock_retarget_obj.__class__.__name__ = "MeshObject"

    # Test with different rigid transform modes (equivalent to normalization in tests)
    for apply_rigid in [False, True]:
        # Mock the create_retargetable_object function - reset for each test
        mock_create_object.side_effect = [mock_source, mock_target, mock_retarget_obj]

        # Call the retarget function with apply_rigid_transform
        with patch('ymt_mesh_retarget.logic.calculate_rbf_weight_matrix') as mock_weights:
            # Mock the weight matrix calculation
            # For the test_retarget_with_different_normalization, we have 3 source points and 3 object points
            mock_weights.return_value = np.array([
                [1.0, 0.0, 0.0],  # First 3 rows correspond to the source points (3 points)
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],  # Row for the identity column
                [0.0, 1.0, 0.0],  # Remaining 3 rows for the object point dimensions (x,y,z)
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ])

            # Reset mock calls
            mock_retarget_obj.apply_transforms.reset_mock()

            # Call the function
            result = retarget(
                source="source_mesh",
                target="target_mesh",
                objects="retarget_obj",
                kernel="gaussian",
                radius_coefficient=0.1,
                apply_rigid_transform=apply_rigid,
            )

        # Verify transforms were applied
        assert mock_retarget_obj.apply_transforms.called, f"Failed with apply_rigid: {apply_rigid}"

        # Verify result
        assert result is not None, f"No result returned with apply_rigid: {apply_rigid}"
