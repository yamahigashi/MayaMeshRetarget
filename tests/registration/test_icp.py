"""Tests for ICP-based mesh retargeting implementation."""

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from maya import cmds

from ymt_mesh_retarget.objects import MeshObject
from ymt_mesh_retarget.registration.icp import (
    ICPOptions,
    JointParameter,
    MeshRetargetICP,
    align_mesh_with_icp,
    compute_bbox_based_scaling,
    compute_error_metric,
    find_closest_points,
)


class TestICPFunctions(unittest.TestCase):
    """Test individual functions of the ICP module."""

    def test_compute_error_metric(self):
        """Test the compute_error_metric function."""
        # Create sample point sets
        source_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        target_points = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 3]], dtype=np.float64)

        # Compute error
        error = compute_error_metric(source_points, target_points)

        # Expected error: mean squared distance between corresponding points
        # Each z-coordinate differs by 1, so squared distance = 1
        # Mean of [1, 1, 1] = 1
        self.assertEqual(error, 1.0)

    def test_find_closest_points(self):
        """Test the find_closest_points function."""
        # Create sample point sets
        source_points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
        ], dtype=np.float64)

        target_points = np.array([
            [0, 0, 0.5],
            [1, 1, 1.5],
            [2, 2, 2.5],
            [3, 3, 3.5],
        ], dtype=np.float64)

        # Test with full sampling
        sampled_source, corresponding_target, sample_indices = find_closest_points(
            source_points, target_points, sampling_rate=1.0,
        )

        # Check that sampled_source has same size as source_points
        self.assertEqual(len(sampled_source), len(source_points))

        # Check that corresponding_target has same length
        self.assertEqual(len(sampled_source), len(corresponding_target))

        # Check that the closest point to [4, 4, 4] is [3, 3, 3.5]
        # (because it's the closest available)
        self.assertTrue(np.allclose(corresponding_target[-1], np.array([3, 3, 3.5])))

        # Test with partial sampling
        sampled_source, corresponding_target, sample_indices = find_closest_points(
            source_points, target_points, sampling_rate=0.5,
        )

        # Check that we've sampled approximately 50% of the points
        self.assertLessEqual(len(sampled_source), 3)  # 5 * 0.5 = 2.5, so 2 or 3 points


@pytest.mark.maya
class TestICPWithMaya:
    """Test ICP functions that require Maya to be running."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up test environment."""
        # Create simple test scene
        cls.create_test_scene()

    @classmethod
    def teardown_class(cls) -> None:
        """Clean up after tests."""
        # Delete test scene
        cmds.delete(cls.source_root, cls.target_root)

    @classmethod
    def create_test_scene(cls) -> None:
        """Create a simple test scene with two meshes and skeletons."""
        # Create source skeleton
        cls.source_root = cmds.joint(p=(0, 0, 0), name="source_root")
        cls.source_spine = cmds.joint(p=(0, 5, 0), name="source_spine")
        cls.source_head = cmds.joint(p=(0, 10, 0), name="source_head")

        # Create target skeleton (slightly different position and scale)
        cls.target_root = cmds.joint(p=(5, 0, 0), name="target_root")
        cls.target_spine = cmds.joint(p=(5, 7, 0), name="target_spine")
        cls.target_head = cmds.joint(p=(5, 15, 0), name="target_head")

        # Create source mesh - simple cube
        cls.source_mesh = cmds.polyCube(width=2, height=10, depth=2, name="source_mesh")[0]

        # Create target mesh - slightly different cube
        cls.target_mesh = cmds.polyCube(width=3, height=15, depth=3, name="target_mesh")[0]
        cmds.move(5, 0, 0, cls.target_mesh)

        # Bind meshes to skeletons
        cls.source_skin = cmds.skinCluster(
            cls.source_root, cls.source_spine, cls.source_head,
            cls.source_mesh, tsb=True, name="source_skin")[0]

        cls.target_skin = cmds.skinCluster(
            cls.target_root, cls.target_spine, cls.target_head,
            cls.target_mesh, tsb=True, name="target_skin")[0]

    def test_bbox_scaling(self):
        """Test the compute_bbox_based_scaling function with Maya objects."""
        source_mesh = MeshObject(self.source_mesh)
        target_mesh = MeshObject(self.target_mesh)

        scale_factor, position_offset = compute_bbox_based_scaling(source_mesh, target_mesh)

        # Target is 1.5x larger in all dimensions
        assert math.isclose(scale_factor, 1.5, rel_tol=0.1)

        # Target is offset by 5 units in X
        assert math.isclose(position_offset[0], 5.0, rel_tol=0.1)

    @patch('ymt_mesh_retarget.registration.icp.minimize')
    def test_icp_setup(self, mock_minimize):
        """Test that ICP setup works correctly."""
        # Mock the SciPy minimize function to avoid actual optimization
        mock_result = MagicMock()
        mock_result.x = np.zeros(9)  # Mock optimized parameters
        mock_minimize.return_value = mock_result

        # Create ICP options
        options = ICPOptions(
            max_iterations=3,
            staged_optimization=True,
            include_scale=True,
            verbose=False,
        )

        # Create ICP object
        icp = MeshRetargetICP(
            source_mesh=self.source_mesh,
            target_mesh=self.target_mesh,
            source_root_joint=self.source_root,
            options=options,
        )

        # Prepare data (this should succeed without errors)
        icp.prepare_data()

        # Check that the data preparation worked
        assert len(icp.joint_nodes) > 0
        assert icp.skeleton_state is not None

        # Run ICP (with mocked optimization)
        error = icp.run_icp()

        # Check that the process ran (we don't care about the actual error value
        # since optimization is mocked)
        assert isinstance(error, float)

        # Verify minimize was called at least once
        assert mock_minimize.call_count > 0

        # Test the public function
        align_mesh_with_icp(
            source_mesh=self.source_mesh,
            target_mesh=self.target_mesh,
            source_root_joint=self.source_root,
            options=options,
        )


class TestJointParameter:
    """Test the JointParameter class."""

    def test_parameter_conversion(self):
        """Test conversion between flat arrays and structured parameters."""
        param = JointParameter(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=np.array([30.0, 45.0, 60.0]),
            scale=np.array([1.5, 1.5, 1.5]),
            joint_index=0,
            active=True,
        )

        # Test to_flat_array with scale
        flat = param.to_flat_array(include_scale=True)
        assert len(flat) == 9
        assert np.allclose(flat[:3], [1.0, 2.0, 3.0])
        assert np.allclose(flat[3:6], [30.0, 45.0, 60.0])
        assert np.allclose(flat[6:9], [1.5, 1.5, 1.5])

        # Test to_flat_array without scale
        flat = param.to_flat_array(include_scale=False)
        assert len(flat) == 6
        assert np.allclose(flat[:3], [1.0, 2.0, 3.0])
        assert np.allclose(flat[3:6], [30.0, 45.0, 60.0])

        # Test from_flat_array with scale
        new_param = JointParameter()
        new_param.from_flat_array(
            np.array([4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 2.0, 2.0, 2.0]),
            include_scale=True,
        )
        assert np.allclose(new_param.translation, [4.0, 5.0, 6.0])
        assert np.allclose(new_param.rotation, [10.0, 20.0, 30.0])
        assert np.allclose(new_param.scale, [2.0, 2.0, 2.0])

        # Test from_flat_array without scale
        new_param = JointParameter()
        new_param.from_flat_array(
            np.array([7.0, 8.0, 9.0, 15.0, 25.0, 35.0]),
            include_scale=False,
        )
        assert np.allclose(new_param.translation, [7.0, 8.0, 9.0])
        assert np.allclose(new_param.rotation, [15.0, 25.0, 35.0])
        assert np.allclose(new_param.scale, [1.0, 1.0, 1.0])  # Default scale


if __name__ == "__main__":
    unittest.main()
