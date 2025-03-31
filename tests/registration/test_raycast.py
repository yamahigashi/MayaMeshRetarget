from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ymt_mesh_retarget.registration.raycast import RaycastEngine, StandardRaycastEngine, get_raycast_engine


class TestRaycastEngine:
    """Test the abstract RaycastEngine class."""

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        # Create mock triangles and indices
        triangles = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        triangle_indices = np.array([0, 1, 2], dtype=np.int32)

        # Create a subclass that doesn't implement abstract methods
        class TestEngine(RaycastEngine):
            pass

        # Instantiating should fail because _prepare_scene is not implemented
        with pytest.raises(NotImplementedError):
            engine = TestEngine(triangles, triangle_indices)

        # Create a minimal subclass that only implements _prepare_scene
        class MinimalEngine(RaycastEngine):
            def _prepare_scene(self) -> None:
                pass

        # Instantiating should work but other methods should still fail
        engine = MinimalEngine(triangles, triangle_indices)

        with pytest.raises(NotImplementedError):
            engine.cast_ray(np.array([0, 0, 1]), np.array([0, 0, -1]))

        with pytest.raises(NotImplementedError):
            engine.cast_rays(
                np.array([[0, 0, 1]]),
                np.array([[0, 0, -1]]),
            )


class TestStandardRaycastEngine:
    """Test the StandardRaycastEngine implementation."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create a simple triangle
        self.triangles = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 1, 0], [0, 0, 1], [1, 0, 1],
        ], dtype=np.float32)

        # Single triangle
        self.triangle_indices = np.array([0, 1, 2], dtype=np.int32)

        # Create engine
        self.engine = StandardRaycastEngine(self.triangles, self.triangle_indices)

    def test_initialization(self):
        """Test initialization and _prepare_scene."""
        assert self.engine.triangles is not None
        assert self.engine.triangle_indices is not None
        assert self.engine.triangle_vertices.shape == (1, 3, 3)  # 1 triangle, 3 vertices, 3 coords

    def test_cast_ray_hit(self):
        """Test ray casting that hits a triangle."""
        # Ray directly above the triangle pointing down
        origin = np.array([0.25, 0.25, 1], dtype=np.float64)
        direction = np.array([0, 0, -1], dtype=np.float64)

        result = self.engine.cast_ray(origin, direction)

        assert result is not None
        assert result["primID"] == 0  # Hit the first (and only) triangle
        assert result["tfar"] > 0  # Positive distance

        # The hit point should be at z=0
        assert np.isclose(origin[0] + direction[0] * result["tfar"], 0.25)
        assert np.isclose(origin[1] + direction[1] * result["tfar"], 0.25)
        assert np.isclose(origin[2] + direction[2] * result["tfar"], 0)

    def test_cast_ray_miss(self):
        """Test ray casting that misses all triangles."""
        # Ray outside the triangle
        origin = np.array([2, 2, 1], dtype=np.float64)
        direction = np.array([0, 0, -1], dtype=np.float64)

        result = self.engine.cast_ray(origin, direction)
        assert result is None

    def test_cast_rays_batch(self):
        """Test batch ray casting."""
        # Create batch of rays
        origins = np.array([
            [0.25, 0.25, 1],  # Hit
            [2, 2, 1],        # Miss
        ], dtype=np.float32)

        directions = np.array([
            [0, 0, -1],  # Down
            [0, 0, -1],  # Down
        ], dtype=np.float32)

        results = self.engine.cast_rays(origins, directions)

        # Check results structure
        assert "primID" in results
        assert "geomID" in results
        assert "tfar" in results
        assert "u" in results
        assert "v" in results

        # First ray should hit, second should miss
        assert results["geomID"][0] >= 0
        assert results["geomID"][1] < 0


def test_get_raycast_engine():
    """Test the get_raycast_engine factory function."""
    triangles = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangle_indices = np.array([0, 1, 2], dtype=np.int32)

    # Test with Embree unavailable
    with patch('ymt_mesh_retarget.registration.raycast.EMBREE_AVAILABLE', False):
        engine = get_raycast_engine(triangles, triangle_indices)
        assert isinstance(engine, StandardRaycastEngine)

    # Test with force_standard=True
    with patch('ymt_mesh_retarget.registration.raycast.EMBREE_AVAILABLE', True):
        engine = get_raycast_engine(triangles, triangle_indices, force_standard=True)
        assert isinstance(engine, StandardRaycastEngine)

    # Test with Embree available but not forcing standard
    # This requires mocking the EmbreeRaycastEngine since Embree might not be available
    with patch('ymt_mesh_retarget.registration.raycast.EMBREE_AVAILABLE', True):
        with patch('ymt_mesh_retarget.registration.raycast.EmbreeRaycastEngine') as mock_embree:
            mock_instance = MagicMock()
            mock_embree.return_value = mock_instance

            engine = get_raycast_engine(triangles, triangle_indices)
            mock_embree.assert_called_once()
