import numpy as np
import pytest
from typing import Any

from ymt_mesh_retarget.registration.scoring_components import (
    IScoringComponent,
    DistanceScoring,
    RayQualityScoring,
    NormalScoring,
    WeightVectorScoring,
    LaplacianScoring,
    create_default_scoring_components
)


class SimpleScoringComponent(IScoringComponent):
    """Simple implementation of IScoringComponent for testing."""
    
    def __init__(self, weight: float = 1.0):
        self.weight_value = weight
    
    def compute_score(self, context: dict[str, Any]) -> float:
        return 0.5
    
    def get_weight(self) -> float:
        return self.weight_value


def test_abstract_base_class():
    """Test that IScoringComponent is an abstract base class and can't be instantiated directly."""
    # Should not be able to instantiate the abstract base class
    with pytest.raises(TypeError):
        IScoringComponent()
    
    # Should be able to instantiate a concrete subclass
    component = SimpleScoringComponent()
    assert component.compute_score({}) == 0.5
    assert component.get_weight() == 1.0


def test_distance_scoring():
    """Test DistanceScoring component."""
    # Create component with default weight
    component = DistanceScoring()
    assert component.weight == 1.0
    
    # Test with no distance in context
    score = component.compute_score({})
    assert score == 0.0
    
    # Test with infinite distance
    score = component.compute_score({'distance': float('inf')})
    assert score == 0.0
    
    # Test with zero distance
    score = component.compute_score({'distance': 0.0})
    assert score == 0.0
    
    # Test with normal distance
    score = component.compute_score({'distance': 1.0})
    expected = np.exp(-1.0)
    assert np.isclose(score, expected)
    
    # Test with custom weight
    component = DistanceScoring(weight=0.5)
    score = component.compute_score({'distance': 1.0})
    expected = 0.5 * np.exp(-1.0)
    assert np.isclose(score, expected)
    assert component.get_weight() == 0.5


def test_ray_quality_scoring():
    """Test RayQualityScoring component."""
    # Create component with default weight
    component = RayQualityScoring()
    assert component.weight == 0.5
    
    # Test with no raycast_result in context
    score = component.compute_score({})
    assert score == 0.0
    
    # Create a mock raycast result
    class MockRaycastResult:
        def __init__(self, weight: float, relate_distance: float):
            self.weight = weight
            self.relate_distance = relate_distance
    
    # Test with raycast result
    mock_result = MockRaycastResult(weight=0.8, relate_distance=0.5)
    score = component.compute_score({'raycast_result': mock_result})
    # Expected: 0.5 (component weight) * 0.8 (ray weight) * (1.0 - 0.5) (relate distance)
    expected = 0.5 * 0.8 * 0.5
    assert np.isclose(score, expected)
    
    # Test with relate_distance > 1.0 (should be clamped)
    mock_result = MockRaycastResult(weight=0.8, relate_distance=2.0)
    score = component.compute_score({'raycast_result': mock_result})
    # Expected: 0.5 (component weight) * 0.8 (ray weight) * (1.0 - 1.0) (clamped relate distance)
    expected = 0.0
    assert np.isclose(score, expected)


def test_normal_scoring():
    """Test NormalScoring component."""
    # Create component with default weight
    component = NormalScoring()
    assert component.weight == 0.8
    
    # Test with no normals in context
    score = component.compute_score({})
    assert score == 0.0
    
    # Test with only source normal
    score = component.compute_score({'normal_src': np.array([1, 0, 0])})
    assert score == 0.0
    
    # Test with only target normal
    score = component.compute_score({'normal_tar': np.array([1, 0, 0])})
    assert score == 0.0
    
    # Test with identical normals
    score = component.compute_score({
        'normal_src': np.array([1, 0, 0]),
        'normal_tar': np.array([1, 0, 0])
    })
    expected = 0.8 * 1.0  # weight * max score
    assert np.isclose(score, expected)
    
    # Test with opposite normals
    score = component.compute_score({
        'normal_src': np.array([1, 0, 0]),
        'normal_tar': np.array([-1, 0, 0])
    })
    expected = 0.8 * 0.0  # weight * min score
    assert np.isclose(score, expected)
    
    # Test with perpendicular normals
    score = component.compute_score({
        'normal_src': np.array([1, 0, 0]),
        'normal_tar': np.array([0, 1, 0])
    })
    expected = 0.8 * 0.5  # weight * middle score
    assert np.isclose(score, expected)
    
    # Test with unnormalized vectors
    score = component.compute_score({
        'normal_src': np.array([2, 0, 0]),
        'normal_tar': np.array([0, 3, 0])
    })
    expected = 0.8 * 0.5  # weight * middle score (after normalization)
    assert np.isclose(score, expected)
    
    # Test with zero vector
    score = component.compute_score({
        'normal_src': np.array([0, 0, 0]),
        'normal_tar': np.array([1, 0, 0])
    })
    assert score == 0.0


def test_weight_vector_scoring():
    """Test WeightVectorScoring component."""
    # Create component with default weight
    component = WeightVectorScoring()
    assert component.weight == 0.7
    
    # Test with no weight vectors in context
    score = component.compute_score({})
    assert score == 0.0
    
    # Test with only source weights
    score = component.compute_score({'weights_src': np.array([0.5, 0.5, 0.0])})
    assert score == 0.0
    
    # Test with only target weights
    score = component.compute_score({'weights_tar': np.array([0.5, 0.5, 0.0])})
    assert score == 0.0
    
    # Test with different length weight vectors
    score = component.compute_score({
        'weights_src': np.array([0.5, 0.5]),
        'weights_tar': np.array([0.5, 0.5, 0.0])
    })
    assert score == 0.0
    
    # Test with identical weight vectors
    score = component.compute_score({
        'weights_src': np.array([0.5, 0.5, 0.0]),
        'weights_tar': np.array([0.5, 0.5, 0.0])
    })
    expected = 0.7 * 1.0  # weight * max score
    assert np.isclose(score, expected)
    
    # Test with orthogonal weight vectors
    score = component.compute_score({
        'weights_src': np.array([1.0, 0.0, 0.0]),
        'weights_tar': np.array([0.0, 1.0, 0.0])
    })
    expected = 0.7 * 0.0  # weight * min score (orthogonal)
    assert np.isclose(score, expected)
    
    # Test with negative similarity (should be clamped to 0)
    score = component.compute_score({
        'weights_src': np.array([1.0, 0.0, 0.0]),
        'weights_tar': np.array([-1.0, 0.0, 0.0])
    })
    expected = 0.7 * 0.0  # weight * min score (clamped negative)
    assert np.isclose(score, expected)
    
    # Test with zero vector
    score = component.compute_score({
        'weights_src': np.array([0.0, 0.0, 0.0]),
        'weights_tar': np.array([1.0, 0.0, 0.0])
    })
    assert score == 0.0


def test_laplacian_scoring():
    """Test LaplacianScoring component."""
    # Create component with default weight
    component = LaplacianScoring()
    assert component.weight == 0.6
    
    # Test with no Laplacian coordinates in context
    score = component.compute_score({})
    assert score == 0.0
    
    # Test with only source Laplacian
    score = component.compute_score({'laplacian_src': np.array([0.1, 0.2, 0.3])})
    assert score == 0.0
    
    # Test with only target Laplacian
    score = component.compute_score({'laplacian_tar': np.array([0.1, 0.2, 0.3])})
    assert score == 0.0
    
    # Test with identical Laplacian coordinates
    score = component.compute_score({
        'laplacian_src': np.array([0.1, 0.2, 0.3]),
        'laplacian_tar': np.array([0.1, 0.2, 0.3])
    })
    expected = 0.6 * 1.0  # weight * max score
    assert np.isclose(score, expected)
    
    # Test with opposite Laplacian coordinates
    score = component.compute_score({
        'laplacian_src': np.array([0.1, 0.2, 0.3]),
        'laplacian_tar': np.array([-0.1, -0.2, -0.3])
    })
    expected = 0.6 * 0.0  # weight * min score
    assert np.isclose(score, expected)
    
    # Test with perpendicular Laplacian coordinates
    score = component.compute_score({
        'laplacian_src': np.array([1, 0, 0]),
        'laplacian_tar': np.array([0, 1, 0])
    })
    expected = 0.6 * 0.5  # weight * middle score
    assert np.isclose(score, expected)
    
    # Test with zero vector
    score = component.compute_score({
        'laplacian_src': np.array([0, 0, 0]),
        'laplacian_tar': np.array([1, 0, 0])
    })
    assert score == 0.0


def test_create_default_scoring_components():
    """Test creating default scoring components."""
    components = create_default_scoring_components()
    
    # Should be a list of 5 components with the correct types and weights
    assert len(components) == 5
    
    # Check each component type and weight
    assert isinstance(components[0], DistanceScoring)
    assert components[0].weight == 1.0
    
    assert isinstance(components[1], RayQualityScoring)
    assert components[1].weight == 0.5
    
    assert isinstance(components[2], NormalScoring)
    assert components[2].weight == 0.8
    
    assert isinstance(components[3], WeightVectorScoring)
    assert components[3].weight == 0.7
    
    assert isinstance(components[4], LaplacianScoring)
    assert components[4].weight == 0.6