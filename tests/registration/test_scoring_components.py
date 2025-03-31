from typing import Any

import numpy as np
import pytest

from ymt_mesh_retarget.registration.scoring_components import (
    CurvatureScoring,
    DistanceScoring,
    HeatKernelSignatureScoring,
    IScoringComponent,
    LaplacianScoring,
    NormalScoring,
    RayQualityScoring,
    SemanticLabelScoring,
    TwoStageScoring,
    UVRegionScoring,
    WeightVectorScoring,
    compute_average_score,
    create_advanced_scoring_components,
    create_default_scoring_components,
    create_optimized_scoring_pipeline,
)


class SimpleScoringComponent(IScoringComponent):
    """Simple implementation of IScoringComponent for testing."""

    def __init__(self, weight: float = 1.0, return_value: float = 0.5) -> None:
        self.weight_value = weight
        self.return_value = return_value

    def compute_score(self, context: dict[str, Any]) -> float:
        return self.return_value

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
        def __init__(self, weight: float, relate_distance: float) -> None:
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
        'normal_tar': np.array([1, 0, 0]),
    })
    expected = 0.8 * 1.0  # weight * max score
    assert np.isclose(score, expected)

    # Test with opposite normals
    score = component.compute_score({
        'normal_src': np.array([1, 0, 0]),
        'normal_tar': np.array([-1, 0, 0]),
    })
    expected = 0.8 * 0.0  # weight * min score
    assert np.isclose(score, expected)

    # Test with perpendicular normals
    score = component.compute_score({
        'normal_src': np.array([1, 0, 0]),
        'normal_tar': np.array([0, 1, 0]),
    })
    expected = 0.8 * 0.5  # weight * middle score
    assert np.isclose(score, expected)

    # Test with unnormalized vectors
    score = component.compute_score({
        'normal_src': np.array([2, 0, 0]),
        'normal_tar': np.array([0, 3, 0]),
    })
    expected = 0.8 * 0.5  # weight * middle score (after normalization)
    assert np.isclose(score, expected)

    # Test with zero vector
    score = component.compute_score({
        'normal_src': np.array([0, 0, 0]),
        'normal_tar': np.array([1, 0, 0]),
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
        'weights_tar': np.array([0.5, 0.5, 0.0]),
    })
    assert score == 0.0

    # Test with identical weight vectors
    score = component.compute_score({
        'weights_src': np.array([0.5, 0.5, 0.0]),
        'weights_tar': np.array([0.5, 0.5, 0.0]),
    })
    expected = 0.7 * 1.0  # weight * max score
    assert np.isclose(score, expected)

    # Test with orthogonal weight vectors
    score = component.compute_score({
        'weights_src': np.array([1.0, 0.0, 0.0]),
        'weights_tar': np.array([0.0, 1.0, 0.0]),
    })
    expected = 0.7 * 0.0  # weight * min score (orthogonal)
    assert np.isclose(score, expected)

    # Test with negative similarity (should be clamped to 0)
    score = component.compute_score({
        'weights_src': np.array([1.0, 0.0, 0.0]),
        'weights_tar': np.array([-1.0, 0.0, 0.0]),
    })
    expected = 0.7 * 0.0  # weight * min score (clamped negative)
    assert np.isclose(score, expected)

    # Test with zero vector
    score = component.compute_score({
        'weights_src': np.array([0.0, 0.0, 0.0]),
        'weights_tar': np.array([1.0, 0.0, 0.0]),
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
        'laplacian_tar': np.array([0.1, 0.2, 0.3]),
    })
    expected = 0.6 * 1.0  # weight * max score
    assert np.isclose(score, expected)

    # Test with opposite Laplacian coordinates
    score = component.compute_score({
        'laplacian_src': np.array([0.1, 0.2, 0.3]),
        'laplacian_tar': np.array([-0.1, -0.2, -0.3]),
    })
    expected = 0.6 * 0.0  # weight * min score
    assert np.isclose(score, expected)

    # Test with perpendicular Laplacian coordinates
    score = component.compute_score({
        'laplacian_src': np.array([1, 0, 0]),
        'laplacian_tar': np.array([0, 1, 0]),
    })
    expected = 0.6 * 0.5  # weight * middle score
    assert np.isclose(score, expected)

    # Test with zero vector
    score = component.compute_score({
        'laplacian_src': np.array([0, 0, 0]),
        'laplacian_tar': np.array([1, 0, 0]),
    })
    assert score == 0.0


def test_curvature_scoring():
    """Test CurvatureScoring component."""
    # Create component with default weight
    component = CurvatureScoring()
    assert component.weight == 0.6

    # Test with no curvature values in context
    score = component.compute_score({})
    assert score == 0.0

    # Test with only source curvature
    score = component.compute_score({'curvature_src': 0.5})
    assert score == 0.0

    # Test with only target curvature
    score = component.compute_score({'curvature_tar': 0.5})
    assert score == 0.0

    # Test with identical curvature values
    score = component.compute_score({
        'curvature_src': 0.5,
        'curvature_tar': 0.5,
    })
    expected = 0.6 * 1.0  # weight * max score (exact match)
    assert np.isclose(score, expected)

    # Test with different curvature values (within range)
    score = component.compute_score({
        'curvature_src': 0.3,
        'curvature_tar': 0.8,
    })
    expected = 0.6 * 0.5  # weight * score (difference of 0.5 / max_diff of 1.0)
    assert np.isclose(score, expected)

    # Test with different curvature values (beyond range)
    score = component.compute_score({
        'curvature_src': 0.0,
        'curvature_tar': 2.0,
    })
    expected = 0.6 * 0.0  # weight * min score (difference exceeds max_diff)
    assert np.isclose(score, expected)


def test_hks_scoring():
    """Test HeatKernelSignatureScoring component."""
    # Create component with default weight
    component = HeatKernelSignatureScoring()
    assert component.weight == 0.7

    # Test with no HKS values in context
    score = component.compute_score({})
    assert score == 0.0

    # Test with only source HKS
    score = component.compute_score({'hks_src': np.array([0.1, 0.2, 0.3, 0.2, 0.1])})
    assert score == 0.0

    # Test with only target HKS
    score = component.compute_score({'hks_tar': np.array([0.1, 0.2, 0.3, 0.2, 0.1])})
    assert score == 0.0

    # Test with identical HKS values
    score = component.compute_score({
        'hks_src': np.array([0.1, 0.2, 0.3, 0.2, 0.1]),
        'hks_tar': np.array([0.1, 0.2, 0.3, 0.2, 0.1]),
    })
    expected = 0.7 * 1.0  # weight * max score
    assert np.isclose(score, expected)

    # Test with opposite HKS values
    score = component.compute_score({
        'hks_src': np.array([0.1, 0.2, 0.3, 0.2, 0.1]),
        'hks_tar': np.array([-0.1, -0.2, -0.3, -0.2, -0.1]),
    })
    expected = 0.7 * 0.0  # weight * min score
    assert np.isclose(score, expected)

    # Test with somewhat similar HKS values
    score = component.compute_score({
        'hks_src': np.array([0.1, 0.2, 0.3, 0.2, 0.1]),
        'hks_tar': np.array([0.2, 0.3, 0.2, 0.1, 0.1]),
    })
    # Expected: weight * score based on cosine similarity (will be between 0 and 1)
    assert 0.0 < score < 0.7

    # Test with lists instead of arrays
    score = component.compute_score({
        'hks_src': [0.1, 0.2, 0.3, 0.2, 0.1],
        'hks_tar': [0.1, 0.2, 0.3, 0.2, 0.1],
    })
    expected = 0.7 * 1.0  # weight * max score
    assert np.isclose(score, expected)

    # Test with zero vector
    score = component.compute_score({
        'hks_src': np.array([0, 0, 0, 0, 0]),
        'hks_tar': np.array([0.1, 0.2, 0.3, 0.2, 0.1]),
    })
    assert score == 0.0


def test_semantic_label_scoring():
    """Test SemanticLabelScoring component."""
    # Create component with default weight
    component = SemanticLabelScoring()
    assert component.weight == 0.9

    # Test with no semantic labels in context
    score = component.compute_score({})
    assert score == 0.0

    # Test with only source label
    score = component.compute_score({'semantic_label_src': 'face'})
    assert score == 0.0

    # Test with only target label
    score = component.compute_score({'semantic_label_tar': 'face'})
    assert score == 0.0

    # Test with identical string labels
    score = component.compute_score({
        'semantic_label_src': 'face',
        'semantic_label_tar': 'face',
    })
    expected = 0.9 * 1.0  # weight * max score (exact match)
    assert np.isclose(score, expected)

    # Test with different string labels
    score = component.compute_score({
        'semantic_label_src': 'face',
        'semantic_label_tar': 'hand',
    })
    expected = 0.0  # No match
    assert np.isclose(score, expected)

    # Test with identical vector labels
    score = component.compute_score({
        'semantic_label_src': np.array([1, 0, 0, 0]),  # One-hot encoding
        'semantic_label_tar': np.array([1, 0, 0, 0]),
    })
    expected = 0.9 * 1.0  # weight * max score (exact match)
    assert np.isclose(score, expected)

    # Test with orthogonal vector labels
    score = component.compute_score({
        'semantic_label_src': np.array([1, 0, 0, 0]),  # One-hot encoding
        'semantic_label_tar': np.array([0, 1, 0, 0]),
    })
    expected = 0.0  # No similarity (orthogonal vectors)
    assert np.isclose(score, expected)

    # Test with similar vector labels
    score = component.compute_score({
        'semantic_label_src': np.array([0.8, 0.2, 0.0, 0.0]),  # Fuzzy encoding
        'semantic_label_tar': np.array([0.7, 0.3, 0.0, 0.0]),
    })
    # Will be the cosine similarity scaled by weight
    similarity = np.dot([0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0])
    similarity = similarity / (np.linalg.norm([0.8, 0.2, 0.0, 0.0]) * np.linalg.norm([0.7, 0.3, 0.0, 0.0]))
    expected = 0.9 * similarity
    assert np.isclose(score, expected)

    # Test with list vector labels
    score = component.compute_score({
        'semantic_label_src': [1, 0, 0, 0],
        'semantic_label_tar': [1, 0, 0, 0],
    })
    expected = 0.9 * 1.0  # weight * max score (exact match)
    assert np.isclose(score, expected)

    # Test with zero vector
    score = component.compute_score({
        'semantic_label_src': np.array([0, 0, 0, 0]),
        'semantic_label_tar': np.array([1, 0, 0, 0]),
    })
    assert score == 0.0


def test_uv_region_scoring():
    """Test UVRegionScoring component."""
    # Create component with default weight
    component = UVRegionScoring()
    assert component.weight == 0.7

    # Test with no UV data in context
    score = component.compute_score({})
    assert score == 0.0

    # Test with region IDs - identical
    score = component.compute_score({
        'uv_region_id_src': 1,
        'uv_region_id_tar': 1,
    })
    expected = 0.7 * 1.0  # weight * max score (exact match)
    assert np.isclose(score, expected)

    # Test with region IDs - different
    score = component.compute_score({
        'uv_region_id_src': 1,
        'uv_region_id_tar': 2,
    })
    expected = 0.0  # No match
    assert np.isclose(score, expected)

    # Test with UV coordinates - identical
    score = component.compute_score({
        'uv_coord_src': np.array([0.5, 0.5]),
        'uv_coord_tar': np.array([0.5, 0.5]),
    })
    expected = 0.7 * 1.0  # weight * max score (exact match)
    assert np.isclose(score, expected)

    # Test with UV coordinates - close
    score = component.compute_score({
        'uv_coord_src': np.array([0.5, 0.5]),
        'uv_coord_tar': np.array([0.6, 0.6]),
    })
    # Distance calculations might vary slightly due to floating point precision
    # Just check that the score is in a reasonable range
    assert 0.6 < score < 0.7

    # Test with UV coordinates - far
    score = component.compute_score({
        'uv_coord_src': np.array([0.0, 0.0]),
        'uv_coord_tar': np.array([1.0, 1.0]),
    })
    # Distance = sqrt(2) = 1.414, normalized by 1.414 = 1.0
    # Score = 1.0 - 1.0 = 0.0
    expected = 0.0
    assert np.isclose(score, expected)

    # Test with UV coordinates - very far (beyond max distance)
    score = component.compute_score({
        'uv_coord_src': np.array([0.0, 0.0]),
        'uv_coord_tar': np.array([2.0, 2.0]),
    })
    expected = 0.0  # Score clamped at 0
    assert np.isclose(score, expected)

    # Test with lists instead of arrays
    score = component.compute_score({
        'uv_coord_src': [0.5, 0.5],
        'uv_coord_tar': [0.5, 0.5],
    })
    expected = 0.7 * 1.0
    assert np.isclose(score, expected)

    # Test with partial data
    score = component.compute_score({
        'uv_coord_src': np.array([0.5, 0.5]),
    })
    assert score == 0.0


def test_compute_average_score():
    """Test compute_average_score function."""
    # Create test components
    components = [
        SimpleScoringComponent(weight=1.0),  # Will return 0.5 * 1.0 = 0.5
        SimpleScoringComponent(weight=2.0),  # Will return 0.5 * 2.0 = 1.0
    ]

    # Empty components list
    score = compute_average_score([], {})
    assert score == 0.0

    # Normal case
    score = compute_average_score(components, {})
    # (0.5*1.0 + 0.5*2.0) / (1.0 + 2.0) = 1.5 / 3.0 = 0.5
    expected = 0.5
    assert np.isclose(score, expected)

    # Test with custom context (doesn't matter for SimpleScoringComponent)
    score = compute_average_score(components, {'test': 123})
    assert np.isclose(score, expected)


def test_two_stage_scoring():
    """Test TwoStageScoring component."""
    # Create component with default settings
    component = TwoStageScoring()

    # Check default components
    assert len(component.first_stage_components) == 2
    assert isinstance(component.first_stage_components[0], DistanceScoring)
    assert isinstance(component.first_stage_components[1], NormalScoring)

    assert len(component.second_stage_components) == 4
    assert isinstance(component.second_stage_components[0], LaplacianScoring)
    assert isinstance(component.second_stage_components[1], WeightVectorScoring)
    assert isinstance(component.second_stage_components[2], CurvatureScoring)
    assert isinstance(component.second_stage_components[3], HeatKernelSignatureScoring)

    # Test with empty context
    score = component.compute_score({})
    assert score == 0.0

    # Create mock context that fails first stage
    # Only distance present, with a high value (bad match)
    mock_context = {
        'distance': 10.0,  # Will give a low score
    }
    score = component.compute_score(mock_context)
    # First stage will have a low score due to high distance
    # Should not proceed to second stage
    assert score < component.first_stage_threshold

    # Create mock context that passes first stage
    # Distance and normal both perfect
    mock_context = {
        'distance': 0.1,  # Will give a high score
        'normal_src': np.array([1, 0, 0]),
        'normal_tar': np.array([1, 0, 0]),  # Perfect normal match
    }

    # First create custom component with only basic scorers
    # This makes testing easier since we know exactly what scores to expect
    first_stage = [DistanceScoring(weight=1.0)]
    second_stage = [NormalScoring(weight=1.0)]
    custom_component = TwoStageScoring(
        first_stage_components=first_stage,
        second_stage_components=second_stage,
        first_stage_threshold=0.5,
    )

    # Calculate expected first stage score
    distance_score = np.exp(-0.1)  # From DistanceScoring
    assert distance_score > 0.5  # Make sure it passes threshold

    # Calculate expected second stage score
    normal_score = 1.0  # Perfect match from NormalScoring

    # Calculate expected combined score
    expected = (distance_score + normal_score) / 2.0

    # Test the component
    score = custom_component.compute_score(mock_context)
    assert np.isclose(score, expected)

    # Test with custom implementation using simple components
    # Create components with fixed return values that don't depend on context
    low_score = SimpleScoringComponent(weight=1.0, return_value=0.1)  # Returns 0.1 directly
    high_score = SimpleScoringComponent(weight=1.0, return_value=0.8)  # Returns 0.8 directly

    # This should fail first stage
    fail_component = TwoStageScoring(
        first_stage_components=[low_score],
        second_stage_components=[high_score],
        first_stage_threshold=0.2,
    )
    score = fail_component.compute_score({})
    # First stage returns 0.1, which is below threshold of 0.2
    # So the result is just 0.1 * component weight (1.0) = 0.1
    assert 0.09 < score < 0.11  # Only first stage score with some tolerance

    # This should pass first stage
    pass_component = TwoStageScoring(
        first_stage_components=[high_score],
        second_stage_components=[high_score],
        first_stage_threshold=0.2,
    )
    score = pass_component.compute_score({})
    # First stage returns 0.8, which is above threshold
    # Second stage also returns 0.8
    # Combined: (0.8 + 0.8) / 2 = 0.8
    assert 0.79 < score < 0.81


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


def test_create_advanced_scoring_components():
    """Test creating advanced scoring components."""
    components = create_advanced_scoring_components()

    # Should be a list of 9 components with the correct types
    assert len(components) == 9

    # Check that it includes all basic components
    assert any(isinstance(comp, DistanceScoring) for comp in components)
    assert any(isinstance(comp, RayQualityScoring) for comp in components)
    assert any(isinstance(comp, NormalScoring) for comp in components)
    assert any(isinstance(comp, WeightVectorScoring) for comp in components)
    assert any(isinstance(comp, LaplacianScoring) for comp in components)

    # Check for advanced components
    assert any(isinstance(comp, CurvatureScoring) for comp in components)
    assert any(isinstance(comp, HeatKernelSignatureScoring) for comp in components)
    assert any(isinstance(comp, SemanticLabelScoring) for comp in components)
    assert any(isinstance(comp, UVRegionScoring) for comp in components)


def test_create_optimized_scoring_pipeline():
    """Test creating optimized scoring pipeline with TwoStageScoring."""
    components = create_optimized_scoring_pipeline()

    # Should be a list with a single TwoStageScoring component
    assert len(components) == 1
    assert isinstance(components[0], TwoStageScoring)

    # Check the structure of the two-stage component
    two_stage = components[0]
    assert two_stage.first_stage_threshold == 0.3

    # First stage should include distance and normal scoring
    assert len(two_stage.first_stage_components) == 2
    assert any(isinstance(comp, DistanceScoring) for comp in two_stage.first_stage_components)
    assert any(isinstance(comp, NormalScoring) for comp in two_stage.first_stage_components)

    # Second stage should include more complex metrics
    assert len(two_stage.second_stage_components) == 4
    assert any(isinstance(comp, LaplacianScoring) for comp in two_stage.second_stage_components)
    assert any(isinstance(comp, WeightVectorScoring) for comp in two_stage.second_stage_components)
    assert any(isinstance(comp, CurvatureScoring) for comp in two_stage.second_stage_components)
    assert any(isinstance(comp, HeatKernelSignatureScoring) for comp in two_stage.second_stage_components)
