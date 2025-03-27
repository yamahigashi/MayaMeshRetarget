import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from maya.api import OpenMaya as om

from ymt_mesh_retarget.objects.mesh import MeshObject


@pytest.fixture
def mock_mesh_fn():
    """Create a mock mesh function set."""
    mesh_fn = MagicMock(spec=om.MFnMesh)
    
    # Mock vertex count
    mesh_fn.numVertices = 8
    
    # Mock point positions
    mesh_fn.getPoints.return_value = [
        om.MPoint(0, 0, 0),
        om.MPoint(1, 0, 0),
        om.MPoint(1, 1, 0),
        om.MPoint(0, 1, 0),
        om.MPoint(0, 0, 1),
        om.MPoint(1, 0, 1),
        om.MPoint(1, 1, 1),
        om.MPoint(0, 1, 1)
    ]
    
    # Mock normals
    mesh_fn.getVertexNormals.return_value = [
        om.MVector(0, 0, 1),
        om.MVector(1, 0, 0),
        om.MVector(0, 1, 0),
        om.MVector(0, 0, 1),
        om.MVector(0, 0, 1),
        om.MVector(1, 0, 0),
        om.MVector(0, 1, 0),
        om.MVector(0, 0, 1)
    ]
    
    return mesh_fn


@pytest.fixture
def mock_dag_path(mock_mesh_fn):
    """Create a mock DAG path."""
    dag_path = MagicMock(spec=om.MDagPath)
    dag_path.fullPathName.return_value = "|top|group|mesh1"
    
    # Patch the get_mesh_fn function to return our mock
    with patch('ymt_mesh_retarget.objects.mesh.get_mesh_fn', return_value=mock_mesh_fn):
        yield dag_path


@patch('ymt_mesh_retarget.objects.mesh.get_mesh_dag')
def test_mesh_object_init_with_string(mock_get_mesh_dag, mock_dag_path, mock_mesh_fn):
    """Test initializing MeshObject with a string path."""
    # Set up the mock to return our dag path
    mock_get_mesh_dag.return_value = mock_dag_path
    
    # Initialize with string
    mesh_obj = MeshObject("mesh1")
    
    # Verify properties
    assert mesh_obj.name == "|top|group|mesh1"
    assert mesh_obj.dag_path == mock_dag_path
    assert mesh_obj.mesh_fn == mock_mesh_fn


def test_mesh_object_init_with_dag_path(mock_dag_path, mock_mesh_fn):
    """Test initializing MeshObject with a DAG path."""
    # Initialize with DAG path
    mesh_obj = MeshObject(mock_dag_path)
    
    # Verify properties
    assert mesh_obj.name == "|top|group|mesh1"
    assert mesh_obj.dag_path == mock_dag_path
    assert mesh_obj.mesh_fn == mock_mesh_fn


@patch('ymt_mesh_retarget.objects.mesh.get_mesh_dag')
def test_mesh_object_invalid_path(mock_get_mesh_dag):
    """Test initializing MeshObject with an invalid path."""
    # Set up the mock to return None (invalid path)
    mock_get_mesh_dag.return_value = None
    
    # Verify exception
    with pytest.raises(ValueError):
        MeshObject("invalid_mesh")


def test_get_points(mock_dag_path, mock_mesh_fn):
    """Test getting mesh points."""
    mesh_obj = MeshObject(mock_dag_path)
    
    # Get points with default stride
    points = mesh_obj.get_points()
    
    # Verify points
    assert isinstance(points, np.ndarray)
    assert points.shape == (8, 3)  # 8 vertices, 3 coordinates
    
    # Get points with custom stride
    points = mesh_obj.get_points(sampling_stride=2)
    
    # Verify points are sampled
    assert isinstance(points, np.ndarray)
    assert points.shape == (4, 3)  # 4 vertices (stride 2), 3 coordinates


@patch('ymt_mesh_retarget.objects.mesh.cmds.listRelatives')
def test_get_transforms(mock_list_relatives, mock_dag_path):
    """Test getting mesh transforms."""
    # Set up mock for listRelatives
    mock_list_relatives.return_value = ["mesh1"]
    
    # Set up mocks for xform queries
    with patch('ymt_mesh_retarget.objects.mesh.cmds.xform') as mock_xform:
        mock_xform.side_effect = [
            [0, 1, 2],  # translate
            [10, 20, 30],  # rotate
            [2, 2, 2]  # scale
        ]
        
        mesh_obj = MeshObject(mock_dag_path)
        transforms = mesh_obj.get_transforms()
        
        # Verify transforms
        assert isinstance(transforms, list)
        assert len(transforms) == 1
        assert transforms[0]["translate"] == [0, 1, 2]
        assert transforms[0]["rotate"] == [10, 20, 30]
        assert transforms[0]["scale"] == [2, 2, 2]


@patch('ymt_mesh_retarget.objects.mesh.cmds.duplicate')
def test_duplicate(mock_duplicate, mock_dag_path):
    """Test duplicating a mesh object."""
    # Set up mock for duplicate
    mock_duplicate.return_value = ["mesh1_retarget"]
    
    # Set up mock for the new mesh object
    with patch('ymt_mesh_retarget.objects.mesh.MeshObject.__init__') as mock_init:
        mock_init.return_value = None
        
        mesh_obj = MeshObject(mock_dag_path)
        duplicate = mesh_obj.duplicate("_test")
        
        # Verify duplicate was created
        mock_duplicate.assert_called_once()
        assert isinstance(duplicate, MeshObject)


@patch('ymt_mesh_retarget.objects.mesh.convert_points_to_numpy')
@patch('ymt_mesh_retarget.objects.mesh.set_points')
def test_apply_transforms(mock_set_points, mock_convert, mock_dag_path):
    """Test applying transforms to a mesh object."""
    # Set up mock for convert_points_to_numpy
    points = np.array([
        [0, 0, 0],
        [1, 0, 0]
    ])
    mock_convert.return_value = points
    
    mesh_obj = MeshObject(mock_dag_path)
    
    # Apply transforms
    transform_data = [
        {"translate": [10, 0, 0], "rotate": [0, 0, 0], "scale": [1, 1, 1]}
    ]
    
    mesh_obj.apply_transforms(transform_data)
    
    # Verify points were set
    mock_set_points.assert_called_once()


def test_calculate_threshold_distance(mock_dag_path, mock_mesh_fn):
    """Test calculating threshold distance."""
    mesh_obj = MeshObject(mock_dag_path)
    
    # Calculate threshold with coefficient
    threshold = mesh_obj.calculate_threshold_distance(2.0)
    
    # Verify threshold (should use bounding box size)
    assert threshold > 0


@patch('ymt_mesh_retarget.objects.mesh.get_skin_cluster')
def test_get_vertex_weight_vector(mock_get_skin_cluster, mock_dag_path):
    """Test getting vertex weight vector."""
    # Mock skin cluster
    mock_skin = MagicMock()
    mock_skin.getWeights.return_value = [0.7, 0.3, 0.0, 0.0]
    mock_get_skin_cluster.return_value = mock_skin
    
    mesh_obj = MeshObject(mock_dag_path)
    
    # Force recalculation by clearing cache
    mesh_obj._is_weights_cached = False
    mesh_obj._weights_cache = {}
    
    # Get weights
    weights = mesh_obj.get_vertex_weight_vector(0)
    
    # Verify weights
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 4
    assert weights[0] == 0.7
    assert weights[1] == 0.3


@patch('ymt_mesh_retarget.objects.mesh.MeshObject.get_vertex_normal')
def test_precompute_vertex_normals(mock_get_normal, mock_dag_path, mock_mesh_fn):
    """Test precomputing vertex normals."""
    # Mock get_vertex_normal
    mock_get_normal.side_effect = [
        np.array([0, 0, 1]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, 1]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ]
    
    mesh_obj = MeshObject(mock_dag_path)
    
    # Force recalculation
    mesh_obj._is_normals_cached = False
    mesh_obj.precompute_vertex_normals()
    
    # Verify cache
    assert mesh_obj._is_normals_cached
    assert len(mesh_obj._normals_cache) == 8
    
    # Cached normals should be used now
    normal = mesh_obj.get_vertex_normal(0)
    assert np.array_equal(normal, mesh_obj._normals_cache[0])


@patch('ymt_mesh_retarget.objects.mesh.MeshObject.compute_laplacian_for_vertex')
def test_precompute_laplacians(mock_compute_lap, mock_dag_path):
    """Test precomputing Laplacian coordinates."""
    # Mock compute_laplacian_for_vertex
    mock_compute_lap.side_effect = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    
    mesh_obj = MeshObject(mock_dag_path)
    
    # Force recalculation
    mesh_obj._is_laplacians_cached = False
    mesh_obj.precompute_laplacians()
    
    # Verify cache
    assert mesh_obj._is_laplacians_cached
    assert len(mesh_obj._laplacians_cache) == 8
    
    # Cached Laplacians should be used now
    laplacian = mesh_obj.compute_laplacian_for_vertex(0)
    assert np.array_equal(laplacian, mesh_obj._laplacians_cache[0])


@patch('ymt_mesh_retarget.objects.mesh.cmds.listRelatives')
def test_get_children(mock_list_relatives, mock_dag_path):
    """Test getting children of a mesh object."""
    # Mesh objects typically don't have children in the sense of RetargetableObject
    mock_list_relatives.return_value = None
    
    mesh_obj = MeshObject(mock_dag_path)
    children = mesh_obj.get_children()
    
    # Verify children (should be empty)
    assert isinstance(children, list)
    assert len(children) == 0


@patch('ymt_mesh_retarget.objects.mesh.get_mesh_dag')
def test_create_from_path(mock_get_mesh_dag, mock_dag_path):
    """Test creating a mesh object from path."""
    # Set up the mock to return our dag path
    mock_get_mesh_dag.return_value = mock_dag_path
    
    # Use the factory method
    with patch('ymt_mesh_retarget.objects.mesh.MeshObject.__init__') as mock_init:
        mock_init.return_value = None
        
        mesh_obj = MeshObject.create_from_path("mesh1")
        
        # Verify factory method created an instance
        assert isinstance(mesh_obj, MeshObject)
        mock_init.assert_called_once()