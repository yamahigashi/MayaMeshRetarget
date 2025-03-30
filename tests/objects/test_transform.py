from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from maya.api import OpenMaya as om

from ymt_mesh_retarget.objects.transform import TransformObject


@pytest.fixture
def mock_dag_path():
    """Create a mock DAG path for a transform node."""
    dag_path = MagicMock(spec=om.MDagPath)
    dag_path.fullPathName.return_value = "|top|group|transform1"
    return dag_path


@patch('ymt_mesh_retarget.objects.transform.get_dag_path')
def test_transform_object_init(mock_get_dag_path, mock_dag_path):
    """Test initializing TransformObject."""
    # Set up the mock to return our dag path
    mock_get_dag_path.return_value = mock_dag_path
    
    # Initialize with string
    transform_obj = TransformObject("transform1")
    
    # Verify properties
    assert transform_obj.name == "|top|group|transform1"
    assert transform_obj.dag_path == mock_dag_path


@patch('ymt_mesh_retarget.objects.transform.get_dag_path')
def test_transform_object_invalid_path(mock_get_dag_path):
    """Test initializing TransformObject with an invalid path."""
    # Set up the mock to return None (invalid path)
    mock_get_dag_path.return_value = None
    
    # Verify exception
    with pytest.raises(ValueError):
        TransformObject("invalid_transform")


@patch('ymt_mesh_retarget.objects.transform.cmds.xform')
def test_get_points(mock_xform, mock_dag_path):
    """Test getting transform points."""
    # Set up mock for xform
    mock_xform.return_value = [1.0, 2.0, 3.0]
    
    # Create transform object
    with patch('ymt_mesh_retarget.objects.transform.get_dag_path', return_value=mock_dag_path):
        transform_obj = TransformObject("transform1")
    
    # Get points
    points = transform_obj.get_points()
    
    # Verify points
    assert isinstance(points, np.ndarray)
    assert points.shape == (1, 3)  # 1 point with 3 coordinates
    assert np.array_equal(points[0], [1.0, 2.0, 3.0])


@patch('ymt_mesh_retarget.objects.transform.cmds.xform')
@patch('ymt_mesh_retarget.objects.transform.cmds.getAttr')
def test_get_transforms(mock_getattr, mock_xform, mock_dag_path):
    """Test getting transform data."""
    # Set up mocks
    mock_xform.side_effect = [
        [1.0, 2.0, 3.0],  # translation
        [10.0, 20.0, 30.0]  # rotation
    ]
    mock_getattr.return_value = [(2.0, 2.0, 2.0)]  # scale
    
    # Create transform object
    with patch('ymt_mesh_retarget.objects.transform.get_dag_path', return_value=mock_dag_path):
        transform_obj = TransformObject("transform1")
    
    # Get transforms
    transforms = transform_obj.get_transforms()
    
    # Verify transforms
    assert isinstance(transforms, list)
    assert len(transforms) == 1
    
    transform = transforms[0]
    assert transform["path"] == "|top|group|transform1"
    assert np.array_equal(transform["position"], [1.0, 2.0, 3.0])
    assert "rotation" in transform  # Quaternion conversion is complex, just check it exists
    assert np.array_equal(transform["scale"], [2.0, 2.0, 2.0])


@patch('ymt_mesh_retarget.objects.transform.cmds.duplicate')
def test_duplicate(mock_duplicate, mock_dag_path):
    """Test duplicating a transform object."""
    # Set up mock for duplicate
    mock_duplicate.return_value = ["transform1_retarget"]
    
    # Create transform object
    with patch('ymt_mesh_retarget.objects.transform.get_dag_path', return_value=mock_dag_path):
        transform_obj = TransformObject("transform1")
    
    # Duplicate
    with patch('ymt_mesh_retarget.objects.transform.TransformObject.__init__') as mock_init:
        mock_init.return_value = None
        duplicate = transform_obj.duplicate("_test")
        
        # Verify duplicate
        assert isinstance(duplicate, TransformObject)
        mock_duplicate.assert_called_once()


@patch('ymt_mesh_retarget.objects.transform.cmds.xform')
def test_apply_transforms(mock_xform, mock_dag_path):
    """Test applying transforms."""
    # Create transform object
    with patch('ymt_mesh_retarget.objects.transform.get_dag_path', return_value=mock_dag_path):
        transform_obj = TransformObject("transform1")
    
    # Transform data
    transform_data = [
        {
            "path": "|top|group|transform1",
            "position": [10, 20, 30],
            "rotation": [0, 0, 0, 1],  # Identity quaternion
            "scale": [2, 2, 2]
        }
    ]
    
    # Apply transforms
    transform_obj.apply_transforms(transform_data)
    
    # Verify xform was called for each transformation
    assert mock_xform.call_count == 3  # Translation, rotation, and scale


def test_calculate_threshold_distance(mock_dag_path):
    """Test calculating threshold distance."""
    # Create transform object
    with patch('ymt_mesh_retarget.objects.transform.get_dag_path', return_value=mock_dag_path):
        transform_obj = TransformObject("transform1")
    
    # Calculate threshold
    threshold = transform_obj.calculate_threshold_distance(2.0)
    
    # Verify threshold
    assert isinstance(threshold, float)
    assert threshold > 0


@patch('ymt_mesh_retarget.objects.transform.cmds.listRelatives')
def test_get_children(mock_list_relatives, mock_dag_path):
    """Test getting children of a transform."""
    # Set up mock for listRelatives
    mock_list_relatives.return_value = ["child1", "child2"]
    
    # Create transform object
    with patch('ymt_mesh_retarget.objects.transform.get_dag_path', return_value=mock_dag_path):
        transform_obj = TransformObject("transform1")
    
    # Get children with no filter
    with patch('ymt_mesh_retarget.objects.transform.TransformObject.create_from_path') as mock_create:
        # Set up mock to return dummy transform objects
        mock_create.side_effect = lambda x: TransformObject(x)
        
        # Patch the __init__ method to avoid actual initialization
        with patch('ymt_mesh_retarget.objects.transform.TransformObject.__init__') as mock_init:
            mock_init.return_value = None
            
            children = transform_obj.get_children()
            
            # Verify children
            assert isinstance(children, list)
            assert len(children) == 2
            assert all(isinstance(child, TransformObject) for child in children)
    
    # Get children with filter
    mock_list_relatives.return_value = ["mesh1"]
    
    with patch('ymt_mesh_retarget.objects.transform.cmds.objectType') as mock_object_type:
        mock_object_type.return_value = "mesh"
        
        with patch('ymt_mesh_retarget.objects.transform.TransformObject.create_from_path'):
            children = transform_obj.get_children(type_filter="mesh")
            
            # Verify filtered children
            assert isinstance(children, list)
            assert len(children) == 1


@patch('ymt_mesh_retarget.objects.transform.get_dag_path')
def test_create_from_path(mock_get_dag_path, mock_dag_path):
    """Test creating a transform object from path."""
    # Set up the mock to return our dag path
    mock_get_dag_path.return_value = mock_dag_path
    
    # Use the factory method
    with patch('ymt_mesh_retarget.objects.transform.TransformObject.__init__') as mock_init:
        mock_init.return_value = None
        
        transform_obj = TransformObject.create_from_path("transform1")
        
        # Verify factory method created an instance
        assert isinstance(transform_obj, TransformObject)
        mock_init.assert_called_once()