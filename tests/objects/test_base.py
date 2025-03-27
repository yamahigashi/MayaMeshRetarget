import pytest
from unittest.mock import MagicMock

import numpy as np
from maya.api import OpenMaya as om

from ymt_mesh_retarget.objects.base import RetargetableObject


class ConcreteRetargetableObject(RetargetableObject):
    """Concrete implementation of RetargetableObject for testing."""
    
    def __init__(self, name="test_object"):
        self.name = name
    
    def get_points(self, sampling_stride=1):
        return np.array([[0, 0, 0], [1, 1, 1]])
    
    def get_transforms(self):
        return [{"translate": [0, 0, 0], "rotate": [0, 0, 0], "scale": [1, 1, 1]}]
    
    def duplicate(self, suffix="_retarget"):
        return ConcreteRetargetableObject(f"{self.name}{suffix}")
    
    def apply_transforms(self, transform_data):
        pass
    
    def calculate_threshold_distance(self, coefficient):
        return 1.0 * coefficient
    
    def get_children(self, type_filter=None):
        return []
    
    @staticmethod
    def create_from_path(path):
        if isinstance(path, str):
            return ConcreteRetargetableObject(path)
        elif isinstance(path, om.MDagPath):
            return ConcreteRetargetableObject(path.fullPathName())
        else:
            raise TypeError("Path must be a string or MDagPath")


def test_abstract_base_class():
    """Test that RetargetableObject is an abstract base class and can't be instantiated directly."""
    with pytest.raises(TypeError):
        RetargetableObject()


def test_concrete_implementation():
    """Test a concrete implementation of RetargetableObject."""
    obj = ConcreteRetargetableObject("test_object")
    
    # Test basic properties
    assert obj.name == "test_object"
    
    # Test methods
    points = obj.get_points()
    assert isinstance(points, np.ndarray)
    assert points.shape == (2, 3)
    
    transforms = obj.get_transforms()
    assert isinstance(transforms, list)
    assert len(transforms) == 1
    assert "translate" in transforms[0]
    
    # Test duplicate
    duplicate = obj.duplicate("_copy")
    assert duplicate.name == "test_object_copy"
    assert isinstance(duplicate, ConcreteRetargetableObject)
    
    # Test threshold calculation
    threshold = obj.calculate_threshold_distance(2.5)
    assert threshold == 2.5
    
    # Test get_children
    children = obj.get_children()
    assert isinstance(children, list)
    assert len(children) == 0


def test_create_from_path():
    """Test the create_from_path factory method."""
    # Test with string path
    obj = ConcreteRetargetableObject.create_from_path("path/to/object")
    assert obj.name == "path/to/object"
    
    # Test with MDagPath
    mock_dag_path = MagicMock(spec=om.MDagPath)
    mock_dag_path.fullPathName.return_value = "dag/path/to/object"
    
    obj = ConcreteRetargetableObject.create_from_path(mock_dag_path)
    assert obj.name == "dag/path/to/object"
    
    # Test with invalid type
    with pytest.raises(TypeError):
        ConcreteRetargetableObject.create_from_path(123)