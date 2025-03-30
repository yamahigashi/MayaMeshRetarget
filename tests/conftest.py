import os
import pathlib
import sys
import unittest.mock as mock

import pytest  # type: ignore


# Setup Python paths
directory = os.path.dirname(os.path.abspath(__file__))
pypath = pathlib.Path(directory).parent / "python"
sys.path.append(str(pypath))
pypath = pathlib.Path(directory).parent / "vendor_python"
sys.path.append(str(pypath))

# Maya standalone mode initialization
try:
    from maya import (
        cmds,
        standalone,
    )
    
    print(sys.path)
    
    # Initialize Maya
    standalone.initialize(name="python")
    
    # Auto fixture to prepare a clean Maya environment before each test
    @pytest.fixture(autouse=True)
    def setup():
        """Setup and teardown executed before and after each test function"""
        print("Setup before test function")
        cmds.file(newFile=True, force=True)
        yield
        print("Teardown after test function")
    
except ImportError:
    # When Maya is not available, use mocks or skip tests
    print("Failed to import Maya modules. Using mocks instead.")
    
    # Create mock versions of Maya modules
    class MockMayaModules:
        def __init__(self):
            self.cmds = mock.MagicMock()
            self.om = mock.MagicMock()
            self.OpenMaya = self.om
            
            # Set up commonly used return values
            self.cmds.nodeType.return_value = "transform"
            self.cmds.listRelatives.return_value = ["shape1"]
            
    # Create mock for maya.cmds and maya.api.OpenMaya
    maya_mock = MockMayaModules()
    sys.modules['maya'] = mock.MagicMock()
    sys.modules['maya.cmds'] = maya_mock.cmds
    sys.modules['maya.api'] = mock.MagicMock()
    sys.modules['maya.api.OpenMaya'] = maya_mock.om

    @pytest.fixture(autouse=True)
    def setup():
        """Setup and teardown executed before and after each test function"""
        print("Setup before test function in mock mode")
        yield
        print("Teardown after test function in mock mode")

# Create test utility mocks
@pytest.fixture
def mock_maya_utils(monkeypatch):
    """Fixture to mock key Maya utility functions used in tests."""
    from ymt_mesh_retarget import util
    
    # Mock get_mesh_dag
    def mock_get_mesh_dag(name):
        from maya.api import OpenMaya as om
        dag_path = mock.MagicMock(spec=om.MDagPath)
        dag_path.fullPathName.return_value = name if isinstance(name, str) else "|mock|path"
        return dag_path
    
    monkeypatch.setattr(util, 'get_mesh_dag', mock_get_mesh_dag)
    
    # Mock get_mesh_fn
    def mock_get_mesh_fn(dag_path):
        from maya.api import OpenMaya as om
        mesh_fn = mock.MagicMock(spec=om.MFnMesh)
        mesh_fn.numVertices = 8
        return mesh_fn
    
    monkeypatch.setattr(util, 'get_mesh_fn', mock_get_mesh_fn)
    
    return {"get_mesh_dag": mock_get_mesh_dag, "get_mesh_fn": mock_get_mesh_fn}

# Import all fixtures from fixtures module
try:
    from fixtures import *
except ImportError:
    print("Failed to import fixtures module")