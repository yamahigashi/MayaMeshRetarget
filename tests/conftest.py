import os
import pathlib
import sys

import pytest  # type: ignore

# Maya standalone mode initialization
try:
    from maya import (
        cmds,
        standalone,
    )
    
    # Setup Python paths
    directory = os.path.dirname(os.path.abspath(__file__))
    pypath = pathlib.Path(directory).parent / "python"
    sys.path.append(str(pypath))
    pypath = pathlib.Path(directory).parent / "vendor_python"
    sys.path.append(str(pypath))
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
    pass

# Import all fixtures from fixtures module
from fixtures import *