import time

import numpy as np
import pytest

from ymt_mesh_retarget.util import get_short_name, timeit


def test_timeit_decorator():
    """Test the timeit decorator."""
    # Test function
    @timeit
    def dummy_function(sleep_time):
        time.sleep(sleep_time)
        return "done"
    
    # Execute with short duration
    short_result = dummy_function(0.01)
    
    # Verify correct function result
    assert short_result == "done"
    
    # Note: We can't directly test the timing output as it's 
    # logged to the console. Here we only verify the function 
    # completes without errors.


def test_get_short_name():
    """Test the get_short_name utility function."""
    test_cases = [
        ("simpleName", "simpleName"),
        ("Root|Joint1", "Joint1"),
        ("Root|Parent|Child", "Child"),
        ("Namespace:Object", "Object"),
        ("Root|Namespace:Object", "Object"),
        ("Root|NS1:NS2:Object", "Object"),
        ("|Root|Joint", "Joint"),
    ]
    
    for full_name, expected in test_cases:
        assert get_short_name(full_name) == expected


def test_get_short_name_empty_string():
    """Test get_short_name function with empty string."""
    assert get_short_name("") == ""