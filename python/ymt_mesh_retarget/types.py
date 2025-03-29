"""Project-wide type definitions."""

from typing import Union

import numpy as np


try:
    from numpy.typing import NDArray
except ImportError:
    # For compatibility with older NumPy versions
    from typing import Any as NDArray

from maya.api import OpenMaya as om


# Common type aliases
MeshPath = Union[om.MDagPath, str]
VertexArray = NDArray[np.float64]
JointWeights = NDArray[np.float64]
PointList = Union[list[om.MPoint], list[list[float]], NDArray[np.float64]]
IndexArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


# Type conversion functions
def to_mpoint(point: np.ndarray) -> om.MPoint:
    """Convert NumPy array to Maya MPoint.

    Args:
        point: NumPy array with 3 elements representing x, y, z coordinates

    Returns:
        Maya MPoint object
    """
    if len(point) < 3:
        raise ValueError(f"Expected array with at least 3 elements, got {len(point)}")
    return om.MPoint(float(point[0]), float(point[1]), float(point[2]))


def to_ndarray(point: om.MPoint) -> np.ndarray:
    """Convert Maya MPoint to NumPy array.

    Args:
        point: Maya MPoint object

    Returns:
        NumPy array with x, y, z coordinates
    """
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def to_mvector(vector: np.ndarray) -> om.MVector:
    """Convert NumPy array to Maya MVector.

    Args:
        vector: NumPy array with 3 elements representing x, y, z components

    Returns:
        Maya MVector object
    """
    if len(vector) < 3:
        raise ValueError(f"Expected array with at least 3 elements, got {len(vector)}")
    return om.MVector(float(vector[0]), float(vector[1]), float(vector[2]))


def to_ndarray_from_vector(vector: om.MVector) -> np.ndarray:
    """Convert Maya MVector to NumPy array.

    Args:
        vector: Maya MVector object

    Returns:
        NumPy array with x, y, z components
    """
    return np.array([vector.x, vector.y, vector.z], dtype=np.float64)


def ensure_list(array_like: Union[list, np.ndarray]) -> list:
    """Convert NumPy array to list if needed.

    Args:
        array_like: NumPy array or list

    Returns:
        List object
    """
    if isinstance(array_like, np.ndarray):
        return array_like.tolist()
    return array_like
