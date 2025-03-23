"""Module for utility functions for working with OpenMaya."""

import functools
import time
from typing import Any, Callable, Optional, TypeVar, Union

from scipy.sparse import (
    lil_matrix,
)

import numpy as np
from maya import cmds
from maya.api import (
    OpenMaya as om,
)
from maya.api import (
    OpenMayaAnim as oma,
)

from .logger import logger
from .types import IntArray, MeshPath, VertexArray


RT = TypeVar("RT")


##############################################################################
# decorators
##############################################################################
def timeit(func: Callable[..., RT]) -> Callable[..., RT]:
    """Decorator to measure and log the execution time of a function.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> RT:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Execution time of {func.__name__}: {end_time - start_time:.3f} seconds")
        return result

    return wrapper


def viewport_off(func: Callable[..., RT]) -> Callable[..., RT]:
    """Decorator - Turn off Maya display while func is running.

    If func will fail, the error will be raised after.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with viewport disabled during execution
    """

    @functools.wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> RT:
        # Turn $gMainPane Off:
        from maya import cmds, mel

        # paneLayout -manage
        gMainPane = mel.eval("global string $gMainPane; $temp = $gMainPane;")  # noqa: N806
        cmds.paneLayout(gMainPane, edit=True, manage=False)

        # ogs
        ogs_paused = cmds.ogs(query=True, pause=True)
        if not ogs_paused:
            cmds.ogs(pause=True)

        # refresh
        cmds.refresh(suspend=True)

        try:
            return func(*args, **kwargs)

        except Exception:
            import traceback

            traceback.print_stack()
            traceback.print_exc()
            raise

        finally:
            cmds.paneLayout(gMainPane, edit=True, manage=True)
            if not ogs_paused:
                cmds.ogs(pause=True)
            cmds.refresh(suspend=False)

    return wrap


def one_undo(func: Callable[..., RT]) -> Callable[..., RT]:
    """Puts the wrapped function into a single Maya Undo action.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with undo chunk handling
    """

    @functools.wraps(func)
    def _undofunc(*args: Any, **kwargs: Any) -> RT:
        import maya.cmds as cmds

        try:
            # start an undo chunk
            cmds.undoInfo(ock=True)
            return func(*args, **kwargs)
        finally:
            # after calling the func, end the undo chunk and undo
            cmds.undoInfo(cck=True)
            # cmds.undo()

    return _undofunc


def autokey_off(func: Callable[..., RT]) -> Callable[..., RT]:
    """Decorator - Turn off AutoKey while func is running.

    If func will fail, the error will be raised after.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with autokey disabled during execution
    """

    @functools.wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> RT:
        import maya.mel as mel  # pylint: disable=unused-import  # noqa
        import maya.cmds as cmds

        current = cmds.autoKeyframe(query=True, state=True)
        if not isinstance(current, bool):
            raise Exception("could not get current frame by cmds.autoKeyframe")

        try:
            cmds.autoKeyframe(state=False)
            return func(*args, **kwargs)

        except Exception:
            import traceback

            traceback.print_stack()
            traceback.print_exc()
            raise

        finally:
            cmds.autoKeyframe(state=current)

    return wrap


##############################################################################
# utility functions for OpenMaya
##############################################################################
def get_bounding_box(mesh_path: MeshPath) -> om.MBoundingBox:
    """Get the bounding box of the given mesh.

    Args:
        mesh_path: The target mesh DAG path

    Returns:
        Maya bounding box object

    Raises:
        ValueError: If invalid mesh name is provided
    """
    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError(f"Invalid mesh name: {mesh_path}")
        mesh_path = res

    mesh_fn = om.MFnMesh(mesh_path)
    bbox = mesh_fn.boundingBox
    return bbox


def get_mesh_fn(name: MeshPath) -> om.MFnMesh:
    """Get the MFnMesh object of the given mesh name.

    Args:
        name: Maya mesh path or name

    Returns:
        Maya mesh function set

    Raises:
        ValueError: If invalid mesh name is provided
    """
    if isinstance(name, str):
        res = get_mesh_dag(name)
        if not res:
            raise ValueError(f"Invalid mesh name: {name}")
        name = res

    return om.MFnMesh(name)


def get_mesh_dag(name: str) -> Optional[om.MDagPath]:
    """Get the MDagPath object of the given mesh name.

    Args:
        name: Maya mesh name

    Returns:
        Maya DAG path object or None if not found
    """
    if cmds.nodeType(name) == "transform":
        mesh = cmds.listRelatives(name, shapes=True, fullPath=True)
        if not mesh:
            return None
        dag = get_dag_path(mesh[0])
    else:
        dag = get_dag_path(name)

    return dag


def get_dag_path(node: str) -> om.MDagPath:
    """Get the DAG path of the given node.

    Args:
        node: Maya node name

    Returns:
        Maya DAG path object
    """
    selection_list = om.MSelectionList()
    selection_list.add(node)
    return selection_list.getDagPath(0)


def convert_points_to_numpy(mesh_path: MeshPath, sampling_stride: int = 1) -> VertexArray:
    """Convert mesh vertices to a numpy array.

    Args:
        mesh_path: Maya mesh path or name
        sampling_stride: Sampling stride for vertex processing (default: 1)

    Returns:
        NumPy array of vertex coordinates

    Raises:
        ValueError: If invalid mesh name is provided
    """
    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError(f"Invalid mesh name: {mesh_path}")
        mesh_path = res

    mesh_fn = get_mesh_fn(mesh_path)
    points = mesh_fn.getPoints()
    sparse_points = points[::sampling_stride]
    return np.array([[p.x, p.y, p.z] for p in sparse_points], dtype=np.float64)


def get_skin_cluster(mesh_path: om.MDagPath) -> oma.MFnSkinCluster:
    """Get the skin cluster for the given mesh.

    Args:
        mesh_path: The mesh to retrieve the skin cluster from

    Returns:
        The skin cluster function set

    Raises:
        ValueError: If mesh has no history or no skin cluster is found
    """
    histories = cmds.listHistory(mesh_path.fullPathName())
    if not isinstance(histories, list):
        raise ValueError(f"mesh_path must have a history: {mesh_path.fullPathName()}")

    skin_path = [h for h in histories if cmds.nodeType(h) == "skinCluster"]
    if not skin_path:
        raise ValueError(f"No skin cluster found for the mesh {mesh_path.fullPathName()}")

    sel_list = om.MSelectionList()
    for p in skin_path:
        sel_list.add(p)

    depend = om.MFnDependencyNode(sel_list.getDependNode(0))
    skin_cluster = oma.MFnSkinCluster(depend.object())

    return skin_cluster


def get_skin_weight_as_sparse_matrix(mesh_path: MeshPath) -> lil_matrix:
    """Retrieve the skinning weights for the given mesh.

    This function extracts the skinning weights from the skin cluster and
    returns them as a sparse matrix.

    Args:
        mesh_path: The mesh to retrieve the weights from

    Returns:
        A sparse matrix containing the skinning weights for the vertices

    Raises:
        ValueError: If invalid mesh name is provided
    """
    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError(f"Invalid mesh name: {mesh_path}")
        mesh_path = res

    skin_fn = get_skin_cluster(mesh_path)

    single_id_component = om.MFnSingleIndexedComponent()
    vertex_component = single_id_component.create(om.MFn.kMeshVertComponent)
    weights, num_influence = skin_fn.getWeights(mesh_path, vertex_component)

    # Convert the weights to a numpy array, reshaping to match the vertex count
    np_weights = np.array(weights).reshape(-1, num_influence)
    sparse_weights = lil_matrix(np_weights.shape, dtype=np.float32)

    # Set values in lil_matrix row by row to avoid index setting errors
    for i in range(np_weights.shape[0]):
        non_zero_indices = np_weights[i].nonzero()[0]  # Get the non-zero indices for the row
        for j in non_zero_indices:
            sparse_weights[i, j] = np_weights[i, j]

    return sparse_weights


def set_points(mesh: MeshPath, points: Union[list[om.MPoint], om.MPointArray]) -> None:
    """Set the deformed points to the mesh.

    Args:
        mesh: Maya mesh path or name
        points: List of points or MPointArray to set on the mesh
    """
    mesh_fn = get_mesh_fn(mesh)
    mesh_fn.setPoints(points)


##############################################################################
def select_vertices(mesh_paths: Union[list[MeshPath], MeshPath], vertices: IntArray) -> None:
    """Select the given vertices on the given meshes.

    Args:
        mesh_paths: Maya mesh path(s) or name(s)
        vertices: Array of vertex indices to select
    """
    if isinstance(mesh_paths, (om.MDagPath, str)):
        mesh_paths = [mesh_paths]

    vertex_path = []
    vertex_offset = 0

    for mesh in mesh_paths:
        mesh_fn = get_mesh_fn(mesh)
        num_vertices = mesh_fn.numVertices

        # Select vertices within the current mesh's range
        valid_vertices = (
            vertices[(vertices >= vertex_offset) & (vertices < vertex_offset + num_vertices)] - vertex_offset
        )

        # Format valid vertices for Maya selection
        vertex_strings = [f"{mesh}.vtx[{v}] " for v in valid_vertices]
        vertex_path.extend(vertex_strings)

        vertex_offset += num_vertices

    cmds.select(mesh_paths)
    cmds.select(vertex_path, add=True)
    cmds.selectMode(component=True)


##############################################################################
def calculate_threshold_distance(mesh_paths: Union[list[MeshPath], MeshPath], threshold_ratio: float) -> float:
    """Returns dbox * threshold_ratio.

    dbox is the target mesh bounding box diagonal length.

    Args:
        mesh_paths: Maya mesh path(s) or name(s)
        threshold_ratio: Ratio to multiply bounding box diagonal length by

    Returns:
        Threshold distance value

    Raises:
        ValueError: If invalid mesh name is provided
    """
    if isinstance(mesh_paths, (om.MDagPath, str)):
        mesh_paths = [mesh_paths]

    bbox = None
    for path in mesh_paths:
        if not bbox:
            bbox = get_bounding_box(path)
        else:
            bbox.expand(get_bounding_box(path))

    if not bbox:
        raise ValueError("Invalid mesh name")

    bbox_min = bbox.min
    bbox_max = bbox.max
    bbox_diag = bbox_max - bbox_min
    bbox_diag_length = bbox_diag.length()

    threshold_distance = bbox_diag_length * threshold_ratio

    return threshold_distance


##############################################################################
def restructure_meshes_hierarchy(suffix: str = "retarget", targets: Optional[list[str]] = None) -> None:
    """Restructure meshes hierarchy by adding suffix to parent node names.

    Args:
        suffix: Suffix to add to parent node names
        targets: List of target meshes or None to use selected objects
    """
    if not targets:
        targets = cmds.ls(sl=True, type="transform", long=True)

    for mesh in cmds.ls(targets, type="transform", long=True):
        depth = len(mesh.split("|"))
        parent = None
        for i in range(depth - 1):
            parts = mesh.split("|")
            new_parts = parts.copy()
            new_parts[1] = "_".join([parts[1], suffix])
            parent_name = "|".join(new_parts[: i + 1])

            if not parent_name:
                continue

            if not cmds.ls(parent_name):
                loc = cmds.createNode("transform", name=new_parts[i])

                parent = cmds.parent(loc, parent)[0] if parent else loc
            else:
                parent = parent_name

        if parent:
            mesh = cmds.parent(mesh, parent)[0]


def get_short_name(name: str) -> str:
    """Get the short name of the given name."""
    return name.split("|")[-1].split(":")[-1].split("|")[-1]
