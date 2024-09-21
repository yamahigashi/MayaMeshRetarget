# -*- coding: utf-8 -*-
"""
Module for utility functions for working with OpenMaya.
"""
import time

import numpy as np
from scipy.sparse import (
    lil_matrix,
)

from maya.api import (
    OpenMaya as om,
    OpenMayaAnim as oma,
)
from maya import (
    cmds
)

from logging import (
    getLogger,
    WARN,  # noqa: F401
    DEBUG,  # noqa: F401
    INFO,  # noqa: F401
)
logger = getLogger(__name__)
logger.setLevel(INFO)
logger.setLevel(DEBUG)



##############################################################################
# decorators
##############################################################################
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug("Execution time of {func.__name__}: {elapsed:.3f} seconds".format(
            func=func,
            elapsed=(end_time - start_time)
        ))
        return result
    return wrapper


##############################################################################
# utility functions for OpenMaya
##############################################################################
def get_bounding_box(mesh_path):
    # type: (om.MDagPath|str) -> om.MBoundingBox
    """
    Get the bounding box of the given mesh.
    
    :param mesh_path: The target mesh DAG path
    :return: A tuple of two points representing the minimum and maximum corners of the bounding box
    """

    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError("Invalid mesh name")
        mesh_path = res

    mesh_fn = om.MFnMesh(mesh_path)
    bbox = mesh_fn.boundingBox
    return bbox


def get_mesh_fn(name):
    # type: (str|om.MDagPath) -> om.MFnMesh
    """Get the MFnMesh object of the given mesh name."""

    if isinstance(name, str):
        res = get_mesh_dag(name)
        if not res:
            raise ValueError("Invalid mesh name")
        name = res

    return om.MFnMesh(name)


def get_mesh_dag(name):
    # type: (str) -> om.MDagPath|None
    """Get the MFnMesh object of the given mesh name."""

    if cmds.nodeType(name) == "transform":
        mesh = cmds.listRelatives(name, shapes=True, fullPath=True)
        if not mesh:
            return None
        dag = get_dag_path(mesh[0])
    else:
        dag = get_dag_path(name)

    return dag


def get_dag_path(node):
    # type: (str) -> om.MDagPath
    """Get the dag path of the given node."""
    selection_list = om.MSelectionList()
    selection_list.add(node)
    return selection_list.getDagPath(0)


def convert_points_to_numpy(mesh_path, sampling_stride=1):
    # type: (om.MDagPath|str, int) -> np.ndarray
    """Convert mesh vertices to a numpy array."""

    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError("Invalid mesh name")
        mesh_path = res

    mesh_fn = get_mesh_fn(mesh_path)
    points = mesh_fn.getPoints()
    sparse_points = points[::sampling_stride]
    return np.array([[p.x, p.y, p.z] for p in sparse_points])


def get_skin_cluster(mesh_path):
    # type: (om.MDagPath) -> oma.MFnSkinCluster
    """
    Get the skin cluster for the given mesh.
    
    :param mesh: The mesh to retrieve the skin cluster from
    :return: The skin cluster object or None if not found
    """

    hostories = cmds.listHistory(mesh_path.fullPathName())
    if not isinstance(hostories, list):
        raise ValueError("mesh_path must have a history: {}".format(mesh_path.fullPathName()))

    skin_path = [h for h in hostories if cmds.nodeType(h) == "skinCluster"]  # type: ignore
    if not skin_path:
        raise ValueError("No skin cluster found for the mesh {}".format(mesh_path.fullPathName()))

    sel_list = om.MSelectionList()
    for p in skin_path:
        sel_list.add(p)
   
    depend = om.MFnDependencyNode(sel_list.getDependNode(0))
    skin_cluster = oma.MFnSkinCluster(depend.object())

    return skin_cluster


def get_skin_weight_as_sparse_matrix(mesh_path):
    # type: (om.MDagPath|str) -> lil_matrix
    """
    Retrieve the skinning weights for the given mesh.
    This function should extract the skinning weights from the skin cluster and
    return them as a sparse matrix.
    
    :param mesh: The mesh to retrieve the weights from
    :return: A sparse matrix containing the skinning weights for the vertex
    """

    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError("Invalid mesh name")
        mesh_path = res
   
    skin_fn = get_skin_cluster(mesh_path)

    single_id_component = om.MFnSingleIndexedComponent()
    vertex_component = single_id_component.create(om.MFn.kMeshVertComponent)  # type: ignore
    weights, num_influence = skin_fn.getWeights(mesh_path, vertex_component)

    # Convert the weights to a numpy array, reshaping to match the vertex count
    np_weights = np.array(weights).reshape(-1, num_influence)
    sparse_weights = lil_matrix(np_weights.shape, dtype=np.float32)
    non_zero_indices = np.nonzero(weights)

    # Set values in lil_matrix row by row to avoid index setting errors
    for i in range(np_weights.shape[0]):
        non_zero_indices = np_weights[i].nonzero()[0]  # Get the non-zero indices for the row
        for j in non_zero_indices:
            sparse_weights[i, j] = np_weights[i, j]

    return sparse_weights


def set_points(mesh, points):
    # type: (str, list[om.MPoint]) -> None
    """Set the deformed points to the mesh."""
    mesh_fn = get_mesh_fn(mesh)
    mesh_fn.setPoints(points)


##############################################################################
def select_vertices(mesh_paths, vertices):
    # type: (list[om.MDagPath]|om.MDagPath, np.ndarray) -> None
    """Select the given vertices on the given meshes."""

    if isinstance(mesh_paths, om.MDagPath):
        mesh_paths = [mesh_paths]

    vertex_path = []
    vertex_offset = 0

    for mesh in mesh_paths:
        mesh_fn = get_mesh_fn(mesh)
        num_vertices = mesh_fn.numVertices

        # Select vertices within the current mesh's range
        valid_vertices = vertices[
                (vertices >= vertex_offset) &
                (vertices < vertex_offset + num_vertices)
        ] - vertex_offset

        # Format valid vertices for Maya selection
        vertex_strings = ["{}.vtx[{}] ".format(mesh, v) for v in valid_vertices]
        vertex_path.extend(vertex_strings)

        vertex_offset += num_vertices

    cmds.select(mesh_paths)
    cmds.select(vertex_path, add=True)
    cmds.selectMode(component=True)


##############################################################################
def calculate_threshold_distance(mesh_paths, threadhold_ratio):
    # type: (list[om.MDagPath]|om.MDagPath, float) -> float
    """Returns dbox * threadhold_ratio.

    dbox is the target mesh bounding box diagonal length.
    """
    if isinstance(mesh_paths, om.MDagPath):
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

    threshold_distance = bbox_diag_length * threadhold_ratio

    return threshold_distance


##############################################################################
def restructure_meshes_hierarchy(suffix="retarget", targets=None):
    # type: (str, None|list[str]) -> None
    """Restructure meshes hierarchy."""
    if not targets:
        targets = cmds.ls(sl=True, type="transform", long=True)

    for mesh in cmds.ls(targets, type="transform", long=True):

        depth = len(mesh.split("|"))
        parent = None
        for i in range(depth - 1):
            parts = mesh.split("|")
            new_parts = parts
            new_parts[1] = "_".join([parts[1], suffix])
            parent_name = "|".join(new_parts[:i + 1])

            if not parent_name:
                continue

            if not cmds.ls(parent_name):
                loc = cmds.createNode("transform", name=new_parts[i])

                if parent:
                    parent = cmds.parent(loc, parent)[0]
                else:
                    parent = loc
            else:
                parent = parent_name

        if parent:
            mesh = cmds.parent(mesh, parent)[0]
