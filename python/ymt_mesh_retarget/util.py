"""Module for utility functions for working with OpenMaya."""

import functools
import time
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np
from maya import cmds
from maya.api import (
    OpenMaya as om,
)
from maya.api import (
    OpenMayaAnim as oma,
)
from numpy.typing import NDArray
from scipy.sparse import (
    lil_matrix,
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
    mat = mesh_path.inclusiveMatrix()
    bbox_min = bbox.min * mat
    bbox_max = bbox.max * mat
    return om.MBoundingBox(bbox_min, bbox_max)


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
    points = mesh_fn.getPoints(om.MSpace.kWorld)
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


def get_inverse_bind_matrix(mesh_path: MeshPath) -> dict[str, np.ndarray]:
    """Get the inverse bind matrix for the given mesh.

    Args:
        mesh_path: Maya mesh path or name

    Returns:
        Inverse bind matrix as a NumPy array
    """
    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError(f"Invalid mesh name: {mesh_path}")
        mesh_path = res

    skin_fn = get_skin_cluster(mesh_path)
    joint_dags = skin_fn.influenceObjects()
    num_joints = len(joint_dags)

    results = {}

    # Get the inverse bind matrices for each joint
    # ibm = np.zeros((num_joints, 4, 4), dtype=np.float32)
    for i in range(num_joints):
        inv_mat = cmds.getAttr(f"{skin_fn.name()}.bindPreMatrix[{i}]")
        # ibm[i] = np.array(inv_mat).reshape(4, 4)
        results[joint_dags[i].fullPathName()] = np.array(inv_mat).reshape(4, 4)

    return results


def set_points(mesh: MeshPath, points: Union[list[om.MPoint], om.MPointArray]) -> None:
    """Set the deformed points to the mesh.

    Args:
        mesh: Maya mesh path or name
        points: List of points or MPointArray to set on the mesh
    """
    mesh_fn = get_mesh_fn(mesh)
    logger.debug(f"Setting {len(points)} points on {mesh} type of points: {type(points)}")

    if isinstance(points, list):
        if isinstance(points[0], om.MPoint):
            points = om.MPointArray(points)

        elif isinstance(points[0], list):
            points = om.MPointArray([om.MPoint(p[0], p[1], p[2]) for p in points])

        elif isinstance(points[0], np.ndarray):
            p = np.array(points)
            points = om.MPointArray([om.MPoint(p[i, 0], p[i, 1], p[i, 2]) for i in range(p.shape[0])])

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


def get_hierarchy(nodes: list[str]) -> list[str]:
    """Get a sorted list of unique parent paths (hierarchy) including the nodes.

    - The list is sorted by hierarchy depth.
    - The list includes the nodes themselves

    Args:
        nodes: List of nodes to build trees from

    Returns:
        List of nodes in tree structure sorted by hierarchy depth and sibling order
    """

    def get_sibling_order(node: str) -> int:
        parent = cmds.listRelatives(node, parent=True, fullPath=True) or []
        if not parent:
            return 0

        siblings = cmds.listRelatives(parent[0], fullPath=True) or []
        return siblings.index(node)

    candidates = []
    for node in nodes:

        try:
            full_path = cmds.ls(node, long=True)[0]
        except IndexError:
            logger.error(f"Node not found: {node}")
            continue

        depth = len(full_path.split("|"))
        dag_order = get_sibling_order(full_path)

        candidates.append((full_path, depth, dag_order))

        for i in range(2, len(full_path.split("|"))):
            parent_path = "|".join(full_path.split("|")[:i])
            entry = (parent_path, i, get_sibling_order(parent_path))
            if entry not in candidates:
                candidates.append(entry)


    # sort by depth and dag order
    hierarchy = [c[0] for c in sorted(candidates, key=lambda x: (x[1], x[2]))]

    return hierarchy


def get_skin_weights(mesh_path: MeshPath) -> tuple[list[list[float]], list[str]]:
    """Get the skin weights for the given mesh.

    Args:
        mesh_path: Maya mesh path or name

    Returns:
        Tuple of vertex weights and joint names
            - List of vertex weights for each joint
            - List of joint names
    """

    if isinstance(mesh_path, str):
        res = get_mesh_dag(mesh_path)
        if not res:
            raise ValueError(f"Invalid mesh name: {mesh_path}")

        mesh_path = res

    fn_skin = get_skin_cluster(mesh_path)
    if not fn_skin:
        raise ValueError(f"No skin cluster found for mesh: {mesh_path}")

    # Get joint information
    influence_objects = fn_skin.influenceObjects()  # type: om.MDagPathArray
    num_influences = len(influence_objects)

    # Create list of joint names
    joint_names = []
    for i in range(len(influence_objects)):
        dag_path = influence_objects[i]
        joint_name = dag_path.fullPathName()
        joint_names.append(joint_name)

    # Get weights for each vertex
    mesh_fn = om.MFnMesh(mesh_path)
    num_vertices = mesh_fn.numVertices

    # Create vertex component
    vert_indices = om.MIntArray(list(range(num_vertices)))
    vert_component = om.MFnSingleIndexedComponent().create(om.MFn.kMeshVertComponent)
    om.MFnSingleIndexedComponent(vert_component).addElements(vert_indices)

    # Create influence indices
    influence_indices = om.MIntArray(list(range(len(influence_objects))))

    # Get skin weights
    weights = fn_skin.getWeights(mesh_path, vert_component, influence_indices)

    # Convert to list format
    weights_list = []
    for i in range(num_vertices):
        vertex_weights = []
        for j in range(num_influences):
            weight = weights[i * num_influences + j]
            vertex_weights.append(weight)
        weights_list.append(vertex_weights)

    return weights_list, joint_names


def decompose_matrix(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4x4 matrix into translation, rotation, and scale components.

    Fixed order of rotation is XYZ for now.

    Args:
        mat: 4x4 transformation matrix (column major)

    Returns:
        Tuple of translation, rotation, and scale components
    """
    # Translation: column majorでは最後の列に並ぶ
    tx, ty, tz = mat[0, 3], mat[1, 3], mat[2, 3]

    # Scale: column majorなので、各列（0〜2列目）の先頭3要素のノルムを取る
    sx = np.linalg.norm(mat[:3, 0])
    sy = np.linalg.norm(mat[:3, 1])
    sz = np.linalg.norm(mat[:3, 2])

    # 回転部分の正規化（各列をスケールで割る）
    rot_mat = mat.copy()
    if abs(sx) > 1e-8:
        rot_mat[:3, 0] /= sx
    if abs(sy) > 1e-8:
        rot_mat[:3, 1] /= sy
    if abs(sz) > 1e-8:
        rot_mat[:3, 2] /= sz

    # 回転行列部分（上3×3）を抽出
    R = rot_mat[:3, :3]  # noqa: N806

    # Euler角（XYZ順）の抽出
    ry = np.arcsin(-R[2, 0])
    cy = np.cos(ry)
    if abs(cy) > 1e-4:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # ジンバルロック時のフォールバック
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0

    # ラジアン→度変換
    rx_deg = np.degrees(rx)
    ry_deg = np.degrees(ry)
    rz_deg = np.degrees(rz)

    translation = np.array([tx, ty, tz], dtype=np.float64)
    rotation = np.array([rx_deg, ry_deg, rz_deg], dtype=np.float64)
    scale = np.array([sx, sy, sz], dtype=np.float64)

    return translation, rotation, scale


def compose_matrix(
    t: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Compose a 4x4 transformation matrix from translation, rotation, and scale components.

    Args:
        t: Translation vector (3,)
        r: Rotation vector (3,) in degrees
        s: Scale vector (3,)

    Returns:
        4x4 transformation matrix (column major)
    """
    rx, ry, rz = np.radians(r)

    # 各軸の回転行列 (3x3 部分) - column major形式
    Rx = np.array([  # noqa: N806
        [1, 0, 0],
        [0, np.cos(rx), np.sin(rx)],
        [0, -np.sin(rx), np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([  # noqa: N806
        [np.cos(ry), 0, -np.sin(ry)],
        [0, 1, 0],
        [np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([  # noqa: N806
        [np.cos(rz), np.sin(rz), 0],
        [-np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # 合成した回転行列（3x3）
    # 回転の合成順序：Z→Y→X（右から掛ける）
    Rm = Rz @ Ry @ Rx  # noqa: N806

    # column majorでは、scaleは各列に掛かる
    M_linear = np.zeros((3, 3), dtype=np.float64)  # noqa: N806
    for i in range(3):
        M_linear[:, i] = Rm[:, i] * s[i]

    # 4x4行列の作成
    M = np.eye(4, dtype=np.float64)  # noqa: N806
    M[:3, :3] = M_linear
    M[:3, 3] = t

    return M
