# -*- coding: utf-8 -*-
"""
Module for inpainting unconvinced distances between two meshes calculated by
RBF kernel using Laplacian matrix.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import (
    lil_matrix,
    csr_matrix,
    dia_matrix,  # noqa: F401
    block_diag as spblock_diag,
    diags as spdiags,
    linalg as splinalg,
)

from maya.api import (
    OpenMaya as om,
)

from . import util


##############################################################################
# Distance Inpainting
##############################################################################
def segregate_vertices_by_confidence(
        src_path,
        dst_paths,
        threshold_dist_coefficient=0.1,
        threshold_angle=180.0,
):
    # type: (om.MDagPath, list[om.MDagPath]|om.MDagPath, float, float) -> tuple[np.ndarray, np.ndarray]
    """segregate vertices by confidence."""

    if not isinstance(dst_paths, list):
        dst_paths = [dst_paths]

    threshold_distance = util.calculate_threshold_distance([src_path], threshold_dist_coefficient)
    target_vertex_data = __create_vertex_data_array(dst_paths)
    closest_points_data = __get_closest_points_by_kdtree(src_path, target_vertex_data)

    confident_vertex_indices = __filter_high_confidence_matches(
            target_vertex_data,
            closest_points_data,
            threshold_distance,
            threshold_angle)

    unconvinced_vertex_indices = np.setdiff1d(
            np.arange(target_vertex_data.shape[0]),
            confident_vertex_indices)

    return confident_vertex_indices, unconvinced_vertex_indices


@util.timeit
def __inpaint_distance_matrix(mesh_paths, D, known_indices, unknown_indices):
    # type: (list[om.MDagPath], np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    """apply inpainting for indices."""

    all_L = []
    all_M = []

    for mesh_path in mesh_paths:
        mesh = util.get_mesh_fn(mesh_path)
        _L, _M = __compute_laplacian_and_mass_matrix(mesh)
        all_L.append(_L)
        all_M.append(_M)

    L = spblock_diag(all_L)
    M = spblock_diag(all_M)
    M_diag = np.clip(M.diagonal(), 1e-8, None)

    Q = -L + L @ spdiags(np.reciprocal(M_diag)) @ L

    S_match = known_indices
    S_nomatch = unknown_indices

    Q_UU = csr_matrix(Q[np.ix_(S_nomatch, S_nomatch)])
    Q_UI = csr_matrix(Q[np.ix_(S_nomatch, S_match)])

    rank = np.linalg.matrix_rank(Q_UU.toarray())
    if rank < Q_UU.shape[0]:
        print(f"Rank of Q_UU: {rank}, expected full rank: {Q_UU.shape[0]}")
        epsilon = 1e-8
        Q_UU = Q_UU + epsilon * np.eye(Q_UU.shape[0])

    D_I = D[S_match, :]
    b = -Q_UI @ D_I
    D_U = splinalg.spsolve(Q_UU, b)

    # TODO: Is checking the rank sufficient? to be removed later
    # valid_mask = ~np.isnan(D_U)
    # D[S_nomatch][valid_mask] = D_U[valid_mask]

    D[S_nomatch] = D_U

    return D


def __create_vertex_data_array(mesh_paths):
    # type: (list[om.MDagPath]) -> np.ndarray
    """Create a structured numpy array containing vertex index, position, and normal."""

    all_vertex_data = []

    for path in mesh_paths:
        mesh = util.get_mesh_fn(path)

        vertex_data = np.zeros(
                mesh.numVertices,
                dtype=[
                    ("index", np.int64),
                    ("position", np.float64, 3),
                    ("normal", np.float64, 3),
                    ("face_index", np.int64),
                ])
     
        for i in range(mesh.numVertices):
            position = mesh.getPoint(i, om.MSpace.kWorld)  # type: ignore
            normal = mesh.getVertexNormal(i, om.MSpace.kWorld)  # type: ignore
            vertex_data[i] = (
                    i,
                    [position.x, position.y, position.z],
                    [normal.x, normal.y, normal.z],
                    -1)

        all_vertex_data.append(vertex_data)

    return np.concatenate(all_vertex_data)


def __get_closest_points_by_kdtree(source_path, target_vertex_data):
    # type: (om.MDagPath, np.ndarray) -> np.ndarray
    """get closest points and return a structured numpy array similar to target_vertex_data."""

    source_vertex_data = __create_vertex_data_array([source_path])
    B_positions = np.array([vertex["position"] for vertex in source_vertex_data])
    A_positions = np.array([vertex["position"] for vertex in target_vertex_data])

    tree = cKDTree(B_positions)
    _, indices = tree.query(A_positions)

    nearest_in_B_for_A = source_vertex_data[indices]
    return nearest_in_B_for_A


def __filter_high_confidence_matches(target_vertex_data, closest_points_data, max_distance, max_angle):
    # type: (np.ndarray, np.ndarray, float, float) -> np.ndarray
    """filter high confidence matches using structured arrays."""

    target_positions = target_vertex_data["position"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["position"]
    source_normals = closest_points_data["normal"]

    # Calculate distances (vectorized)
    distances = np.linalg.norm(source_positions - target_positions, axis=1)

    # Calculate angles between normals (vectorized)
    cos_angles = np.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles /= np.linalg.norm(source_normals, axis=1) * np.linalg.norm(target_normals, axis=1)
    cos_angles = np.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_indices = np.where((distances <= max_distance) & (angles <= max_angle))[0]

    return high_confidence_indices


def __add_laplacian_entry_in_place(L, tri_positions, tri_indices):
    # type: (lil_matrix, np.ndarray, np.ndarray) -> None
    """add laplacian entry.

    CAUTION: L is modified in-place.
    """

    i1 = tri_indices[0]
    i2 = tri_indices[1]
    i3 = tri_indices[2]

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]

    # calculate cotangent
    cotan1 = __compute_cotangent(v2, v1, v3)
    cotan2 = __compute_cotangent(v1, v2, v3)
    cotan3 = __compute_cotangent(v1, v3, v2)

    # update laplacian matrix
    L[i1, i2] += cotan1  # type: ignore
    L[i2, i1] += cotan1  # type: ignore
    L[i1, i1] -= cotan1  # type: ignore
    L[i2, i2] -= cotan1  # type: ignore

    L[i2, i3] += cotan2  # type: ignore
    L[i3, i2] += cotan2  # type: ignore
    L[i2, i2] -= cotan2  # type: ignore
    L[i3, i3] -= cotan2  # type: ignore

    L[i1, i3] += cotan3  # type: ignore
    L[i3, i1] += cotan3  # type: ignore
    L[i1, i1] -= cotan3  # type: ignore
    L[i3, i3] -= cotan3  # type: ignore


def __add_area_in_place(areas, tri_positions, tri_indices):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> None
    """add area.

    CAUTION: areas is modified in-place.
    """

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    for idx in tri_indices:
        areas[idx] += area


def __compute_laplacian_and_mass_matrix(mesh):
    # type: (om.MFnMesh) -> tuple[csr_matrix, dia_matrix]
    """compute laplacian matrix from mesh.

    treat area as mass matrix.
    """

    # initialize sparse laplacian matrix
    n_vertices = mesh.numVertices
    L = lil_matrix((n_vertices, n_vertices))
    areas = np.zeros(n_vertices)

    # for each edge and face, calculate the laplacian entry and area
    face_iter = om.MItMeshPolygon(mesh.dagPath())
    while not face_iter.isDone():

        n_tri = face_iter.numTriangles()

        for j in range(n_tri):

            tri_positions, tri_indices = face_iter.getTriangle(j)
            __add_laplacian_entry_in_place(L, tri_positions, tri_indices)
            __add_area_in_place(areas, tri_positions, tri_indices)

        face_iter.next()

    L_csr = L.tocsr()
    M_csr = spdiags(areas)

    return L_csr, M_csr


def __compute_cotangent(v1, v2, v3):
    # type: (om.MPoint, om.MPoint, om.MPoint) -> float
    """compute cotangent from three points."""

    edeg1 = v2 - v1
    edeg2 = v3 - v1

    norm1 = edeg1 ^ edeg2

    area = norm1.length()
    try:
        cotan = edeg1 * edeg2 / area
    except ZeroDivisionError:
        cotan = 0.0001

    return cotan


@util.timeit
def inpaint_distance(
        source_path,
        target_paths,
        distances,
        labels,
        threshold_dist_coefficient=0.1,
        threshold_angle=180.0,
):
    # type: (om.MDagPath, list[om.MDagPath], np.ndarray, np.ndarray, float, float) -> None
    """Inpaint the distance matrix for unconvinced vertices.
    
    This function fills in the distances for vertices that were not confidently matched and
    were marked as isolated or part of a cluster. The inpainting process uses the known
    distances from confident matches to estimate the distances for the unconvinced vertices.

    :param source_path: The source mesh DAG path
    :param target_paths: The target mesh DAG paths
    :param distances: The distance matrix between source and target vertices
    :param labels: The cluster labels for each vertex
    :param threshold_distance: The threshold distance for confident matches
    :param threshold_angle: The threshold angle for confident matches
    """

    # Segregate vertices based on confidence and inpaint distances
    tmp = segregate_vertices_by_confidence(source_path, target_paths, threshold_dist_coefficient, threshold_angle)
    confident_indices, unconvinced_indices = tmp

    # TODO: Rigid transformation for unconvinced vertices
    # # Inpaint distances for unconvinced vertices
    # if len(unconvinced_indices) > 0:
    #     distances = __inpaint_distance_matrix(
    #             target_paths,
    #             distances,
    #             confident_indices,
    #             unconvinced_indices)
    # 
    # return distances

    all_indices = np.arange(distances.shape[0])

    # Isolated vertices are those that are not part of any cluster
    isolated_indices = np.where(labels == -1)[0]
    unconvinced_iso_indices = np.intersect1d(unconvinced_indices, isolated_indices)
    confident_all_indices = np.setdiff1d(all_indices, unconvinced_iso_indices)
    if len(unconvinced_indices) > 0:
        distances = __inpaint_distance_matrix(
                target_paths,
                distances,
                confident_all_indices,
                unconvinced_iso_indices)

    return distances


def select_inpaint_area(src_path, dst_paths, threshold_distance=0.1, threshold_angle=180.0):
    # type: (om.MDagPath, list[om.MDagPath]|om.MDagPath, float, float) -> None
    """Select vertices to inpaint."""

    if not isinstance(dst_paths, list):
        dst_paths = [dst_paths]

    confident_vertex_indices, unconvinced_vertex_indices = segregate_vertices_by_confidence(
            src_path,
            dst_paths,
            threshold_distance,
            threshold_angle)
    print(f"Confident: {len(confident_vertex_indices)}")
    print(f"Unconvinced: {len(unconvinced_vertex_indices)}")

    util.select_vertices(dst_paths, unconvinced_vertex_indices)
