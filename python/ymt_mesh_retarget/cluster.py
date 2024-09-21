# -*- coding: utf-8 -*-
"""
Module for clustering vertices based on skin weights and topological adjacency.
"""
import time
import collections

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import (
    lil_matrix,
)

from maya.api import (
    OpenMaya as om,
)
from maya import (
    cmds,
    mel,
)

from . import util


##############################################################################
# Cluster vertices across multiple meshes
##############################################################################
@util.timeit
def cluster_vertices_by_skin_weight(mesh_paths, precision=3, min_vertices_per_cluster=6):
    # type: (list[om.MDagPath|str], float, int) -> np.ndarray
    """
    Cluster vertices based on their weight similarity across multiple meshes using a custom distance function.

    :param mesh_paths: A list of meshes to cluster
    :param precision: The number of decimal places to round the skin weights (default: 3)
    :param min_vertices_per_cluster: The minimum number of vertices required for a cluster (default: 6)
    :return: A list of cluster labels for each vertex across all meshes
    """

    # Initialize variables to hold combined data
    all_sparse_weights = []
    vertex_offsets = []
    vertex_offset = 0

    # Process each mesh and collect sparse weights
    for mesh_path in mesh_paths:

        try:
            sparse_weights = util.get_skin_weight_as_sparse_matrix(mesh_path)
        except ValueError:
            # If the mesh has no skin cluster, create an empty sparse matrix
            mesh = util.get_mesh_fn(mesh_path)
            num_vertices = mesh.numVertices
            sparse_weights = lil_matrix((num_vertices, 0))

        num_vertices = sparse_weights.shape[0]
        all_sparse_weights.append(sparse_weights)
        vertex_offsets.append(vertex_offset)
        vertex_offset += num_vertices

    # Combine all sparse weights into one matrix
    total_vertices = vertex_offset
    num_joints = max(sparse.shape[1] for sparse in all_sparse_weights)
    combined_sparse_weights = lil_matrix((total_vertices, num_joints))

    current_vertex = 0
    for sparse_weights in all_sparse_weights:
        num_vertices = sparse_weights.shape[0]
        combined_sparse_weights[current_vertex:current_vertex + num_vertices, :sparse_weights.shape[1]] = sparse_weights
        current_vertex += num_vertices

    # Convert to CSR format for efficient row operations
    csr_weights = combined_sparse_weights.tocsr()

    # Round the non-zero data to the specified precision
    csr_weights.data = np.round(csr_weights.data, precision)

    # Create a unique key for each vertex based on its non-zero indices and rounded weights
    # Use a hashable representation (tuple of indices and weights)
    # Since we have variable-length data per row, we'll represent each row as a tuple of (indices, weights)

    # Get the indices where rows have non-zero entries
    non_zero_row_indices = np.diff(csr_weights.indptr) > 0
    vertex_indices = np.where(non_zero_row_indices)[0]

    # Extract the data for non-zero rows
    indptr = csr_weights.indptr
    indices = csr_weights.indices
    data = csr_weights.data

    # Create a list to hold the keys for each vertex
    vertex_keys = []

    for idx in vertex_indices:
        start = indptr[idx]
        end = indptr[idx + 1]
        joint_indices = indices[start:end]
        weight_values = data[start:end]

        # Create a tuple of (joint_indices, weight_values)
        # For consistent ordering, sort by joint_indices
        sorted_order = np.argsort(joint_indices)
        joint_indices = joint_indices[sorted_order]
        weight_values = weight_values[sorted_order]
        key = tuple(zip(joint_indices, weight_values))
        vertex_keys.append(key)

    # Map keys to vertex indices
    key_to_vertices = collections.defaultdict(list)
    for vertex_index, key in zip(vertex_indices, vertex_keys):
        key_to_vertices[key].append(vertex_index)

    # Initialize labels
    labels = np.full(total_vertices, -1, dtype=int)
    cluster_id = 0
    for indices in key_to_vertices.values():
        if len(indices) > min_vertices_per_cluster:
            labels[indices] = cluster_id
            cluster_id += 1

    return labels


@util.timeit
def refine_clusters_by_topology(mesh_paths, labels, tolerance=1e-6):
    # type: (list[om.MDagPath|str], np.ndarray, float) -> np.ndarray
    """
    Refine clusters by checking topological adjacency across multiple meshes.

    :param mesh_paths: A list of meshes for which to refine the clusters
    :param labels: Initial cluster labels based on weight similarity
    :return: Refined cluster labels taking topology into account
    """
    total_vertices = labels.shape[0]
    adjacency_matrix = lil_matrix((total_vertices, total_vertices), dtype=bool)
    all_points = []
    vertex_offsets = []
    vertex_offset = 0

    bar = mel.eval("$tmp = $gMainProgressBar")

    # Collect all points and build adjacency matrix
    for mesh_path in mesh_paths:
        mesh_fn = util.get_mesh_fn(mesh_path)
        num_vertices = mesh_fn.numVertices
        points = util.convert_points_to_numpy(mesh_path)
        all_points.append(points)
        vertex_offsets.append(vertex_offset)

        # Build local adjacency matrix
        vertex_iter = om.MItMeshVertex(mesh_path)
        while not vertex_iter.isDone():
            local_index = vertex_iter.index()
            global_index = local_index + vertex_offset
            connected_vertices = vertex_iter.getConnectedVertices()
            connected_vertices = [idx + vertex_offset for idx in connected_vertices]
            adjacency_matrix[global_index, connected_vertices] = True
            vertex_iter.next()

        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=num_vertices, status=f"Collecting to connectivities for {mesh_path}...")

        vertex_offset += num_vertices

    # Combine all points into a single array
    all_points = np.vstack(all_points)

    # Also check for vertices that are in the same position (within the tolerance)
    tree = cKDTree(all_points)
    pairs = tree.query_pairs(r=tolerance)

    for i, j in pairs:
        adjacency_matrix[i, j] = True
        adjacency_matrix[j, i] = True

    # Array to mark visited vertices
    visited = np.full(total_vertices, False)

    # Function to check if all neighbors of a vertex are within the same cluster
    def is_fully_connected_within_cluster(vertex, current_cluster):
        neighbors = adjacency_matrix[vertex].nonzero()[1]
        return np.all(labels[neighbors] == current_cluster)

    # Function to perform breadth-first search (BFS) to find connected components
    def bfs(start_vertex, current_cluster):
        queue = [start_vertex]
        cluster_vertices = []
        dissolve = False  # Flag to determine if the cluster should be dissolved

        while queue:
            vertex = queue.pop(0)
            if visited[vertex]:
                continue

            visited[vertex] = True
            cluster_vertices.append(vertex)

            # Check if all adjacent vertices are in the same cluster
            if not is_fully_connected_within_cluster(vertex, current_cluster):
                dissolve = True  # Mark the cluster for dissolution

            # Get all connected vertices (where adjacency_matrix is True)
            neighbors = adjacency_matrix[vertex].nonzero()[1]
            queue.extend(neighbors[(labels[neighbors] == current_cluster) & (~visited[neighbors])])

        return cluster_vertices, dissolve

    # Refine clusters based on adjacency and dissolve clusters if necessary
    # Initialize refined labels (starting with the existing labels)
    cluster_id = 0
    refined_labels = np.full_like(labels, -1)

    for vertex_index in range(total_vertices):
        if visited[vertex_index] or labels[vertex_index] == -1:
            continue

        # Perform BFS to find all connected vertices in the same cluster
        current_cluster = labels[vertex_index]
        connected_vertices, dissolve = bfs(vertex_index, current_cluster)

        # If the cluster should not be dissolved, assign a new cluster ID
        if not dissolve and len(connected_vertices) > 1:
            refined_labels[connected_vertices] = cluster_id
            cluster_id += 1
        else:
            # Cluster is dissolved, all vertices are marked as -1 (isolated)
            refined_labels[connected_vertices] = -1

        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1, status="Refining clusters...")

    return refined_labels


@util.timeit
def cluster_vertices(mesh_paths):
    # type: (list[om.MDagPath|str]) -> np.ndarray
    """
    Cluster vertices across multiple meshes.

    :param mesh_paths: A list of meshes to cluster
    :return: A list of cluster labels for each vertex across all meshes
    """

    vertex_offset = 0

    for mesh_path in mesh_paths:
        mesh_fn = util.get_mesh_fn(mesh_path)
        num_vertices = mesh_fn.numVertices
        vertex_offset += num_vertices

    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                status="Clustering vertices...",
                maxValue=vertex_offset * 4,
        )

    labels = cluster_vertices_by_skin_weight(mesh_paths)
    refined_labels = refine_clusters_by_topology(mesh_paths, labels)

    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)

    return refined_labels
